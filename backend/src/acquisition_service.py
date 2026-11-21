from statistics import fmean
from fastapi import HTTPException
from sqlalchemy import select,delete
from .run_models import TunnelRun
from .acquisition_models import AcquisitionBatch,ChannelReading
from .calibration_service import locked
from . import lift_service
from .services import pack,audit

def create(s,d,u):
    r=locked(s,TunnelRun,d.run_id)
    if r.state!='running':raise HTTPException(409,'只有运行中的试验可开采集批次')
    if s.scalar(select(AcquisitionBatch.id).where(AcquisitionBatch.run_id==r.id)):raise HTTPException(409,'本试验已有采集批次')
    x=AcquisitionBatch(**d.model_dump());s.add(x);s.flush();audit(s,u,'建立试验采集 '+str(x.id));return view(s,x)
def view(s,x):return pack(x)|{'readings':[pack(v) for v in s.scalars(select(ChannelReading).where(ChannelReading.batch_id==x.id).order_by(ChannelReading.probe_id))],'curve':lift_service.curve_summary(s,x.id)}
def context(s,id):
    pre=s.get(AcquisitionBatch,id)
    if not pre:raise HTTPException(404,'采集批次不存在')
    r=locked(s,TunnelRun,pre.run_id);return locked(s,AcquisitionBatch,id),r
def write(s,id,d,u):
    x,r=context(s,id)
    if x.state!='collecting' or r.state!='running':raise HTTPException(409,'仅运行中未冻结批次可写采样')
    ids=[p['id'] for p in r.snapshot['probes']];incoming=[v.probe_id for v in d.readings]
    if len(incoming)!=len(set(incoming)):raise HTTPException(409,'同一通道不能重复提交')
    if not set(incoming)<=set(ids):raise HTTPException(409,'采样不能包含开车布置以外测点')
    s.execute(delete(ChannelReading).where(ChannelReading.batch_id==x.id));s.flush()
    for v in d.readings:s.add(ChannelReading(batch_id=x.id,probe_id=v.probe_id,values=v.values,value=fmean(v.values),samples=len(v.values)))
    s.flush()
    # 采样源数据变化，旧版曲线立即失效；迟到的重算只能另开新版本，不能盖回 active
    lift_service.mark_stale(s,x.id)
    audit(s,u,'保存实际采样 '+str(id));return view(s,x)
def summary(s,x,r):
    rows=s.scalars(select(ChannelReading).where(ChannelReading.batch_id==x.id)).all();ids=[p['id'] for p in r.snapshot['probes']];expected=x.sample_rate*x.expected_seconds
    channels=[]
    for pid in ids:
        row=next((v for v in rows if v.probe_id==pid),None);count=row.samples if row else 0
        channels.append({'probe_id':pid,'samples':count,'mean_pa':round(fmean(row.values),6) if row else None,'missing_rate':round(max(0,expected-count)/expected,6)})
    return {'expected_samples_per_channel':expected,'channels':channels,'missing_rate':round(fmean(c['missing_rate'] for c in channels),6),'mean_pressure_pa':round(fmean(v.value for v in rows),6) if rows else None,'channel_count':len(rows),'required_channels':len(ids)}
def freeze(s,id,u):
    x,r=context(s,id)
    if x.state!='collecting':raise HTTPException(409,'采集已经冻结，不能重复签署')
    report=summary(s,x,r)
    if report['channel_count']!=report['required_channels'] or any(c['samples']<report['expected_samples_per_channel'] for c in report['channels']):raise HTTPException(409,'通道缺失或实际样本不足，不能冻结完整采集')
    # 冻结只认与当前实样同版、且覆盖全部通道的完整曲线（READ COMMITTED 下在此处重新取数）
    curve=lift_service.freeze_guard(s,x)
    x.summary=report;x.state='frozen';x.frozen_by=u.email
    audit(s,u,'主管冻结完整采集 '+str(id)+f'（沿用升力曲线 v{curve.version} / 模型 v{curve.model_archive_id}）');return view(s,x)
