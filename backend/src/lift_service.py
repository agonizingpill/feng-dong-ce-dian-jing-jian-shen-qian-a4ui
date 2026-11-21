from datetime import datetime,timezone
from statistics import fmean
import hashlib,json
import numpy as np
from fastapi import HTTPException
from sqlalchemy import select,func
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,r2_score
from .models import Probe,Sweep,Pressure,Layout
from .lift_models import LiftModelArchive,LiftCurveVersion
from .run_models import TunnelRun
from .acquisition_models import AcquisitionBatch,ChannelReading
from .calibration_service import locked
from .services import get,selected,pack,audit

RIDGE_ALPHA=1.0
def now_iso():return datetime.now(timezone.utc).isoformat()

def lift_checksum(s,batch):
    rows=s.scalars(select(ChannelReading).where(ChannelReading.batch_id==batch.id).order_by(ChannelReading.probe_id,ChannelReading.id)).all()
    content=[{'probe_id':v.probe_id,'samples':v.samples,'values':[float(z) for z in v.values]} for v in rows]
    return hashlib.sha256(json.dumps(content,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def seal_model(s,l,u):
    """布置审签时封存：以已签来源的全部扫角在选中通道上拟合一次，系数此后不随实样重拟合。"""
    existing=s.scalar(select(LiftModelArchive).where(LiftModelArchive.layout_id==l.id).with_for_update())
    if existing:return existing
    ids=selected(s,l.id)
    if not ids:raise HTTPException(409,'布置尚未选点，不能封存升力反演模型')
    probes=[get(s,Probe,i) for i in ids]
    sweeps=s.scalars(select(Sweep).order_by(Sweep.id)).all()
    if len(sweeps)<40:raise HTTPException(409,'至少40组完整扫角才能封存升力反演模型')
    values={(p.sweep_id,p.probe_id):p.pascal for p in s.scalars(select(Pressure))}
    try:X=np.array([[values[(sw.id,pid)] for pid in ids] for sw in sweeps],dtype=float)
    except KeyError:raise HTTPException(409,'扫角压力矩阵存在缺测点，不能封存升力反演模型')
    y=np.array([sw.lift for sw in sweeps],dtype=float)
    model=make_pipeline(StandardScaler(),Ridge(alpha=RIDGE_ALPHA));model.fit(X,y)
    scaler=model.named_steps['standardscaler'];ridge=model.named_steps['ridge']
    pred=model.predict(X)
    archive=LiftModelArchive(
        layout_id=l.id,version=1,source_checksum=l.source_checksum,
        coefficients={'model':'StandardScaler + Ridge(alpha=1.0)','feature_probe_ids':ids,
                      'scaler_mean':[float(v) for v in scaler.mean_],'scaler_scale':[float(v) for v in scaler.scale_],
                      'ridge_coef':[float(v) for v in ridge.coef_],'intercept':float(ridge.intercept_),
                      'predict':'intercept + sum(coef_i * (mean_pressure_i - scaler_mean_i)/scaler_scale_i)'},
        metrics={'train_rows':len(sweeps),'train_mae_n':round(float(mean_absolute_error(y,pred)),3),
                 'train_r2':round(float(r2_score(y,pred)),4)},
        training_summary={'sweep_codes':[sw.code for sw in sweeps],'probe_codes':[p.code for p in probes],
                          'alpha':RIDGE_ALPHA,'note':'审签时全量扫角一次拟合封存，现场实样只做前向反演，不重新拟合'},
        sealed_by=u.email,sealed_at=now_iso())
    s.add(archive);s.flush();audit(s,u,'封存实样升力反演模型 '+l.code);return archive

def active_archive(s,layout_id,u=None):
    a=s.scalar(select(LiftModelArchive).where(LiftModelArchive.layout_id==layout_id))
    if a:return a
    # 兼容升级前已审签的旧布置：来源既已冻结，首次反演时按同版来源补封存，结果等价于签时封存
    if u is None:raise HTTPException(409,'该布置尚未封存升力反演模型，不能反演实样升力')
    l=locked(s,Layout,layout_id)
    if l.status!='signed':raise HTTPException(409,'只有已审签布置可以封存升力反演模型')
    return seal_model(s,l,u)

def compute_curve(s,batch,archive):
    """各通道按同一采样位置对齐：只输出全部通道齐全的时刻；缺样位置单列，不插值、不补齐。"""
    run=s.get(TunnelRun,batch.run_id)
    probe_ids=[p['id'] for p in run.snapshot['probes']]
    feat=archive.coefficients['feature_probe_ids']
    if set(feat)!=set(probe_ids):raise HTTPException(409,'封存模型通道与开车快照布置不一致，不能沿用该模型')
    rows={v.probe_id:v for v in s.scalars(select(ChannelReading).where(ChannelReading.batch_id==batch.id))}
    n=batch.sample_rate*batch.expected_seconds
    align={pid:i for i,pid in enumerate(feat)};order=[align[pid] for pid in probe_ids]
    mean=[float(archive.coefficients['scaler_mean'][i]) for i in order]
    scale=[float(archive.coefficients['scaler_scale'][i]) for i in order]
    coef=[float(archive.coefficients['ridge_coef'][i]) for i in order]
    intercept=float(archive.coefficients['intercept'])
    def infer(xs):return intercept+sum(coef[i]*(xs[i]-mean[i])/scale[i] for i in range(len(probe_ids)))
    channels=[];means=[]
    for pid in probe_ids:
        row=rows.get(pid);count=row.samples if row else 0
        m=fmean(row.values) if row else None;means.append(m)
        channels.append({'probe_id':pid,'samples':count,'mean_pa':round(m,6) if row else None,
                         'missing_rate':round(max(0,n-count)/n,6),'present':row is not None})
    # 完整=每个通道在全部期望采样位置都有实样（无缺样位置）；仅通道存在但长度不足仍算缺样
    complete=all(c['samples']>=n for c in channels)
    lift_n=round(infer(means),4) if complete else None
    points=[]
    # 各通道按同一采样位置 k 对齐：只有该时刻全部通道都有实样才输出曲线点，缺样时刻跳过不补齐
    for k in range(n):
        step=[];ok=True
        for pid in probe_ids:
            row=rows.get(pid)
            if not row or k>=row.samples:step.append(None);ok=False
            else:step.append(float(row.values[k]))
        if not ok:continue
        points.append({'sample_index':k,'time_s':round((k+1)/batch.sample_rate,4),'lift_n':round(infer(step),4),
                       'pressure_pa':{str(pid):round(step[i],4) for i,pid in enumerate(probe_ids)}})
    missing_positions=[]
    for k in range(n):
        miss=[pid for pid in probe_ids if not (pid in rows and k<rows[pid].samples)]
        if miss:missing_positions.append({'sample_index':k,'time_s':round((k+1)/batch.sample_rate,4),'missing_probe_ids':miss})
    curve={'model_archive_id':archive.id,'model_version':archive.version,'complete':complete,'lift_n':lift_n,
           'channel_mean_pressure_pa':round(fmean(means),6) if complete else None,
           'point_count':len(points),'points':points}
    missing={'expected_samples_per_channel':n,'channels':channels,
             'missing_position_count':len(missing_positions),'missing_positions':missing_positions}
    return curve,missing

def active_version(s,batch_id,lock=False):
    q=select(LiftCurveVersion).where(LiftCurveVersion.batch_id==batch_id,LiftCurveVersion.state=='active').order_by(LiftCurveVersion.id.desc())
    if lock:q=q.with_for_update()
    return s.scalar(q)

def mark_stale(s,batch_id):
    """采样源一旦改写，旧曲线立即失效；迟到重算只能再开新版本，不能盖回 active。"""
    for v in s.scalars(select(LiftCurveVersion).where(LiftCurveVersion.batch_id==batch_id,LiftCurveVersion.state=='active')):
        v.state='stale';v.superseded_at=now_iso()

def snapshot_curve(s,batch_id,u):
    pre=s.get(AcquisitionBatch,batch_id)
    if not pre:raise HTTPException(404,'采集批次不存在')
    run=locked(s,TunnelRun,pre.run_id);batch=locked(s,AcquisitionBatch,batch_id)
    archive=active_archive(s,run.layout_id,u)
    checksum=lift_checksum(s,batch);current=active_version(s,batch.id)
    if batch.state=='frozen':
        if not current or current.source_checksum!=checksum:raise HTTPException(409,'冻结批次缺少同版曲线，数据一致性异常，请联系负责人')
        return pack(current)
    if current and current.source_checksum==checksum:return pack(current)
    if current:mark_stale(s,batch.id)
    curve,missing=compute_curve(s,batch,archive)
    prev=s.scalar(select(func.max(LiftCurveVersion.version)).where(LiftCurveVersion.batch_id==batch.id)) or 0
    v=LiftCurveVersion(batch_id=batch.id,model_archive_id=archive.id,version=prev+1,state='active',
                       source_checksum=checksum,curve=curve,missing=missing,created_by=u.email,created_at=now_iso())
    s.add(v);s.flush()
    if curve['complete']:audit(s,u,f'刷新实样升力曲线 v{v.version}（完整 {curve["point_count"]} 点）')
    else:audit(s,u,f'刷新实样升力曲线 v{v.version}（缺样，仅出齐全时刻 {curve["point_count"]} 点）')
    return pack(v)

def versions(s,batch_id):
    if not s.get(AcquisitionBatch,batch_id):raise HTTPException(404,'采集批次不存在')
    return [pack(v) for v in s.scalars(select(LiftCurveVersion).where(LiftCurveVersion.batch_id==batch_id).order_by(LiftCurveVersion.id.desc()))]

def curve_summary(s,batch_id):
    v=active_version(s,batch_id)
    if not v:return None
    return {'id':v.id,'version':v.version,'state':v.state,'source_checksum':v.source_checksum,
            'model_archive_id':v.model_archive_id,'complete':v.curve['complete'],'lift_n':v.curve['lift_n'],
            'point_count':v.curve['point_count'],'missing_position_count':v.missing['missing_position_count'],
            'stale':v.source_checksum!=lift_checksum(s,s.get(AcquisitionBatch,batch_id))}

def freeze_guard(s,batch):
    """冻结只认与当前实样同版、且通道齐全的完整曲线。"""
    v=active_version(s,batch.id,lock=True)
    if not v:raise HTTPException(409,'缺少与当前实样同版的升力曲线，请先刷新曲线再冻结')
    if v.source_checksum!=lift_checksum(s,batch):raise HTTPException(409,'采样源已变化，旧曲线已失效，请刷新同版曲线后再冻结')
    if not v.curve.get('complete'):raise HTTPException(409,'曲线仍有缺样通道，不能冻结完整采集')
    return v

def archive_view(s,layout_id):
    a=s.scalar(select(LiftModelArchive).where(LiftModelArchive.layout_id==layout_id))
    return pack(a) if a else None
