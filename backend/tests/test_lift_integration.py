"""v4 实样升力反演集成测试：模型封存、同版曲线、缺样对齐、版本失效、并发冻结、交付快照。
用 SQLite 文件库（无 MySQL 环境），业务代码不改分支。运行：python tests/test_lift_integration.py
"""
import os,sys,time,threading,traceback
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['DATABASE_URL']='sqlite:////tmp/tunnel_v4_test.db'
os.environ['DB_ISOLATION_LEVEL']='SERIALIZABLE'
os.environ['REDIS_URL']='redis://localhost:6399/0'
os.environ['JWT_SECRET']='test-secret'
os.environ['SEED_DEMO']='false'
if os.path.exists('/tmp/tunnel_v4_test.db'):os.remove('/tmp/tunnel_v4_test.db')
import fakeredis
from fastapi.testclient import TestClient
from sqlalchemy import select,event
from src.db import Base,engine,Session
from src import models as M
from src.auth import hash_password
from src.models import User,Probe,Sweep,Pressure,Layout,LayoutProbe
from src.run_models import TunnelRun
from src.acquisition_models import AcquisitionBatch,ChannelReading
from src.lift_models import LiftModelArchive,LiftCurveVersion
from src.delivery_models import TunnelDelivery
from src import main as main_mod
main_mod.r=fakeredis.FakeStrictRedis(decode_responses=True)

@event.listens_for(engine,'connect')
def sqlite_pragmas(dbapi_con,_):
    cur=dbapi_con.cursor();cur.execute('PRAGMA busy_timeout=8000');cur.execute('PRAGMA journal_mode=WAL');cur.close()

PASS=0;FAIL=0
def check(name,cond,detail=''):
    global PASS,FAIL
    if cond:PASS+=1;print('  PASS',name)
    else:FAIL+=1;print('  FAIL',name,detail)

def seed():
    Base.metadata.create_all(engine)
    with Session() as s:
        s.add(User(email='admin@yiheng.demo',role='admin',password_hash=hash_password('pw')))
        s.add(User(email='engineer@yiheng.demo',role='engineer',password_hash=hash_password('pw')))
        s.add(User(email='viewer@yiheng.demo',role='viewer',password_hash=hash_password('pw')))
        probes=[]
        for i in range(12):
            p=Probe(code=f'P{i+1:02d}',zone=['root','mid','tip'][i//4],x_m=.1+(i%4)*.2,y_m=.2+(i//4)*.8,calibrated=True,enabled=True)
            s.add(p);probes.append(p)
        s.flush()
        import random
        rng=random.Random(7)
        for i in range(120):
            latent=[rng.gauss(0,1) for _ in range(3)]
            pressure=[100+rng.uniform(-4,16)*2+rng.uniform(18,42)*.5+latent[j//4]*(8-(j%4))+rng.gauss(0,1.2) for j in range(12)]
            lift=30+.18*pressure[0]+.14*pressure[4]+.11*pressure[8]+rng.gauss(0,.25)
            sw=Sweep(code=f'SW-{i+1:03d}',speed=round(rng.uniform(18,42),2),angle=round(rng.uniform(-4,16),2),lift=round(lift,3))
            s.add(sw);s.flush()
            for p,v in zip(probes,pressure):s.add(Pressure(sweep_id=sw.id,probe_id=p.id,pascal=round(v,3)))
        s.commit();return [p.id for p in probes]

def make_signed_layout(probe_ids):
    with Session() as s:
        l=Layout(code='LY-V4',max_channels=4,status='ready',auto_select=False,
                 source_checksum=__import__('src.services',fromlist=['source_checksum']).source_checksum(s))
        s.add(l);s.flush()
        for pid in probe_ids:s.add(LayoutProbe(layout_id=l.id,probe_id=pid))
        s.commit();return l.id

def make_run(layout_id,probe_objs):
    with Session() as s:
        r=TunnelRun(layout_id=layout_id,code='RUN-V4',speed=20.0,target_angle=5.0,
                    checklist={'air_path_clear':True,'emergency_stop_tested':True,'model_secured':True},
                    state='running',started_by='admin@yiheng.demo',started_at='2026-11-20T00:00:00+00:00',
                    snapshot={'layout':{'id':layout_id,'code':'LY-V4'},'probes':probe_objs,'certificates':[],'mounting':{}})
        s.add(r);s.flush()
        b=AcquisitionBatch(run_id=r.id,sample_rate=2,expected_seconds=3,state='collecting')
        s.add(b);s.commit();return r.id,b.id

def main():
    probe_ids=seed()
    client=TestClient(main_mod.app)
    def login(role):
        res=client.post('/auth/login',json={'email':f'{role}@yiheng.demo','password':'pw'})
        return {'Authorization':'Bearer '+res.json()['token']}
    H={'admin':None,'engineer':None,'viewer':None}
    for r in H:H[r]=login(r)
    selected=probe_ids[0:1]+probe_ids[4:5]+probe_ids[8:9]  # root/mid/tip 各一
    layout_id=make_signed_layout(selected)
    zone_by_id={pid:z for pid,z in zip(probe_ids,['root']*4+['mid']*4+['tip']*4)}
    probe_objs=[{'id':pid,'code':f'P{pid:02d}','zone':zone_by_id[pid]} for pid in selected]

    print('\n[1] 审签即封存模型，重复审签不另建')
    r=client.post(f'/layouts/{layout_id}/sign',headers=H['admin']);check('审签成功',r.status_code==200,r.text)
    with Session() as s:
        archives=s.scalars(select(LiftModelArchive).where(LiftModelArchive.layout_id==layout_id)).all()
        check('封存恰好一份模型',len(archives)==1)
        a=archives[0];check('模型版本v1',a.version==1)
        check('系数含缩放与Ridge',all(k in a.coefficients for k in('scaler_mean','scaler_scale','ridge_coef','intercept')))
        check('特征通道=选点',a.coefficients['feature_probe_ids']==selected)
        check('训练指标存在','train_mae_n' in a.metrics and a.metrics['train_rows']==120)
        check('模型来源SHA=布置来源',a.source_checksum==s.get(Layout,layout_id).source_checksum)

    print('\n[2] 开车后建立批次并写部分通道：缺样只出齐全时刻')
    run_id,batch_id=make_run(layout_id,probe_objs)
    r=client.post(f'/acquisition-batches/{batch_id}/lift-curves',headers=H['engineer'])
    check('无任何实样可先出曲线（0 齐全点）',r.status_code==200 and r.json()['curve']['point_count']==0,r.text)
    check('无实样整段升力为空',r.json()['curve']['lift_n'] is None)
    empty_miss=r.json()['missing']['missing_positions']
    check('缺样时刻=期望数，每时刻列全部缺失通道',len(empty_miss)==6 and all(set(m['missing_probe_ids'])==set(selected) for m in empty_miss))
    r=client.put(f'/acquisition-batches/{batch_id}/readings',headers=H['engineer'],
                 json={'readings':[{'probe_id':selected[0],'values':[101,102,103,104,105,106]},
                                   {'probe_id':selected[1],'values':[201,202,203,204]},
                                   {'probe_id':selected[2],'values':[301,302,303,304]}]})
    check('部分通道保存成功',r.status_code==200,r.text)
    r=client.post(f'/acquisition-batches/{batch_id}/lift-curves',headers=H['engineer'])
    body=r.json()
    check('齐全时刻=最短通道长度4',body['curve']['point_count']==4,str(body['curve']['point_count']))
    check('缺样时整段升力不出',body['curve']['complete'] is False and body['curve']['lift_n'] is None)
    miss={m['sample_index']:set(m['missing_probe_ids']) for m in body['missing']['missing_positions']}
    check('缺样位置逐时刻列出（位置4/5缺两个短通道）',set(miss)=={4,5} and miss[4]==set([selected[1],selected[2]]))
    check('曲线点按同一采样位置对齐',all(set(map(int,p['pressure_pa'].keys()))==set(selected) for p in body['curve']['points']))
    check('时刻从1/rate开始递增',[p['time_s'] for p in body['curve']['points']]==[.5,1.0,1.5,2.0])

    print('\n[3] 冻结拒绝缺样曲线')
    r=client.post(f'/acquisition-batches/{batch_id}/freeze',headers=H['admin'])
    check('缺样不能冻结',r.status_code==409 and '同版' not in r.json()['detail'],r.text)

    print('\n[4] 补齐后刷新：完整曲线与整段升力')
    r=client.put(f'/acquisition-batches/{batch_id}/readings',headers=H['engineer'],
                 json={'readings':[{'probe_id':pid,'values':[100+k+idx*2 for k in range(6)]} for idx,pid in enumerate(selected)]})
    check('全通道保存',r.status_code==200,r.text)
    r=client.post(f'/acquisition-batches/{batch_id}/lift-curves',headers=H['engineer']);full=r.json()
    check('完整曲线6点',full['curve']['point_count']==6 and full['curve']['complete'] is True)
    check('整段升力已反演',isinstance(full['curve']['lift_n'],(int,float)))
    check('缺样位清零',full['missing']['missing_position_count']==0)
    check('新版本号递增（空版v1→缺样v2→完整v3）',full['version']==3,str(full['version']))
    with Session() as s:
        vers=s.scalars(select(LiftCurveVersion).where(LiftCurveVersion.batch_id==batch_id).order_by(LiftCurveVersion.id)).all()
        check('旧版本保留且标 stale',all(v.state=='stale' for v in vers[:-1]) and vers[-1].state=='active')
        check('旧版曲线数据未被覆盖（缺样版4点/空版0点）',vers[0].curve['point_count']==0 and vers[1].curve['point_count']==4)
        # 手算反演与服务端一致
        import numpy as np
        a=s.get(LiftModelArchive,vers[1].model_archive_id)
        means=[np.mean([100+k+idx*2 for k in range(6)]) for idx in range(3)]
        expect=a.coefficients['intercept']+sum(a.coefficients['ridge_coef'][i]*(means[i]-a.coefficients['scaler_mean'][i])/a.coefficients['scaler_scale'][i] for i in range(3))
        check('整段升力=封存模型对均压的前向反演',abs(expect-full['curve']['lift_n'])<1e-3,f'{expect} vs {full["curve"]["lift_n"]}')

    print('\n[5] 采样源变化旧曲线立即失效，迟到重算不能盖回 active')
    client.put(f'/acquisition-batches/{batch_id}/readings',headers=H['engineer'],
               json={'readings':[{'probe_id':pid,'values':[900+k for k in range(6)]} for pid in selected]})
    with Session() as s:
        check('改写后 active 立即变 stale',s.scalars(select(LiftCurveVersion).where(LiftCurveVersion.batch_id==batch_id,LiftCurveVersion.state=='active')).all()==[])
    r=client.post(f'/acquisition-batches/{batch_id}/freeze',headers=H['admin'])
    check('冻结拒绝失效曲线',r.status_code==409 and '同版' in r.json()['detail'],r.text)
    # 模拟“迟到的重算”：用旧 checksum 直接写一条版本，验证冻结仍只认真正同版
    with Session() as s:
        old=s.scalars(select(LiftCurveVersion).where(LiftCurveVersion.batch_id==batch_id).order_by(LiftCurveVersion.id.desc())).first()
        late=LiftCurveVersion(batch_id=batch_id,model_archive_id=old.model_archive_id,version=99,state='active',
                              source_checksum='stale-checksum-from-late-job',curve=old.curve,missing=old.missing,
                              created_by='late-job',created_at='2026-11-20T00:00:00+00:00')
        s.add(late);s.commit()
    r=client.post(f'/acquisition-batches/{batch_id}/freeze',headers=H['admin'])
    check('迟到重算盖不住新采样（checksum 不符仍拒冻）',r.status_code==409,r.text)
    r=client.post(f'/acquisition-batches/{batch_id}/lift-curves',headers=H['engineer']);new=r.json()
    check('正常重算开新版本（v100）且 active',new['version']==100 and new['state']=='active',r.text)
    with Session() as s:
        late=s.scalar(select(LiftCurveVersion).where(LiftCurveVersion.batch_id==batch_id,LiftCurveVersion.version==99))
        check('迟到版本被置 stale 保留',late.state=='stale')

    print('\n[6] 同版幂等：未改采样重复刷新不新增版本')
    r1=client.post(f'/acquisition-batches/{batch_id}/lift-curves',headers=H['engineer'])
    r2=client.post(f'/acquisition-batches/{batch_id}/lift-curves',headers=H['viewer'])
    check('观察员只读不能刷新(403)',r2.status_code==403,r2.text)
    check('同版重复刷新幂等',r1.json()['id']==new['id'])

    print('\n[7] 冻结并发：冻结期间的改写必须让冻结失败，绝不锁旧快照')
    # 先恢复一份干净同版曲线
    client.put(f'/acquisition-batches/{batch_id}/readings',headers=H['engineer'],
               json={'readings':[{'probe_id':pid,'values':[110+k for k in range(6)]} for pid in selected]})
    client.post(f'/acquisition-batches/{batch_id}/lift-curves',headers=H['engineer'])
    barrier=threading.Event();results={}
    def freezer():
        # 模拟冻结事务：先取批次（与 context() 相同的早期读取），再等待并发改写，
        # 最后在同一步骤重新取数校验。MySQL READ COMMITTED 下自动看到新提交；
        # SQLite 快照隔离需显式结束旧读事务后重取，两者都验证“冻结不得锁旧曲线”。
        with Session() as s:
            from src.acquisition_service import context,summary
            from src import lift_service
            x,run=context(s,batch_id)
            barrier.set()
            time.sleep(0.4)
            try:
                s.commit()  # 结束早期读快照，等价于 READ COMMITTED 语句级快照刷新
                x,run=context(s,batch_id)
                report=summary(s,x,run)
                curve=lift_service.freeze_guard(s,x)
                x.summary=report;x.state='frozen';x.frozen_by='admin@yiheng.demo';s.commit()
                results['freeze']='ok'
            except Exception as ex:
                s.rollback();results['freeze']=str(ex)
    def rewriter():
        barrier.wait();time.sleep(0.1)
        with Session() as s:
            from src.acquisition_models import ChannelReading
            from sqlalchemy import delete
            s.execute(delete(ChannelReading).where(ChannelReading.batch_id==batch_id,ChannelReading.probe_id==selected[2]))
            from src import lift_service
            lift_service.mark_stale(s,batch_id);s.commit();results['rewrite']='ok'
    t1=threading.Thread(target=freezer);t2=threading.Thread(target=rewriter)
    t1.start();t2.start();t1.join();t2.join()
    check('改写事务已提交',results.get('rewrite')=='ok',str(results))
    check('并发冻结被拒绝（未锁旧曲线）',results.get('freeze')!='ok',str(results))
    with Session() as s:
        check('批次仍为采集中',s.get(AcquisitionBatch,batch_id).state=='collecting')

    print('\n[8] 恢复完整后冻结，交付快照保留曲线与模型依据')
    client.put(f'/acquisition-batches/{batch_id}/readings',headers=H['engineer'],
               json={'readings':[{'probe_id':pid,'values':[120+k for k in range(6)]} for pid in selected]})
    r=client.post(f'/acquisition-batches/{batch_id}/lift-curves',headers=H['engineer']);frozen_curve=r.json()
    r=client.post(f'/acquisition-batches/{batch_id}/freeze',headers=H['admin'])
    check('同版完整曲线冻结成功',r.status_code==200 and r.json()['state']=='frozen',r.text)
    with Session() as s:
        check('冻结后 active 曲线 checksum 与实样一致',s.get(LiftCurveVersion,frozen_curve['id']).source_checksum==__import__('src.lift_service',fromlist=['lift_checksum']).lift_checksum(s,s.get(AcquisitionBatch,batch_id)))
    r=client.post(f'/acquisition-batches/{batch_id}/lift-curves',headers=H['engineer'])
    check('冻结后刷新返回锁定的同版曲线',r.json()['id']==frozen_curve['id'])
    with Session() as s:
        s.get(TunnelRun,run_id).state='completed';s.commit()
    r=client.post('/tunnel-deliveries',headers=H['engineer'],json={'run_id':run_id,'recipient':'气动设计组'})
    check('交付提交成功',r.status_code==200,r.text)
    did=r.json()['id']
    with Session() as s:
        snap=s.get(TunnelDelivery,did).snapshot
        check('交付快照含曲线',snap['lift_curve']['id']==frozen_curve['id'])
        check('交付快照含封存模型',snap['lift_model']['layout_id']==layout_id and snap['lift_model']['sealed_by']=='admin@yiheng.demo')
        check('快照曲线含完整点表',len(snap['lift_curve']['curve']['points'])==6)
        # 冻结后再改写源表，交付快照不变
        s.execute(__import__('sqlalchemy').delete(ChannelReading).where(ChannelReading.batch_id==batch_id))
        s.commit()
        snap2=s.get(TunnelDelivery,did).snapshot
        check('源数据后变不影响交付快照',len(snap2['lift_curve']['curve']['points'])==6)
    r=client.post(f'/tunnel-deliveries/{did}/approve',headers=H['admin']);check('主管核准',r.status_code==200,r.text)
    r=client.post(f'/tunnel-deliveries/{did}/acknowledge',headers=H['engineer'],json={'receipt':'客户已签收归档凭证'})
    check('回执登记',r.status_code==200,r.text)

    print('\n[9] 查看接口与权限')
    r=client.get(f'/acquisition-batches/{batch_id}/lift-curves',headers=H['viewer'])
    check('观察员可查全部曲线版本',r.status_code==200 and len(r.json())>=3)
    r=client.get('/lift-model-archives',headers=H['viewer'])
    check('模型封存台账可查',r.status_code==200 and any(x['layout_id']==layout_id for x in r.json()))
    r=client.get('/acquisition-batches',headers=H['viewer'])
    b=next(x for x in r.json() if x['id']==batch_id)
    check('批次视图带曲线摘要',b['curve'] is not None and b['curve']['version']==frozen_curve['version'])

    print('\n[10] 升级兼容：v3 旧已签布置无封存模型，首次反演惰性补封存')
    legacy_ids=probe_ids[1:2]+probe_ids[5:6]+probe_ids[9:10]
    with Session() as s:
        from src.services import source_checksum as sc
        ll=Layout(code='LY-LEGACY',max_channels=4,status='signed',auto_select=False,source_checksum=sc(s),signed_by='old-admin')
        s.add(ll);s.flush()
        for pid in legacy_ids:s.add(LayoutProbe(layout_id=ll.id,probe_id=pid))
        s.commit();legacy_lid=ll.id
        check('旧布置确无封存模型',s.scalar(select(LiftModelArchive.id).where(LiftModelArchive.layout_id==legacy_lid)) is None)
    with Session() as s:
        rr=TunnelRun(layout_id=legacy_lid,code='RUN-LEGACY',speed=20.0,target_angle=5.0,
                     checklist={'air_path_clear':True,'emergency_stop_tested':True,'model_secured':True},state='running',
                     snapshot={'layout':{'id':legacy_lid,'code':'LY-LEGACY'},'probes':[{'id':pid,'code':f'P{pid:02d}','zone':zone_by_id[pid]} for pid in legacy_ids],'certificates':[],'mounting':{}})
        s.add(rr);s.flush();bb=AcquisitionBatch(run_id=rr.id,sample_rate=1,expected_seconds=2);s.add(bb);s.commit();legacy_bid=bb.id
    r=client.put(f'/acquisition-batches/{legacy_bid}/readings',headers=H['engineer'],
                 json={'readings':[{'probe_id':pid,'values':[100.0,101.0]} for pid in legacy_ids]})
    check('旧布置试验可正常录样',r.status_code==200,r.text)
    r=client.post(f'/acquisition-batches/{legacy_bid}/lift-curves',headers=H['engineer'])
    check('首次刷新惰性补封存并出曲线',r.status_code==200 and r.json()['curve']['complete'] is True,r.text)
    with Session() as s:
        a=s.scalar(select(LiftModelArchive).where(LiftModelArchive.layout_id==legacy_lid))
        check('补封存模型已生成且通道匹配',a is not None and a.coefficients['feature_probe_ids']==legacy_ids)
        check('补封存来源与布置一致',a.source_checksum==s.get(Layout,legacy_lid).source_checksum)
    r=client.post(f'/acquisition-batches/{legacy_bid}/lift-curves',headers=H['engineer'])
    check('补封存后同版刷新幂等',r.json()['model_archive_id']==a.id)

    print(f'\n==== {PASS} passed, {FAIL} failed ====')
    return FAIL
if __name__=='__main__':
    try:sys.exit(1 if main() else 0)
    except Exception:traceback.print_exc();sys.exit(2)
