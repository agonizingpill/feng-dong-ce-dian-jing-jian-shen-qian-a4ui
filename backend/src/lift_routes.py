from fastapi import APIRouter,Depends,Query
from sqlalchemy import select
from .db import db_session
from .auth import current,writers
from .lift_models import LiftModelArchive
from .services import pack
from . import lift_service
router=APIRouter()
@router.get('/acquisition-batches/{batch_id}/lift-curves')
def curve_versions(batch_id:int,s=Depends(db_session),u=Depends(current)):
    return lift_service.versions(s,batch_id)
@router.post('/acquisition-batches/{batch_id}/lift-curves')
def refresh_curve(batch_id:int,s=Depends(db_session),u=Depends(writers)):
    x=lift_service.snapshot_curve(s,batch_id,u);s.commit();return x
@router.get('/lift-model-archives')
def archives(layout_id:int|None=Query(default=None),s=Depends(db_session),u=Depends(current)):
    q=select(LiftModelArchive).order_by(LiftModelArchive.id.desc())
    if layout_id is not None:q=q.where(LiftModelArchive.layout_id==layout_id)
    return [pack(a) for a in s.scalars(q)]
