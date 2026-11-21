from sqlalchemy import String,Integer,ForeignKey,JSON,UniqueConstraint,CheckConstraint
from sqlalchemy.orm import Mapped,mapped_column
from .db import Base

class LiftModelArchive(Base):
    """布置审签时封存的实样升力反演模型；开车后的曲线反演只沿用这一版系数。"""
    __tablename__='lift_model_archives'
    __table_args__=(UniqueConstraint('layout_id'),CheckConstraint('version > 0'))
    id:Mapped[int]=mapped_column(primary_key=True)
    layout_id:Mapped[int]=mapped_column(ForeignKey('layouts.id'))
    version:Mapped[int]=mapped_column(Integer,default=1)
    source_checksum:Mapped[str]=mapped_column(String(64))
    coefficients:Mapped[dict]=mapped_column(JSON)
    metrics:Mapped[dict]=mapped_column(JSON)
    training_summary:Mapped[dict]=mapped_column(JSON)
    sealed_by:Mapped[str]=mapped_column(String(100))
    sealed_at:Mapped[str]=mapped_column(String(40))

class LiftCurveVersion(Base):
    """同版采样的完整升力曲线；采样源变化即出新版本，旧版本立即置 stale 且永不被迟到重算覆盖。"""
    __tablename__='lift_curve_versions'
    __table_args__=(UniqueConstraint('batch_id','version'),CheckConstraint('version > 0'),CheckConstraint("state in ('active','stale')"))
    id:Mapped[int]=mapped_column(primary_key=True)
    batch_id:Mapped[int]=mapped_column(ForeignKey('acquisition_batches.id'))
    model_archive_id:Mapped[int]=mapped_column(ForeignKey('lift_model_archives.id'))
    version:Mapped[int]=mapped_column(Integer)
    state:Mapped[str]=mapped_column(String(20),default='active')
    source_checksum:Mapped[str]=mapped_column(String(64))
    curve:Mapped[dict]=mapped_column(JSON)
    missing:Mapped[dict]=mapped_column(JSON)
    created_by:Mapped[str]=mapped_column(String(100))
    created_at:Mapped[str]=mapped_column(String(40))
    superseded_at:Mapped[str|None]=mapped_column(String(40),nullable=True)
