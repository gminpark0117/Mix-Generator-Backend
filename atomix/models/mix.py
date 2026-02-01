import uuid
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from atomix.models.base import Base

class Mix(Base):
    __tablename__ = "mixes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    current_ready_revision_no: Mapped[int] = mapped_column(Integer, nullable=True)

    items: Mapped[list["MixItem"]] = relationship(back_populates="mix", cascade="all, delete-orphan")
    revisions: Mapped[list["MixRevision"]] = relationship(back_populates="mix", cascade="all, delete-orphan")


class MixItem(Base):
    __tablename__ = "mix_items"
    __table_args__ = (
        Index("ix_mix_items_mix_id_added_at", "mix_id", "added_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    mix_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("mixes.id", ondelete="CASCADE"), nullable=False)
    added_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=True)   # {song_name, artist_name}
    analysis_json: Mapped[dict] = mapped_column(JSONB, nullable=True)

    source_audio_asset_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("audio_assets.id"), nullable=False)

    mix: Mapped["Mix"] = relationship(back_populates="items")


class MixRevision(Base):
    __tablename__ = "mix_revisions"
    __table_args__ = (
        UniqueConstraint("mix_id", "revision_no", name="uq_mix_revisions_mix_id_revision_no"),
        CheckConstraint("switchover_ms >= 0", name="ck_mix_revisions_switchover_ms_nonneg"),
        CheckConstraint("length_ms > 0", name="ck_mix_revisions_length_ms_pos"),
        Index("ix_mix_revisions_mix_id_revision_no", "mix_id", "revision_no"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    mix_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("mixes.id", ondelete="CASCADE"), nullable=False)

    revision_no: Mapped[int] = mapped_column(Integer, nullable=False)
    switchover_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    length_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    audio_asset_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("audio_assets.id"), nullable=False)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    mix: Mapped["Mix"] = relationship(back_populates="revisions")
    segments: Mapped[list["MixSegment"]] = relationship(back_populates="revision", cascade="all, delete-orphan")


class MixSegment(Base):
    __tablename__ = "mix_segments"
    __table_args__ = (
        UniqueConstraint("mix_revision_id", "position", name="uq_mix_segments_revision_pos"),
        CheckConstraint("start_ms >= 0", name="ck_mix_segments_start_nonneg"),
        CheckConstraint("end_ms > start_ms", name="ck_mix_segments_end_gt_start"),
        CheckConstraint("source_start_ms >= 0", name="ck_mix_segments_source_start_nonneg"),
        Index("ix_mix_segments_revision_pos", "mix_revision_id", "position"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    mix_revision_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("mix_revisions.id", ondelete="CASCADE"), nullable=False)

    position: Mapped[int] = mapped_column(Integer, nullable=False)
    mix_item_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("mix_items.id"), nullable=False)

    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    source_start_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    revision: Mapped["MixRevision"] = relationship(back_populates="segments")
