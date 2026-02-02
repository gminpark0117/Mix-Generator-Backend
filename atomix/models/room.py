import uuid
from datetime import datetime
from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from atomix.models.base import Base

class Room(Base):
    __tablename__ = "rooms"
    __table_args__ = (
        Index("ix_rooms_deleted_at", "deleted_at"),
        Index("ix_rooms_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    mix_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("mixes.id", ondelete="CASCADE"), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    play_started_at_epoch_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)

    deleted_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=True)
