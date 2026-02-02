from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atomix.models import Room


class RoomRepo:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_room(self, room_id: uuid.UUID, *, include_deleted: bool = False) -> Room | None:
        stmt = select(Room).where(Room.id == room_id)
        if not include_deleted:
            stmt = stmt.where(Room.deleted_at.is_(None))
        res = await self.db.execute(stmt)
        return res.scalar_one_or_none()

    async def create_room(
        self,
        *,
        name: str,
        mix_id: uuid.UUID,
        play_started_at_epoch_ms: int,
    ) -> Room:
        room = Room(
            name=name,
            mix_id=mix_id,
            play_started_at_epoch_ms=play_started_at_epoch_ms,
        )
        self.db.add(room)
        await self.db.flush()  # assign room.id
        return room

    async def list_rooms(self, *, include_deleted: bool = False) -> list[Room]:
        stmt = select(Room)
        if not include_deleted:
            stmt = stmt.where(Room.deleted_at.is_(None))
        stmt = stmt.order_by(Room.created_at.desc())
        res = await self.db.execute(stmt)
        return list(res.scalars().all())

    async def rename_room(self, room_id: uuid.UUID, *, name: str) -> Room | None:
        room = await self.get_room(room_id)
        if room is None:
            return None
        room.name = name
        await self.db.flush()
        return room

    async def soft_delete_room(self, room_id: uuid.UUID) -> Room | None:
        room = await self.get_room(room_id, include_deleted=True)
        if room is None:
            return None
        if room.deleted_at is None:
            room.deleted_at = datetime.now(timezone.utc)
            await self.db.flush()
        return room

    async def list_rooms_for_mix(self, mix_id: uuid.UUID, *, include_deleted: bool = False) -> list[Room]:
        stmt = select(Room).where(Room.mix_id == mix_id)
        if not include_deleted:
            stmt = stmt.where(Room.deleted_at.is_(None))
        stmt = stmt.order_by(Room.created_at.desc())
        res = await self.db.execute(stmt)
        return list(res.scalars().all())
