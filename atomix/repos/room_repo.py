from __future__ import annotations

import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atomix.models import Room


class RoomRepo:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_room(self, room_id: uuid.UUID) -> Room | None:
        res = await self.db.execute(select(Room).where(Room.id == room_id))
        return res.scalar_one_or_none()

    # add create/list/rename later
