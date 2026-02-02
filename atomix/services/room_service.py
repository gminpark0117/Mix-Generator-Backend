from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from atomix.repos.room_repo import RoomRepo
from atomix.services.mix_service import MixService, parse_tracks_metadata


class RoomService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.rooms = RoomRepo(db)
        self.mix_svc = MixService(db)

    async def create_room(
        self,
        *,
        name: str,
        files: list[UploadFile],
        tracks_metadata: str,
    ):
        """
        Create a room with an initial mix.
        
        Steps:
        1. Create mix via MixService (handles file storage, mix_items, rendering, revision)
        2. Create room in DB with the mix_id
        """
        async with self.db.begin():
            # 1) Create initial mix
            mix_result = await self.mix_svc.create_mix(
                files=files,
                tracks_metadata=tracks_metadata,
            )
            mix_id = uuid.UUID(mix_result.mix_id)
            
            # 2) Get current epoch ms for play_started_at_epoch_ms
            # Using current epoch time as the start point (client will synchronize playhead)
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            
            # 3) Create room in DB
            room = await self.rooms.create_room(
                name=name,
                mix_id=mix_id,
                play_started_at_epoch_ms=now_ms,
            )
        
        return room

    async def list_rooms(self) -> list:
        """List all active (non-deleted) rooms."""
        return await self.rooms.list_rooms(include_deleted=False)

    async def get_room(self, room_id: uuid.UUID):
        """Get a specific room by ID."""
        return await self.rooms.get_room(room_id, include_deleted=False)

    async def rename_room(self, room_id: uuid.UUID, *, name: str):
        """Rename a room."""
        return await self.rooms.rename_room(room_id, name=name)

    async def delete_room(self, room_id: uuid.UUID):
        """Soft-delete a room."""
        return await self.rooms.soft_delete_room(room_id)
