from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from atomix.repos.mix_repo import MixRepo
from atomix.repos.room_repo import RoomRepo
from atomix.schemas.mix import SegmentOut
from atomix.services.mix_service import MixService


class RoomService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.rooms = RoomRepo(db)
        self.mixes = MixRepo(db)
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
        # 1) Create initial mix (MixService handles its own transaction)
        mix_result = await self.mix_svc.create_mix(
            files=files,
            tracks_metadata=tracks_metadata,
        )
        mix_id = uuid.UUID(mix_result.mix_id)

        # 2) Get current epoch ms for play_started_at_epoch_ms
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        # 3) Create room in DB (use a new transaction)
        async with self.db.begin():
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
        async with self.db.begin():
            return await self.rooms.rename_room(room_id, name=name)

    async def delete_room(self, room_id: uuid.UUID):
        """Soft-delete a room."""
        async with self.db.begin():
            return await self.rooms.soft_delete_room(room_id)

    async def get_current_segmentout(
        self,
        room,
        *,
        server_now_epoch_ms: int,
    ) -> SegmentOut | None:
        """Resolve currently playing segment for a room at server_now_epoch_ms."""
        playhead_ms = max(0, server_now_epoch_ms - room.play_started_at_epoch_ms)

        current_revision = await self.mixes.get_current_ready_revision(room.mix_id)
        if current_revision is None:
            return None

        row = await self.mixes.get_segment_with_item_metadata_at_playhead(current_revision.id, playhead_ms)
        if row is None:
            return None

        position, _mix_item_id, start_ms, end_ms, source_start_ms, metadata_json = row
        metadata = metadata_json or {}

        return SegmentOut(
            position=position,
            start_ms=start_ms,
            end_ms=end_ms,
            source_start_ms=source_start_ms,
            song_name=str(metadata.get("song_name", "")),
            artist_name=str(metadata.get("artist_name", "")),
        )
