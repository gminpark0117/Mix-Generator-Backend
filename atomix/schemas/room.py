from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel
from typing import List

from atomix.schemas.mix import SegmentOut


class RoomCreateIn(BaseModel):
    name: str


class RoomRenameIn(BaseModel):
    name: str


class RoomOut(BaseModel):
    room_id: str
    name: str
    created_at: datetime
    participant_count: int = 0


class RoomListEntryOut(BaseModel):
    room: RoomOut
    current_playing: SegmentOut | None = None


class RoomListOut(BaseModel):
    rooms: List[RoomListEntryOut]
