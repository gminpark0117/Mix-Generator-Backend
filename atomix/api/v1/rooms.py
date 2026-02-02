from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response, status

from atomix.schemas.room import RoomCreateIn, RoomRenameIn, RoomOut, RoomListOut

router = APIRouter()

# In-memory store for stubs (resets on reload).
_ROOMS: Dict[uuid.UUID, RoomOut] = {}

# Optional: internal mapping if you still track room -> mix on server but don't expose it.
# _ROOM_MIX: Dict[uuid.UUID, uuid.UUID] = {}


@router.post("", response_model=RoomOut)
async def create_room(body: RoomCreateIn) -> RoomOut:
    room_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    room = RoomOut(
        room_id=str(room_id),
        name=body.name,
        created_at=now,
        participant_count=0,
    )
    _ROOMS[room_id] = room

    # If later you create a mix automatically, you'd store it internally here:
    # _ROOM_MIX[room_id] = created_mix_id

    return room


@router.get("", response_model=RoomListOut)
async def list_rooms() -> RoomListOut:
    # Later: filter deleted rooms, compute participant_count from ws presence, etc.
    return RoomListOut(rooms=list(_ROOMS.values()))


@router.patch("/{room_id}", response_model=RoomOut)
async def rename_room(room_id: uuid.UUID, body: RoomRenameIn) -> RoomOut:
    room = _ROOMS.get(room_id)
    if room is None:
        raise HTTPException(status_code=404, detail="room not found")

    updated = room.model_copy(update={"name": body.name})
    _ROOMS[room_id] = updated
    return updated


@router.post("/{room_id}/tracks:upload", status_code=status.HTTP_204_NO_CONTENT)
async def upload_room_tracks(
    room_id: uuid.UUID,
    files: list[UploadFile] = File(...),
    tracks_metadata: str = Form(...),  # JSON string aligned to files
) -> Response:
    """
    Stub endpoint: accept uploads + metadata and return empty response.
    Later, this should:
      - save files (StorageService)
      - insert audio_assets/mix_items
      - render new revision
      - broadcast revision_ready over WS
    """
    if room_id not in _ROOMS:
        raise HTTPException(status_code=404, detail="room not found")

    # Optional validation stub (enable if you want):
    # from atomix.services.mix_service import parse_tracks_metadata
    # parse_tracks_metadata(tracks_metadata, len(files))

    return Response(status_code=status.HTTP_204_NO_CONTENT)
