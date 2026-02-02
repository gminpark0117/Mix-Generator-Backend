from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from atomix.core import get_db
from atomix.schemas.room import RoomRenameIn, RoomOut, RoomListOut
from atomix.services.room_service import RoomService
from atomix.runtime.presence import room_presence
from atomix.api.v1.ws_rooms import _ROOM_RUNTIME, RoomOp, _process_room_ops, RoomRuntime, FileData

router = APIRouter()


def _build_room_out(room, participant_count: int | None = None) -> RoomOut:
    """Helper to build RoomOut with participant count from presence."""
    if participant_count is None:
        participant_count = len(room_presence.get(room.id, set()))
    return RoomOut(
        room_id=str(room.id),
        name=room.name,
        created_at=room.created_at,
        participant_count=participant_count,
    )


@router.post("", response_model=RoomOut)
async def create_room(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    tracks_metadata: str = Form(...),
    db: AsyncSession = Depends(get_db),
) -> RoomOut:
    svc = RoomService(db)
    try:
        room = await svc.create_room(
            name=name,
            files=files,
            tracks_metadata=tracks_metadata,
        )
        # Initialize room presence tracking
        room_presence[room.id]
        
        return _build_room_out(room, participant_count=0)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=RoomListOut)
async def list_rooms(
    db: AsyncSession = Depends(get_db),
) -> RoomListOut:
    svc = RoomService(db)
    rooms = await svc.list_rooms()
    return RoomListOut(rooms=[_build_room_out(room) for room in rooms])


@router.patch("/{room_id}", response_model=RoomOut)
async def rename_room(
    room_id: uuid.UUID,
    body: RoomRenameIn,
    db: AsyncSession = Depends(get_db),
) -> RoomOut:
    svc = RoomService(db)
    room = await svc.rename_room(room_id, name=body.name)
    if room is None:
        raise HTTPException(status_code=404, detail="room not found")
    
    return _build_room_out(room)


@router.delete("/{room_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_room(
    room_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """
    Soft-delete a room (for debugging purposes).
    Sets deleted_at timestamp and cleans up runtime state.
    """
    from atomix.api.v1.ws_rooms import _close_all_websockets
    
    svc = RoomService(db)
    room = await svc.delete_room(room_id)
    if room is None:
        raise HTTPException(status_code=404, detail="room not found")
    
    # Close all WebSocket connections for this room
    await _close_all_websockets(room_id, reason="room deleted")
    
    # Clean up runtime state
    _ROOM_RUNTIME.pop(room_id, None)
    room_presence.pop(room_id, None)
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{room_id}/tracks:upload", status_code=status.HTTP_204_NO_CONTENT)
async def upload_room_tracks(
    room_id: uuid.UUID,
    files: list[UploadFile] = File(...),
    tracks_metadata: str = Form(...),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """
    Accept track uploads and queue for processing.
    Returns 204 immediately, processes in background, and broadcasts revision_ready via WebSocket.
    Client playhead is calculated server-side from current_time - play_started_at_epoch_ms.
    """
    svc = RoomService(db)
    room = await svc.get_room(room_id)
    if room is None:
        raise HTTPException(status_code=404, detail="room not found")

    # Read file data immediately (before files are closed by FastAPI)
    file_data_list = []
    for upload in files:
        data = await upload.read()
        file_data_list.append(FileData(
            filename=upload.filename or "unknown",
            content_type=upload.content_type or "application/octet-stream",
            data=data,
        ))

    # Get or create room runtime
    rt = _ROOM_RUNTIME.get(room_id)
    if rt is None:
        # No active WebSocket connections, but still queue the operation
        rt = RoomRuntime()
        _ROOM_RUNTIME[room_id] = rt
    
    # Add operation to queue (with file data, not UploadFile objects)
    rt.room_ops.append(RoomOp(
        files=file_data_list,
        tracks_metadata=tracks_metadata,
    ))
    
    # Trigger background processing (non-blocking)
    asyncio.create_task(_process_room_ops(room_id))

    return Response(status_code=status.HTTP_204_NO_CONTENT)
