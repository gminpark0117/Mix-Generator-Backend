from __future__ import annotations

import asyncio
import json
import uuid
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Set, Any
from collections import deque

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile
from fastapi.encoders import jsonable_encoder
from starlette.datastructures import Headers

from atomix.core.db import AsyncSessionLocal
from atomix.schemas.ws import (
    RoomSnapshotOut,
    ChatSendIn,
    ChatMessageOut,
    ParticipantCountUpdateOut,
    RevisionReadyOut,
)
from atomix.schemas.room import RoomOut
from atomix.services.mix_service import MixService
from atomix.services.room_service import RoomService
from atomix.runtime.presence import room_presence


router = APIRouter(prefix="/rooms", tags=["rooms-ws"])


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _now_ms() -> int:
    return int(_utc_now().timestamp() * 1000)


@dataclass
class FileData:
    """In-memory file data for background processing."""
    filename: str
    content_type: str
    data: bytes


@dataclass
class RoomOp:
    """Pending room operation (track upload)."""
    files: list[FileData]
    tracks_metadata: str


@dataclass
class RoomRuntime:
    sockets: Set[WebSocket] = field(default_factory=set)
    conn_id_by_socket: Dict[WebSocket, str] = field(default_factory=dict)
    conn_id_to_name: Dict[str, str] = field(default_factory=dict)
    next_user_idx: int = 1

    play_started_at_epoch_ms: int = field(default_factory=_now_ms)
    last_activity_time: datetime = field(default_factory=_utc_now)

    chat_seq: int = 0
    chat_recent: list[ChatMessageOut] = field(default_factory=list)
    # All chat messages live in chat_recent; no db persistence for now
    
    room_ops: deque[RoomOp] = field(default_factory=deque)
    processing_ops: bool = False

# Runtime store keyed by room UUID
_ROOM_RUNTIME: Dict[uuid.UUID, RoomRuntime] = {}


async def _broadcast(rt: RoomRuntime, model: Any) -> None:
    """
    Broadcast a pydantic model to all sockets in the room.
    Uses jsonable_encoder to safely serialize datetimes.
    """
    payload = jsonable_encoder(model)
    dead: list[WebSocket] = []

    for ws in list(rt.sockets):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)

    for ws in dead:
        rt.sockets.discard(ws)
        rt.conn_id_by_socket.pop(ws, None)


async def _close_all_websockets(room_id: uuid.UUID, reason: str = "room closed") -> None:
    """
    Close all WebSocket connections for a room.
    """
    rt = _ROOM_RUNTIME.get(room_id)
    if rt is None:
        return
    
    for ws in list(rt.sockets):
        try:
            await ws.close(code=1000, reason=reason)
        except Exception:
            pass  # ignore errors when closing
    
    rt.sockets.clear()
    rt.conn_id_by_socket.clear()


def _assign_name(rt: RoomRuntime) -> str:
    name = f"user-{rt.next_user_idx}"
    rt.next_user_idx += 1
    return name


async def _broadcast_participant_count(room_id: uuid.UUID, rt: RoomRuntime) -> None:
    msg = ParticipantCountUpdateOut(participant_count=len(room_presence[room_id]))
    await _broadcast(rt, msg)


async def _process_room_ops(room_id: uuid.UUID) -> None:
    """
    Background processor: processes pending room operations sequentially.
    Called when a new op is added to the queue.
    """
    rt = _ROOM_RUNTIME.get(room_id)
    if rt is None:
        return
    
    # Prevent multiple processors running simultaneously
    if rt.processing_ops:
        return
    
    rt.processing_ops = True
    try:
        while rt.room_ops:
            op = rt.room_ops.popleft()
            
            # Process the track upload
            try:
                # First, read room info (using a separate session to avoid transaction conflicts)
                async with AsyncSessionLocal() as db_read:
                    room_svc = RoomService(db_read)
                    room = await room_svc.get_room(room_id)
                    if room is None:
                        continue
                    mix_id = room.mix_id
                    play_started_at_epoch_ms = room.play_started_at_epoch_ms
                
                # Calculate client playhead from server time
                now_ms = _now_ms()
                client_playhead_ms = now_ms - play_started_at_epoch_ms
                
                # Then, add tracks to mix (using a fresh session with its own transaction)
                # Convert FileData back to UploadFile-like objects
                upload_files = []
                for fd in op.files:
                    # Create an UploadFile from bytes
                    upload_files.append(UploadFile(
                        file=io.BytesIO(fd.data),
                        filename=fd.filename,
                        headers=Headers({"content-type": fd.content_type})
                    ))
                
                async with AsyncSessionLocal() as db_write:
                    mix_svc = MixService(db_write)
                    result = await mix_svc.add_tracks_to_mix(
                        mix_id=mix_id,
                        client_playhead_ms=client_playhead_ms,
                        files=upload_files,
                        tracks_metadata=op.tracks_metadata,
                    )
                
                # Broadcast revision_ready to all room participants
                msg = RevisionReadyOut(
                    revision=result.revision,
                    tracklist=result.tracklist,
                )
                await _broadcast(rt, msg)
                
            except Exception as e:
                # Log error but continue processing queue
                print(f"Error processing room op for room {room_id}: {e}")
                import traceback
                traceback.print_exc()
                
    finally:
        rt.processing_ops = False


async def _cleanup_empty_rooms() -> None:
    """
    Background task: clean up rooms with no active connections for 10+ minutes.
    Removes from _ROOM_RUNTIME, soft-deletes from DB, and clears room_presence.
    """
    import asyncio
    
    IDLE_THRESHOLD_SECONDS = 600  # 10 minutes
    
    while True:
        await asyncio.sleep(60)  # Check every minute
        
        now = _utc_now()
        rooms_to_delete = []
        
        for room_id, rt in list(_ROOM_RUNTIME.items()):
            # Check if room has no active connections
            if len(room_presence[room_id]) == 0:
                idle_time = (now - rt.last_activity_time).total_seconds()
                if idle_time >= IDLE_THRESHOLD_SECONDS:
                    rooms_to_delete.append(room_id)
        
        # Clean up idle rooms
        for room_id in rooms_to_delete:
            try:
                # Close all WebSocket connections
                await _close_all_websockets(room_id, reason="room cleaned up due to inactivity")
                
                # Soft-delete room from DB
                async with AsyncSessionLocal() as db:
                    room_svc = RoomService(db)
                    await room_svc.delete_room(room_id)
                
                # Remove from runtime
                _ROOM_RUNTIME.pop(room_id, None)
                room_presence.pop(room_id, None)
                
            except Exception as e:
                print(f"Error cleaning up room {room_id}: {e}")


@router.websocket("/{room_id}/ws")
async def room_ws(room_id: uuid.UUID, websocket: WebSocket):
    async with AsyncSessionLocal() as db:
        room_svc = RoomService(db)
        mix_svc = MixService(db)

        room = await room_svc.get_room(room_id)
        if room is None:
            await websocket.accept()
            await websocket.close(code=1008, reason="room not found")
            return

        await websocket.accept()

        rt = _ROOM_RUNTIME.get(room_id)
        if rt is None:
            rt = RoomRuntime()
            _ROOM_RUNTIME[room_id] = rt
        
        # Register connection (but don't add to rt.sockets yet)
        conn_id = str(uuid.uuid4())
        your_name = _assign_name(rt)
        rt.conn_id_to_name[conn_id] = your_name

        room_presence[room_id].add(conn_id)
        rt.last_activity_time = _utc_now()

        # Build mix snapshot
        try:
            mix_state = await mix_svc.get_mix_state(room.mix_id)
            if mix_state.current_ready_revision_no is None:
                await websocket.close(code=1011, reason="mix revision not ready")
                return

            mix_rev = await mix_svc.get_mix_revision(room.mix_id, mix_state.current_ready_revision_no)
        except KeyError:
            await websocket.close(code=1011, reason="mix not found")
            return

        snapshot = RoomSnapshotOut(
            your_name=your_name,
            room=RoomOut(
                room_id=str(room.id),
                name=room.name,
                created_at=room.created_at,
                participant_count=len(room_presence[room_id]),
            ),
            play_started_at_epoch_ms=room.play_started_at_epoch_ms,
            server_now_epoch_ms=_now_ms(),
            current_revision=mix_rev.revision,
            tracklist=mix_rev.tracklist,
            chat_recent=rt.chat_recent[-50:],
        )
        await websocket.send_json(jsonable_encoder(snapshot))

        # Now add to sockets and broadcast to others
        rt.sockets.add(websocket)
        rt.conn_id_by_socket[websocket] = conn_id
        await _broadcast_participant_count(room_id, rt)

        try:
            while True:
                raw = await websocket.receive_text()

                # Parse incoming message
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "chat_send":
                    try:
                        chat_in = ChatSendIn(**data)
                    except Exception:
                        continue

                    rt.chat_seq += 1
                    sender_name = rt.conn_id_to_name.get(conn_id, "unknown")
                    chat_out = ChatMessageOut(
                        seq=rt.chat_seq,
                        sender_name=sender_name,
                        message=chat_in.message,
                        created_at=_utc_now(),
                    )

                    rt.chat_recent.append(chat_out)
                    if len(rt.chat_recent) > 200:
                        rt.chat_recent = rt.chat_recent[-200:]

                    await _broadcast(rt, chat_out)

                # Ignore unknown message types for now

        except WebSocketDisconnect:
            pass
        finally:
            rt.sockets.discard(websocket)
            rt.conn_id_by_socket.pop(websocket, None)

            room_presence[room_id].discard(conn_id)
            rt.conn_id_to_name.pop(conn_id, None)
            rt.last_activity_time = _utc_now()

            await _broadcast_participant_count(room_id, rt)

            # Optional: cleanup runtime state when empty
            # if not rt.sockets:
            #     _ROOM_RUNTIME.pop(room_id, None)
