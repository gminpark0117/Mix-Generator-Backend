# atomix/api/v1/ws_rooms.py
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Set, Optional, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder

from atomix.schemas.ws import (
    RoomSnapshotOut,
    ChatSendIn,
    ChatMessageOut,
    RevisionReadyOut,
)
from atomix.schemas.room import RoomOut
from atomix.schemas.mix import RevisionOut, SegmentOut

# Import the in-memory room store from your REST stub.
# This avoids duplication and ensures snapshot room fields match REST /v1/rooms response.
from atomix.api.v1.rooms import _ROOMS  # noqa: F401


router = APIRouter(prefix="/rooms", tags=["rooms-ws"])


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _now_ms() -> int:
    return int(_utc_now().timestamp() * 1000)


def _roomout_field_name() -> str:
    """
    Support either RoomOut.room_id or RoomOut.id depending on your current schema.
    """
    fields = getattr(RoomOut, "model_fields", None)
    if isinstance(fields, dict):
        if "room_id" in fields:
            return "room_id"
        if "id" in fields:
            return "id"
    # Fallback (shouldn’t happen)
    return "room_id"


@dataclass
class RoomRuntime:
    sockets: Set[WebSocket] = field(default_factory=set)
    name_by_socket: Dict[WebSocket, str] = field(default_factory=dict)
    next_user_idx: int = 1

    play_started_at_epoch_ms: int = field(default_factory=_now_ms)

    chat_seq: int = 0
    chat_recent: list[ChatMessageOut] = field(default_factory=list)

    current_revision: RevisionOut = field(default_factory=lambda: RevisionOut(
        revision_no=1,
        switchover_ms=0,
        length_ms=180_000,
        audio_url="/storage/mix/placeholder.wav",  # stub URL
    ))

    tracklist: list[SegmentOut] = field(default_factory=list)


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
        rt.name_by_socket.pop(ws, None)


def _assign_name(rt: RoomRuntime) -> str:
    name = f"user-{rt.next_user_idx}"
    rt.next_user_idx += 1
    return name


def _build_room_out(room_id: uuid.UUID, participant_count: int) -> RoomOut:
    """
    Build RoomOut from REST stub store if present, else synthesize.
    Handles RoomOut.id vs RoomOut.room_id.
    """
    # Find in stub store (keys might be UUIDs)
    stub = _ROOMS.get(room_id)

    field_id = _roomout_field_name()
    if stub is not None:
        data = stub.model_dump()
        data["participant_count"] = participant_count
        # Ensure the identifier field exists and matches room_id
        data[field_id] = str(room_id)
        return RoomOut(**data)

    # synthesize room if missing (you may instead prefer to reject)
    payload = {
        field_id: str(room_id),
        "name": f"room-{str(room_id)[:6]}",
        "created_at": _utc_now(),
        "participant_count": participant_count,
    }
    return RoomOut(**payload)


async def broadcast_revision_ready(
    room_id: uuid.UUID,
    revision: RevisionOut,
    tracklist: list[SegmentOut],
) -> None:
    """
    Helper you can call later from your mix-building code.
    Not an HTTP route.
    """
    rt = _ROOM_RUNTIME.get(room_id)
    if rt is None:
        return

    rt.current_revision = revision
    rt.tracklist = tracklist

    msg = RevisionReadyOut(revision=revision, tracklist=tracklist)
    await _broadcast(rt, msg)


@router.websocket("/{room_id}/ws")
async def room_ws(room_id: uuid.UUID, websocket: WebSocket):
    # Optional: reject if room doesn't exist in REST stub store
    if room_id not in _ROOMS:
        await websocket.accept()
        await websocket.close(code=1008, reason="room not found")
        return

    await websocket.accept()

    rt = _ROOM_RUNTIME.get(room_id)
    if rt is None:
        rt = RoomRuntime()
        _ROOM_RUNTIME[room_id] = rt

    # Register connection
    rt.sockets.add(websocket)
    your_name = _assign_name(rt)
    rt.name_by_socket[websocket] = your_name

    # Snapshot
    snapshot = RoomSnapshotOut(
        your_name=your_name,
        room=_build_room_out(room_id, participant_count=len(rt.sockets)),
        play_started_at_epoch_ms=rt.play_started_at_epoch_ms,
        server_now_epoch_ms=_now_ms(),
        current_revision=rt.current_revision,
        tracklist=rt.tracklist,
        chat_recent=rt.chat_recent[-50:],
    )
    await websocket.send_json(jsonable_encoder(snapshot))

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
                chat_out = ChatMessageOut(
                    seq=rt.chat_seq,
                    sender_name=rt.name_by_socket.get(websocket, "unknown"),
                    message=chat_in.message,
                    created_at=_utc_now(),  # datetime now
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
        rt.name_by_socket.pop(websocket, None)

        # Update participant_count in REST stub store (optional)
        # We keep REST stub count purely cosmetic; WS runtime count is source of truth.
        # If you want REST list to reflect current presence, you can update it here.
        try:
            stub = _ROOMS.get(room_id)
            if stub is not None:
                _ROOMS[room_id] = RoomOut(**{
                    **stub.model_dump(),
                    "participant_count": len(rt.sockets),
                })
        except Exception:
            pass

        # Optional: cleanup runtime state when empty
        # if not rt.sockets:
        #     _ROOM_RUNTIME.pop(room_id, None)
