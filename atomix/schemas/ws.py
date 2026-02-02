from __future__ import annotations

from datetime import datetime
from typing import Literal, List, Union
from pydantic import BaseModel, Field

from atomix.schemas.room import RoomOut
from atomix.schemas.mix import RevisionOut, SegmentOut


# ---- client -> server ----

class ChatSendIn(BaseModel):
    type: Literal["chat_send"] = "chat_send"
    message: str = Field(..., min_length=1, max_length=1000)

ClientToServer = Union[ChatSendIn]


# ---- server -> clients ----

class ChatMessageOut(BaseModel):
    type: Literal["chat_message"] = "chat_message"
    seq: int
    sender_name: str
    message: str
    created_at: datetime  # changed from epoch-ms


class RoomSnapshotOut(BaseModel):
    type: Literal["room_snapshot"] = "room_snapshot"
    your_name: str
    room: RoomOut

    play_started_at_epoch_ms: int
    server_now_epoch_ms: int

    current_revision: RevisionOut
    tracklist: List[SegmentOut] = []
    chat_recent: List[ChatMessageOut] = []


class RevisionReadyOut(BaseModel):
    type: Literal["revision_ready"] = "revision_ready"
    revision: RevisionOut
    tracklist: List[SegmentOut] = []


class ParticipantCountUpdateOut(BaseModel):
    type: Literal["participant_count_update"] = "participant_count_update"
    participant_count: int


ServerToClient = Union[RoomSnapshotOut, ChatMessageOut, RevisionReadyOut, ParticipantCountUpdateOut]
