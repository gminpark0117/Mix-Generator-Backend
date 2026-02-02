from pydantic import BaseModel
from typing import List, Optional

class MixStateOut(BaseModel):
    mix_id: str
    current_ready_revision_no: Optional[int]

class RevisionOut(BaseModel):
    revision_no: int
    switchover_ms: int
    length_ms: int
    audio_url: str

class SegmentOut(BaseModel):
    position: int
    start_ms: int
    end_ms: int
    source_start_ms: int
    song_name: str
    artist_name: str

class MixRevisionResponse(BaseModel):
    mix_id: str
    revision: RevisionOut
    tracklist: List[SegmentOut]
