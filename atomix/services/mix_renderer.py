from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass
from typing import Protocol
import uuid

@dataclass(frozen=True)
class TrackInput:
    mix_item_id: uuid.UUID
    abs_path: str               # stored source file path (not used by placeholder)
    song_name: str
    artist_name: str

@dataclass(frozen=True)
class SegmentSpec:
    position: int
    mix_item_id: uuid.UUID
    start_ms: int
    end_ms: int
    source_start_ms: int

@dataclass(frozen=True)
class RenderResult:
    audio_bytes: bytes
    mime: str
    length_ms: int
    segments: list[SegmentSpec]

class MixRenderer(Protocol):
    async def render(self, tracks: list[TrackInput]) -> RenderResult:
        ...

class PlaceholderMixRenderer:
    """
    Placeholder renderer:
    - makes a silent WAV
    - assigns each track a fixed duration (e.g., 3000ms)
    - produces sequential segments
    """
    PER_TRACK_MS = 3000
    SAMPLE_RATE = 44100
    CHANNELS = 2
    SAMPWIDTH = 2  # bytes per sample

    async def render(self, tracks: list[TrackInput]) -> RenderResult:
        n = len(tracks)
        length_ms = max(1, n) * self.PER_TRACK_MS  # avoid 0ms mixes

        segments: list[SegmentSpec] = []
        t = 0
        for i, tr in enumerate(tracks):
            segments.append(SegmentSpec(
                position=i,
                mix_item_id=tr.mix_item_id,
                start_ms=t,
                end_ms=t + self.PER_TRACK_MS,
                source_start_ms=0,
            ))
            t += self.PER_TRACK_MS

        wav_bytes = self._generate_silence_wav(length_ms)
        return RenderResult(
            audio_bytes=wav_bytes,
            mime="audio/wav",
            length_ms=length_ms,
            segments=segments,
        )

    def _generate_silence_wav(self, duration_ms: int) -> bytes:
        frames = int(self.SAMPLE_RATE * duration_ms / 1000)
        silence_frame = struct.pack("<hh", 0, 0)  # 2 channels
        raw = silence_frame * frames

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.SAMPWIDTH)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(raw)
        return buf.getvalue()
