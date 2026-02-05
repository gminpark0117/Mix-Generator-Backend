from __future__ import annotations

import uuid
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atomix.models import Mix, MixItem, MixRevision, MixSegment


class MixRepo:
    def __init__(self, db: AsyncSession):
        self.db = db

    # --- mixes ---
    async def create_mix(self) -> Mix:
        mix = Mix()
        self.db.add(mix)
        await self.db.flush()  # assign mix.id
        return mix

    async def get_mix(self, mix_id: uuid.UUID) -> Mix | None:
        res = await self.db.execute(select(Mix).where(Mix.id == mix_id))
        return res.scalar_one_or_none()

    async def set_current_ready_revision_no(self, mix: Mix, revision_no: int) -> None:
        mix.current_ready_revision_no = revision_no
        await self.db.flush()

    # --- mix_items ---
    async def add_mix_item(
        self,
        *,
        mix_id: uuid.UUID,
        source_audio_asset_id: uuid.UUID,
        metadata_json: dict,
        analysis_json: dict | None = None,
    ) -> MixItem:
        item = MixItem(
            mix_id=mix_id,
            source_audio_asset_id=source_audio_asset_id,
            metadata_json=metadata_json,
            analysis_json=analysis_json,
        )
        self.db.add(item)
        await self.db.flush()
        return item

    async def list_mix_items(self, mix_id: uuid.UUID) -> list[MixItem]:
        res = await self.db.execute(select(MixItem).where(MixItem.mix_id == mix_id))
        return list(res.scalars().all())

    # --- mix_revisions ---
    async def create_revision(
        self,
        *,
        mix_id: uuid.UUID,
        revision_no: int,
        switchover_ms: int,
        length_ms: int,
        audio_asset_id: uuid.UUID,
    ) -> MixRevision:
        rev = MixRevision(
            mix_id=mix_id,
            revision_no=revision_no,
            switchover_ms=switchover_ms,
            length_ms=length_ms,
            audio_asset_id=audio_asset_id,
        )
        self.db.add(rev)
        await self.db.flush()
        return rev

    # --- mix_segments (not needed for placeholder, but keep for later) ---
    async def add_segments(
        self,
        *,
        mix_revision_id: uuid.UUID,
        segments: Iterable[dict],
    ) -> list[MixSegment]:
        """
        segments: iterable of dicts like:
          {
            "position": 0,
            "mix_item_id": <uuid>,
            "start_ms": 0,
            "end_ms": 12345,
            "source_start_ms": 0
          }
        """
        rows: list[MixSegment] = []
        for seg in segments:
            row = MixSegment(
                mix_revision_id=mix_revision_id,
                position=int(seg["position"]),
                mix_item_id=seg["mix_item_id"],
                start_ms=int(seg["start_ms"]),
                end_ms=int(seg["end_ms"]),
                source_start_ms=int(seg["source_start_ms"]),
            )
            self.db.add(row)
            rows.append(row)

        await self.db.flush()
        return rows
    
    async def get_revision_by_no(self, mix_id: uuid.UUID, revision_no: int) -> MixRevision | None:
        res = await self.db.execute(
            select(MixRevision).where(
                MixRevision.mix_id == mix_id,
                MixRevision.revision_no == revision_no,
            )
        )
        return res.scalar_one_or_none()

    async def get_current_ready_revision(self, mix_id: uuid.UUID) -> MixRevision | None:
        mix = await self.get_mix(mix_id)
        if mix is None or mix.current_ready_revision_no is None:
            return None
        return await self.get_revision_by_no(mix_id, mix.current_ready_revision_no)

    async def list_segments_for_revision(self, mix_revision_id: uuid.UUID) -> list[MixSegment]:
        res = await self.db.execute(
            select(MixSegment)
            .where(MixSegment.mix_revision_id == mix_revision_id)
            .order_by(MixSegment.position.asc())
        )
        return list(res.scalars().all())

    async def get_items_by_ids(self, item_ids: list[uuid.UUID]) -> list[MixItem]:
        if not item_ids:
            return []
        res = await self.db.execute(select(MixItem).where(MixItem.id.in_(item_ids)))
        return list(res.scalars().all())
    
    
    async def list_segments_with_item_metadata(self, mix_revision_id: uuid.UUID):
        """
        Returns rows with:
          position, mix_item_id, start_ms, end_ms, source_start_ms, metadata_json
        """
        stmt = (
            select(
                MixSegment.position,
                MixSegment.mix_item_id,
                MixSegment.start_ms,
                MixSegment.end_ms,
                MixSegment.source_start_ms,
                MixItem.metadata_json,
            )
            .join(MixItem, MixItem.id == MixSegment.mix_item_id)
            .where(MixSegment.mix_revision_id == mix_revision_id)
            .order_by(MixSegment.position.asc())
        )
        res = await self.db.execute(stmt)
        return res.all()  # list of Row tuples

    async def get_segment_with_item_metadata_at_playhead(self, mix_revision_id: uuid.UUID, playhead_ms: int):
        """
        Returns one row with:
          position, mix_item_id, start_ms, end_ms, source_start_ms, metadata_json
        for the segment that contains playhead_ms.
        """
        stmt = (
            select(
                MixSegment.position,
                MixSegment.mix_item_id,
                MixSegment.start_ms,
                MixSegment.end_ms,
                MixSegment.source_start_ms,
                MixItem.metadata_json,
            )
            .join(MixItem, MixItem.id == MixSegment.mix_item_id)
            .where(
                MixSegment.mix_revision_id == mix_revision_id,
                MixSegment.start_ms <= playhead_ms,
                MixSegment.end_ms > playhead_ms,
            )
            .order_by(MixSegment.position.asc())
            .limit(1)
        )
        res = await self.db.execute(stmt)
        return res.first()
