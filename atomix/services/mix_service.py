from __future__ import annotations

import json
import uuid
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from atomix.repos.audio_asset_repo import AudioAssetRepo
from atomix.repos.mix_repo import MixRepo
from atomix.services.storage_service import StorageService
from atomix.renderers.mix_renderer import PlaceholderMixRenderer, TrackInput
from atomix.analyzers.mix_analyzer import MixAnalyzer
from atomix.schemas.mix import MixStateOut, MixRevisionResponse, RevisionOut, SegmentOut
from atomix.core.config import settings


def parse_tracks_metadata(tracks_metadata: str, n_files: int) -> list[dict]:
    try:
        data = json.loads(tracks_metadata)
    except json.JSONDecodeError as e:
        raise ValueError(f"tracks_metadata must be valid JSON: {e}")

    if not isinstance(data, list):
        raise ValueError("tracks_metadata must be a JSON array")

    if len(data) != n_files:
        raise ValueError(f"tracks_metadata length ({len(data)}) must match files length ({n_files})")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"tracks_metadata[{i}] must be an object")
        if "song_name" not in item or "artist_name" not in item:
            raise ValueError(f"tracks_metadata[{i}] must contain song_name and artist_name")
        if not isinstance(item["song_name"], str) or not isinstance(item["artist_name"], str):
            raise ValueError(f"tracks_metadata[{i}].song_name/artist_name must be strings")

    return data


class MixService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.assets = AudioAssetRepo(db)
        self.mixes = MixRepo(db)

        self.storage = StorageService()
        self.renderer = PlaceholderMixRenderer()  # swap later with real renderer
        self.analyzer = MixAnalyzer()

    async def create_mix(self, *, files: list[UploadFile], tracks_metadata: str) -> MixRevisionResponse:
        metas = parse_tracks_metadata(tracks_metadata, len(files))

        async with self.db.begin():
            # 1) create mix
            mix = await self.mixes.create_mix()

            # 2) persist sources => audio_assets(kind=source) + mix_items
            track_inputs: list[TrackInput] = []

            for up, meta in zip(files, metas):
                stored = await self.storage.save_upload(up, kind="source")
                src_asset = await self.assets.create(
                    kind="source",
                    storage_url=stored.url,
                    mime=stored.mime,
                )
                analysis_result = await self.analyzer.analyze(stored.abs_path, meta)
                item = await self.mixes.add_mix_item(
                    mix_id=mix.id,
                    source_audio_asset_id=src_asset.id,
                    metadata_json=analysis_result.metadata_json,
                    analysis_json=analysis_result.analysis_json,
                )
                track_inputs.append(TrackInput(
                    mix_item_id=item.id,
                    abs_path=stored.abs_path,
                    metadata_json=analysis_result.metadata_json,
                    analysis_json=analysis_result.analysis_json,
                ))

            # 3) render mix (placeholder)
            render_result = await self.renderer.render(track_inputs, [])

            # 4) persist rendered mix => audio_assets(kind=mix)
            mix_obj = await self.storage.save_bytes(
                render_result.audio_bytes,
                kind="mix",
                mime=render_result.mime,
                ext=".wav",
            )
            mix_asset = await self.assets.create(
                kind="mix",
                storage_url=mix_obj.url,
                mime=mix_obj.mime,
            )

            # 5) create revision 1
            rev = await self.mixes.create_revision(
                mix_id=mix.id,
                revision_no=1,
                switchover_ms=0,
                length_ms=render_result.length_ms,
                audio_asset_id=mix_asset.id,
            )

            # 6) create segments for rev1
            await self.mixes.add_segments(
                mix_revision_id=rev.id,
                segments=[
                    {
                        "position": s.position,
                        "mix_item_id": s.mix_item_id,
                        "start_ms": s.start_ms,
                        "end_ms": s.end_ms,
                        "source_start_ms": s.source_start_ms,
                    }
                    for s in render_result.segments
                ],
            )

            # 7) update mix pointer
            await self.mixes.set_current_ready_revision_no(mix, 1)

        # Build response tracklist from metas + segment specs
        # (In a later iteration, you can fetch segments from DB and join metadata.)
        seg_out: list[SegmentOut] = [
            SegmentOut(
                position=s.position,
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                source_start_ms=s.source_start_ms,
                song_name=s.song_name,
                artist_name=s.artist_name,
            )
            for s in render_result.segments
        ]

        return MixRevisionResponse(
            mix_id=str(mix.id),
            revision=RevisionOut(
                revision_no=1,
                switchover_ms=0,
                length_ms=render_result.length_ms,
                audio_url=mix_obj.url,
            ),
            tracklist=seg_out,
        )

    async def get_mix_state(self, mix_id: uuid.UUID) -> MixStateOut:
        mix = await self.mixes.get_mix(mix_id)
        if mix is None:
            raise KeyError("mix not found")

        return MixStateOut(
            mix_id=str(mix.id),
            current_ready_revision_no=mix.current_ready_revision_no,
        )

    async def get_mix_revision(self, mix_id: uuid.UUID, revision_no: int) -> MixRevisionResponse:
        rev = await self.mixes.get_revision_by_no(mix_id, revision_no)
        if rev is None:
            raise KeyError("revision not found")

        asset = await self.assets.get(rev.audio_asset_id)
        if asset is None:
            raise RuntimeError("audio asset missing for revision")

        rows = await self.mixes.list_segments_with_item_metadata(rev.id)

        tracklist: list[SegmentOut] = []
        for (position, mix_item_id, start_ms, end_ms, source_start_ms, metadata_json) in rows:
            meta = metadata_json or {}
            tracklist.append(
                SegmentOut(
                    position=position,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    source_start_ms=source_start_ms,
                    song_name=str(meta.get("song_name", "")),
                    artist_name=str(meta.get("artist_name", "")),
                )
            )

        return MixRevisionResponse(
            mix_id=str(mix_id),
            revision=RevisionOut(
                revision_no=rev.revision_no,
                switchover_ms=rev.switchover_ms,
                length_ms=rev.length_ms,
                audio_url=asset.storage_url,
            ),
            tracklist=tracklist,
        )

    async def add_tracks_to_mix(
        self,
        *,
        mix_id: uuid.UUID,
        client_playhead_ms: int,
        files: list[UploadFile],
        tracks_metadata: str,
    ) -> MixRevisionResponse:
        """Add new tracks to an existing mix and render a new revision."""
        metas = parse_tracks_metadata(tracks_metadata, len(files))

        async with self.db.begin():
            # Get current mix state
            mix = await self.mixes.get_mix(mix_id)
            if mix is None:
                raise KeyError("mix not found")

            # Get current revision for reference
            if mix.current_ready_revision_no is None:
                raise ValueError("mix has no current revision")
            
            current_rev = await self.mixes.get_revision_by_no(mix_id, mix.current_ready_revision_no)
            if current_rev is None:
                raise RuntimeError("current revision not found")

            # 1) Persist new source files and add to track_inputs
            track_inputs: list[TrackInput] = []
            new_item_ids: set[uuid.UUID] = set()

            for up, meta in zip(files, metas):
                stored = await self.storage.save_upload(up, kind="source")
                src_asset = await self.assets.create(
                    kind="source",
                    storage_url=stored.url,
                    mime=stored.mime,
                )
                # Analyze track
                analysis_result = await self.analyzer.analyze(stored.abs_path, meta)
                item = await self.mixes.add_mix_item(
                    mix_id=mix_id,
                    source_audio_asset_id=src_asset.id,
                    metadata_json=analysis_result.metadata_json,
                    analysis_json=analysis_result.analysis_json,
                )
                new_item_ids.add(item.id)
                track_inputs.append(TrackInput(
                    mix_item_id=item.id,
                    abs_path=stored.abs_path,
                    metadata_json=analysis_result.metadata_json,
                    analysis_json=analysis_result.analysis_json,
                ))

            # 2) Get all mix items (old + new)
            all_items = await self.mixes.list_mix_items(mix_id)
            
            for item in all_items:
                # Skip new items (already in track_inputs)
                if item.id not in new_item_ids:
                    # Fetch audio asset to get storage_url, then reconstruct abs_path
                    asset = await self.assets.get(item.source_audio_asset_id)
                    if asset is None:
                        continue  # skip if asset missing
                    
                    # Extract key from storage_url by removing base_url prefix
                    key = asset.storage_url.removeprefix(self.storage.base_url).lstrip("/")
                    abs_path = str(self.storage.abs_path(key))
                    
                    track_inputs.append(TrackInput(
                        mix_item_id=item.id,
                        abs_path=abs_path,
                        metadata_json=item.metadata_json,
                        analysis_json=item.analysis_json,
                    ))

            # 3) Determine fixed tracks
            freeze_ms = client_playhead_ms + settings.LOOKAHEAD_MS
            fixed_track_ids: set[uuid.UUID] = set()

            current_segments = await self.mixes.list_segments_for_revision(current_rev.id)
            for seg in current_segments:
                if seg.start_ms < freeze_ms:
                    fixed_track_ids.add(seg.mix_item_id)

            fixed_tracks = [t for t in track_inputs if t.mix_item_id in fixed_track_ids]

            # 4) Render new mix
            render_result = await self.renderer.render(track_inputs, fixed_tracks)

            # 5) Persist rendered mix => audio_assets(kind=mix)
            mix_obj = await self.storage.save_bytes(
                render_result.audio_bytes,
                kind="mix",
                mime=render_result.mime,
                ext=".wav",
            )
            mix_asset = await self.assets.create(
                kind="mix",
                storage_url=mix_obj.url,
                mime=mix_obj.mime,
            )

            # 6) Create new revision
            new_revision_no = mix.current_ready_revision_no + 1
            rev = await self.mixes.create_revision(
                mix_id=mix_id,
                revision_no=new_revision_no,
                switchover_ms=freeze_ms,
                length_ms=render_result.length_ms,
                audio_asset_id=mix_asset.id,
            )

            # 7) Create segments for new revision
            await self.mixes.add_segments(
                mix_revision_id=rev.id,
                segments=[
                    {
                        "position": s.position,
                        "mix_item_id": s.mix_item_id,
                        "start_ms": s.start_ms,
                        "end_ms": s.end_ms,
                        "source_start_ms": s.source_start_ms,
                    }
                    for s in render_result.segments
                ],
            )

            # 8) Update mix pointer
            await self.mixes.set_current_ready_revision_no(mix, new_revision_no)

        # Build response
        seg_out: list[SegmentOut] = [
            SegmentOut(
                position=s.position,
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                source_start_ms=s.source_start_ms,
                song_name=s.song_name,
                artist_name=s.artist_name,
            )
            for s in render_result.segments
        ]

        return MixRevisionResponse(
            mix_id=str(mix_id),
            revision=RevisionOut(
                revision_no=new_revision_no,
                switchover_ms=freeze_ms,
                length_ms=render_result.length_ms,
                audio_url=mix_obj.url,
            ),
            tracklist=seg_out,
        )
