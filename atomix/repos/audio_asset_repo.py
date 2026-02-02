from __future__ import annotations

import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atomix.models import AudioAsset


class AudioAssetRepo:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, *, kind: str, storage_url: str, mime: str) -> AudioAsset:
        asset = AudioAsset(
            kind=kind,
            storage_url=storage_url,
            mime=mime,
        )
        self.db.add(asset)
        await self.db.flush()  # assign asset.id
        return asset
    
    async def get(self, asset_id: uuid.UUID) -> AudioAsset | None:
        res = await self.db.execute(select(AudioAsset).where(AudioAsset.id == asset_id))
        return res.scalar_one_or_none()
