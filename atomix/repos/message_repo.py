from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

class MessageRepo:
    def __init__(self, db: AsyncSession):
        self.db = db

    # add insert/list later
