from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from atomix.core.config import settings
from collections.abc import AsyncGenerator

engine = create_async_engine(settings.DATABASE_URL_ASYNC, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
