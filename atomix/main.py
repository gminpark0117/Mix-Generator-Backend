from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import asyncio

from atomix.api.v1.router import router as v1_router
from atomix.api.v1.ws_rooms import _cleanup_empty_rooms
from atomix.core import settings

app = FastAPI(title="atomix API", version="0.1.0")
app.include_router(v1_router, prefix="/v1")

Path(settings.STORAGE_DIR).mkdir(parents=True, exist_ok=True)
app.mount(settings.STORAGE_BASE_URL, StaticFiles(directory=settings.STORAGE_DIR), name="storage")


@app.on_event("startup")
async def startup_event():
    """Start background cleanup task for empty rooms."""
    asyncio.create_task(_cleanup_empty_rooms())
