from fastapi import APIRouter
from atomix.api.v1 import mixes, rooms, ws_rooms

router = APIRouter()
router.include_router(mixes.router, prefix="/mixes", tags=["mixes"])
router.include_router(rooms.router, prefix="/rooms", tags=["rooms"])
router.include_router(ws_rooms.router, tags=["rooms-ws"])
