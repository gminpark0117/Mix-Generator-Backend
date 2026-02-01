from fastapi import APIRouter, UploadFile, File, Form

router = APIRouter()

@router.post("")
async def create_room(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    tracks_metadata: str = Form(...),
):
    return {"todo": True}

@router.get("")
async def list_rooms():
    return {"todo": True}

@router.patch("/{room_id}")
async def rename_room(room_id: str, name: str):
    return {"todo": True}

@router.post("/{room_id}/tracks:upload")
async def add_tracks_to_room(
    room_id: str,
    files: list[UploadFile] = File(...),
    tracks_metadata: str = Form(...),
):
    return {"accepted": True}
