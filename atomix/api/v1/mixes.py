import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from atomix.core import get_db
from atomix.models import Mix
from atomix.schemas.mix import MixStateOut
router = APIRouter()

@router.post("")
async def create_mix(
    files: list[UploadFile] = File(...),
    tracks_metadata: str = Form(...),  # JSON string array aligned to files
):
    # TODO: parse tracks_metadata, store assets, analyze, render rev1, write DB
    return {"todo": True}

@router.post("/{mix_id}/tracks:upload")
async def add_tracks_to_mix(
    mix_id: str,
    client_playhead_ms: int = Form(...),
    files: list[UploadFile] = File(...),
    tracks_metadata: str = Form(...),
):
    # TODO: store/analyze items, compute switchover = playhead + LOOKAHEAD_MS, render new revision
    return {"todo": True}

@router.get("/{mix_id}", response_model=MixStateOut)
async def get_mix(
    mix_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> MixStateOut:
    mix = await db.get(Mix, mix_id)
    if mix is None:
        raise HTTPException(status_code=404, detail="mix not found")

    return MixStateOut(
        mix_id=str(mix.id),
        current_ready_revision_no=mix.current_ready_revision_no,
    )

@router.get("/{mix_id}/revisions/{revision_no}")
async def get_mix_revision(mix_id: str, revision_no: int):
    # TODO: fetch revision + segments (join mix_items metadata) and return
    return {"todo": True}

