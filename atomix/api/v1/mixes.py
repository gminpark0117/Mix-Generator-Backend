import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from atomix.core import get_db
from atomix.models import Mix
from atomix.schemas.mix import MixRevisionResponse, MixStateOut
from atomix.services.mix_service import MixService
router = APIRouter()

@router.post("", response_model=MixRevisionResponse)
async def create_mix(
    files: list[UploadFile] = File(...),
    tracks_metadata: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    svc = MixService(db)
    try:
        return await svc.create_mix(files=files, tracks_metadata=tracks_metadata)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{mix_id}/tracks:upload", response_model=MixRevisionResponse)
async def add_tracks_to_mix(
    mix_id: uuid.UUID,
    client_playhead_ms: int = Form(...),
    files: list[UploadFile] = File(...),
    tracks_metadata: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    svc = MixService(db)
    try:
        return await svc.add_tracks_to_mix(
            mix_id=mix_id,
            client_playhead_ms=client_playhead_ms,
            files=files,
            tracks_metadata=tracks_metadata,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="mix not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{mix_id}", response_model=MixStateOut)
async def get_mix(
    mix_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> MixStateOut:
    svc = MixService(db)
    try:
        return await svc.get_mix_state(mix_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="mix not found")

@router.get("/{mix_id}/revisions/{revision_no}", response_model=MixRevisionResponse)
async def get_mix_revision(
    mix_id: uuid.UUID,
    revision_no: int,
    db: AsyncSession = Depends(get_db),
):
    svc = MixService(db)
    try:
        return await svc.get_mix_revision(mix_id, revision_no)
    except KeyError:
        raise HTTPException(status_code=404, detail="mix revision not found")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))