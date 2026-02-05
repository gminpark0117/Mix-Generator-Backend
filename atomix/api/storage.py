from __future__ import annotations

import os
from mimetypes import guess_type
from pathlib import Path
from typing import Optional, Tuple

import anyio
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from atomix.core import settings


router = APIRouter()


def _safe_storage_path(rel_path: str) -> Path:
    base = Path(settings.STORAGE_DIR).resolve()
    target = (base / rel_path).resolve()
    if not target.is_relative_to(base):
        raise HTTPException(status_code=404, detail="not found")
    return target


def _parse_range_header(range_header: str, *, size: int) -> Optional[Tuple[int, int]]:
    """
    Returns (start, end) inclusive, or None if header is missing/empty.
    Only supports a single range (bytes=start-end | bytes=start- | bytes=-suffix).
    Raises HTTPException(416) on invalid ranges.
    """
    if not range_header:
        return None

    unit, _, spec = range_header.partition("=")
    if unit.strip().lower() != "bytes" or not spec:
        raise HTTPException(status_code=416, detail="invalid range unit")

    # Multiple ranges (e.g. "0-1, 4-5") are not supported.
    if "," in spec:
        raise HTTPException(status_code=416, detail="multiple ranges not supported")

    spec = spec.strip()
    if "-" not in spec:
        raise HTTPException(status_code=416, detail="invalid range spec")

    start_s, end_s = (s.strip() for s in spec.split("-", 1))

    if start_s == "":
        # suffix range: bytes=-N (last N bytes)
        try:
            suffix_len = int(end_s)
        except ValueError:
            raise HTTPException(status_code=416, detail="invalid suffix range")
        if suffix_len <= 0:
            raise HTTPException(status_code=416, detail="invalid suffix range")
        if suffix_len >= size:
            return (0, size - 1)
        return (size - suffix_len, size - 1)

    try:
        start = int(start_s)
    except ValueError:
        raise HTTPException(status_code=416, detail="invalid range start")

    if start < 0 or start >= size:
        raise HTTPException(status_code=416, detail="range start out of bounds")

    if end_s == "":
        return (start, size - 1)

    try:
        end = int(end_s)
    except ValueError:
        raise HTTPException(status_code=416, detail="invalid range end")

    if end < start:
        raise HTTPException(status_code=416, detail="range end before start")

    end = min(end, size - 1)
    return (start, end)


async def _iter_file(path: Path, *, start: int, count: int, chunk_size: int = 64 * 1024):
    async with await anyio.open_file(path, mode="rb") as f:
        await f.seek(start)
        remaining = count
        while remaining > 0:
            chunk = await f.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


@router.get(f"{settings.STORAGE_BASE_URL}/{{rel_path:path}}")
@router.head(f"{settings.STORAGE_BASE_URL}/{{rel_path:path}}")
async def get_storage_file(rel_path: str, request: Request) -> Response:
    """
    Serve files under STORAGE_DIR with HTTP Range (bytes) support.
    """
    path = _safe_storage_path(rel_path)
    try:
        stat_result = await anyio.to_thread.run_sync(os.stat, path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="not found")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="not found")

    size = stat_result.st_size
    content_type = guess_type(str(path))[0] or "application/octet-stream"

    range_header = request.headers.get("range", "")
    try:
        byte_range = _parse_range_header(range_header, size=size)
    except HTTPException as exc:
        # Per RFC, include Content-Range: bytes */<size> for 416
        if exc.status_code == 416:
            return Response(
                status_code=416,
                headers={
                    "Content-Range": f"bytes */{size}",
                    "Accept-Ranges": "bytes",
                },
            )
        raise

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": content_type,
    }

    if request.method.upper() == "HEAD":
        # HEAD returns headers only; keep behavior aligned with GET.
        if byte_range is None:
            headers["Content-Length"] = str(size)
            return Response(status_code=200, headers=headers)
        start, end = byte_range
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Content-Length"] = str(end - start + 1)
        return Response(status_code=206, headers=headers)

    if byte_range is None:
        headers["Content-Length"] = str(size)
        return StreamingResponse(_iter_file(path, start=0, count=size), status_code=200, headers=headers)

    start, end = byte_range
    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(end - start + 1)
    return StreamingResponse(
        _iter_file(path, start=start, count=end - start + 1),
        status_code=206,
        headers=headers,
    )

