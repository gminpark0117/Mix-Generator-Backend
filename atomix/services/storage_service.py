from __future__ import annotations

import mimetypes
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import UploadFile
from atomix.core import settings

Kind = Literal["source", "mix"]


@dataclass(frozen=True)
class StoredObject:
    """
    key: relative path under STORAGE_DIR (e.g. "source/2026/02/02/<uuid>.mp3")
    abs_path: absolute filesystem path to the stored file
    url: public URL path (e.g. "/storage/source/2026/02/02/<uuid>.mp3")
    mime: MIME type string used for Content-Type and stored in DB
    """
    key: str
    abs_path: str
    url: str
    mime: str


class StorageService:
    """
    Local filesystem storage.

    Guarantees:
    - Generates safe file keys (no user path traversal)
    - Writes atomically (tmp file + replace)
    - Produces a URL your client can GET directly (served under /storage with byte-range support)
    """

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        base_url: str | None = None,
    ):
        self.storage_dir = Path(storage_dir or settings.STORAGE_DIR)
        self.base_url = (base_url or settings.STORAGE_BASE_URL).rstrip("/")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # ---------- public API ----------

    async def save_upload(
        self,
        upload: UploadFile,
        *,
        kind: Kind,
        ext: Optional[str] = None,
    ) -> StoredObject:
        """
        Save an uploaded file. Returns storage key + public URL.

        ext: optional override extension (".mp3"). If not provided:
             - uses upload.filename suffix if present, else derives from mime.
        """
        mime = self._resolve_mime(upload.filename, upload.content_type)
        suffix = ext or self._resolve_suffix(upload.filename, mime)
        key = self._make_key(kind=kind, suffix=suffix)
        abs_path = self.storage_dir / key
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        # Stream-write to disk (avoid reading entire file into memory)
        tmp_path = abs_path.with_suffix(abs_path.suffix + ".tmp")

        # Ensure stream at start (in case caller reused UploadFile)
        try:
            await upload.seek(0)
        except Exception:
            # not fatal; some backends may not support seek
            pass

        await self._write_upload_to_path(upload, tmp_path)
        os.replace(tmp_path, abs_path)  # atomic on Windows

        return StoredObject(
            key=key.replace("\\", "/"),
            abs_path=str(abs_path),
            url=self.public_url(key),
            mime=mime,
        )

    async def save_bytes(
        self,
        data: bytes,
        *,
        kind: Kind,
        mime: str,
        ext: Optional[str] = None,
    ) -> StoredObject:
        """
        Save in-memory bytes (useful for rendered mix output).
        """
        suffix = ext or self._suffix_from_mime(mime) or ".bin"
        key = self._make_key(kind=kind, suffix=suffix)
        abs_path = self.storage_dir / key
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = abs_path.with_suffix(abs_path.suffix + ".tmp")
        tmp_path.write_bytes(data)
        os.replace(tmp_path, abs_path)

        return StoredObject(
            key=key.replace("\\", "/"),
            abs_path=str(abs_path),
            url=self.public_url(key),
            mime=mime,
        )

    def public_url(self, key: str) -> str:
        key_norm = key.replace("\\", "/").lstrip("/")
        return f"{self.base_url}/{key_norm}"

    def abs_path(self, key: str) -> Path:
        # key should be relative to STORAGE_DIR
        key_norm = key.replace("\\", "/").lstrip("/")
        return self.storage_dir / key_norm

    # ---------- internals ----------

    def _make_key(self, *, kind: Kind, suffix: str) -> str:
        # shard by date to avoid huge directories
        now = datetime.now(timezone.utc)
        date_prefix = now.strftime("%Y/%m/%d")
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        return f"{kind}/{date_prefix}/{uuid.uuid4().hex}{safe_suffix}"

    def _resolve_mime(self, filename: Optional[str], content_type: Optional[str]) -> str:
        if content_type and content_type != "application/octet-stream":
            return content_type
        if filename:
            guess, _ = mimetypes.guess_type(filename)
            if guess:
                return guess
        return "application/octet-stream"

    def _resolve_suffix(self, filename: Optional[str], mime: str) -> str:
        if filename:
            suf = Path(filename).suffix
            if suf and len(suf) <= 10:
                return suf
        return self._suffix_from_mime(mime) or ".bin"

    def _suffix_from_mime(self, mime: str) -> Optional[str]:
        # common audio types; extend as needed
        if mime in ("audio/mpeg", "audio/mp3"):
            return ".mp3"
        if mime == "audio/wav":
            return ".wav"
        if mime in ("audio/flac",):
            return ".flac"
        if mime in ("audio/ogg", "application/ogg"):
            return ".ogg"
        return None

    async def _write_upload_to_path(self, upload: UploadFile, path: Path) -> None:
        # Stream from UploadFile to disk
        chunk_size = 1024 * 1024  # 1MB
        with path.open("wb") as f:
            while True:
                chunk = await upload.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
