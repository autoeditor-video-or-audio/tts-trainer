"""Thin MinIO wrapper for the tts-trainer service.

Local volumes are working cache only; MinIO is the source of truth
for datasets, run checkpoints, and exported voicepacks.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Optional

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)


def _require(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(f"required env var {name} is empty")
    return val


MINIO_ENDPOINT = _require("MINIO_ENDPOINT") if os.environ.get("MINIO_ENDPOINT") else ""
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "tts-training")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() in ("1", "true", "yes")


class Storage:
    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: Optional[str] = None,
        secure: Optional[bool] = None,
    ) -> None:
        self.endpoint = endpoint or MINIO_ENDPOINT or _require("MINIO_ENDPOINT")
        self.access_key = access_key or MINIO_ACCESS_KEY or _require("MINIO_ACCESS_KEY")
        self.secret_key = secret_key or MINIO_SECRET_KEY or _require("MINIO_SECRET_KEY")
        self.bucket = bucket or MINIO_BUCKET
        self.secure = MINIO_SECURE if secure is None else secure
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )

    def health(self) -> bool:
        try:
            self.client.bucket_exists(self.bucket)
            return True
        except Exception:
            logger.exception("MinIO health probe failed")
            return False

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def put_file(self, key: str, local_path: Path, content_type: str = "application/octet-stream") -> None:
        self.client.fput_object(self.bucket, key, str(local_path), content_type=content_type)

    def get_file(self, key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.fget_object(self.bucket, key, str(local_path))

    def list(self, prefix: str) -> Iterable[dict]:
        for obj in self.client.list_objects(self.bucket, prefix=prefix, recursive=True):
            yield {
                "key": obj.object_name,
                "size": obj.size,
                "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                "etag": obj.etag,
            }

    def exists(self, key: str) -> bool:
        try:
            self.client.stat_object(self.bucket, key)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            raise

    def presigned_get(self, key: str, expires_minutes: int = 15) -> str:
        return self.client.presigned_get_object(
            self.bucket, key, expires=timedelta(minutes=expires_minutes)
        )

    def remove(self, key: str) -> None:
        self.client.remove_object(self.bucket, key)


_storage: Optional[Storage] = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage
