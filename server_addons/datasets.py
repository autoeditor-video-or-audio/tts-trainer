"""Dataset adapters for the tts-trainer service.

Each adapter downloads a corpus, normalises it into a working layout
(`wavs/<id>.wav`, `metadata.csv` with `filename|transcript|speaker`),
and uploads the result to MinIO at
`s3://<bucket>/datasets/<id>/...`.

Common Voice PT-BR is the default path; CETUC / CORAA stubs return a
clear `NotImplementedError` until implemented.
"""

from __future__ import annotations

import csv
import logging
import os
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from server_addons.storage import get_storage

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(os.environ.get("TTS_TRAINER_DATA_DIR", "/app/data")).resolve()
DATASETS_DIR = DEFAULT_DATA_DIR / "datasets"
TARGET_SAMPLE_RATE = 22050  # StyleTTS2 / Kokoro standard


@dataclass
class DatasetInfo:
    id: str
    hours: Optional[float]
    speakers: Optional[int]
    downloaded_at: Optional[str]
    status: str  # "downloading" | "ready" | "failed"


def _key(dataset_id: str, suffix: str) -> str:
    return f"datasets/{dataset_id}/{suffix}"


def _local_root(dataset_id: str) -> Path:
    return DATASETS_DIR / dataset_id


def is_ready(dataset_id: str) -> bool:
    """True if MinIO holds a packaged tarball under datasets/<id>/."""
    storage = get_storage()
    return storage.exists(_key(dataset_id, "data.tar.zst")) or storage.exists(
        _key(dataset_id, "ready.json")
    )


def list_local() -> list[DatasetInfo]:
    storage = get_storage()
    seen: dict[str, DatasetInfo] = {}
    for entry in storage.list(prefix="datasets/"):
        key = entry["key"]  # e.g. datasets/commonvoice_ptbr/ready.json
        parts = key.split("/")
        if len(parts) < 3:
            continue
        ds_id = parts[1]
        if ds_id in seen:
            continue
        downloaded_at = entry.get("last_modified")
        status = "ready" if key.endswith("ready.json") or key.endswith("data.tar.zst") else "downloading"
        seen[ds_id] = DatasetInfo(
            id=ds_id,
            hours=None,
            speakers=None,
            downloaded_at=downloaded_at,
            status=status,
        )
    return list(seen.values())


def _stream_and_pack(
    dataset_id: str,
    *,
    iterable_rows,
    audio_field: str = "audio",
    text_field: str = "sentence",
    speaker_field: str = "client_id",
    label: str,
) -> DatasetInfo:
    """Shared download / resample / pack / push pipeline.

    `iterable_rows` is an iterable of dict-like records from
    `datasets.load_dataset(..., streaming=True)`.
    """
    import soundfile as sf
    import numpy as np

    storage = get_storage()
    local_root = _local_root(dataset_id)
    wavs_dir = local_root / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = local_root / "metadata.csv"

    n_rows = 0
    total_seconds = 0.0
    speakers: set[str] = set()
    with metadata_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|")
        for row in iterable_rows:
            audio = row.get(audio_field)
            sentence = (row.get(text_field) or "").strip()
            speaker = str(row.get(speaker_field) or "unknown")
            if not audio or not sentence:
                continue
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
            if sr != TARGET_SAMPLE_RATE:
                import librosa
                arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
                sr = TARGET_SAMPLE_RATE
            n_rows += 1
            file_stem = f"{n_rows:06d}"
            wav_path = wavs_dir / f"{file_stem}.wav"
            sf.write(str(wav_path), arr, sr, subtype="PCM_16")
            duration = float(arr.shape[0]) / sr
            total_seconds += duration
            speakers.add(speaker)
            writer.writerow([f"wavs/{file_stem}.wav", sentence, speaker])
            if n_rows % 500 == 0:
                logger.info(
                    "[%s] %d rows (%.1f h, %d speakers)",
                    label,
                    n_rows,
                    total_seconds / 3600.0,
                    len(speakers),
                )

    hours = total_seconds / 3600.0
    logger.info("[%s] packing (%.2f h, %d speakers)", label, hours, len(speakers))

    # Tarball + push to MinIO.
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp_tar:
        tmp_tar_path = Path(tmp_tar.name)
    try:
        with tarfile.open(tmp_tar_path, "w") as tar:
            tar.add(local_root, arcname=dataset_id)
        storage.put_file(
            _key(dataset_id, "data.tar"),
            tmp_tar_path,
            content_type="application/x-tar",
        )
    finally:
        tmp_tar_path.unlink(missing_ok=True)

    ready_payload = {
        "id": dataset_id,
        "hours": hours,
        "speakers": len(speakers),
        "rows": n_rows,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "sample_rate": TARGET_SAMPLE_RATE,
    }
    ready_path = local_root / "ready.json"
    import json
    ready_path.write_text(json.dumps(ready_payload, indent=2))
    storage.put_file(_key(dataset_id, "ready.json"), ready_path, content_type="application/json")

    return DatasetInfo(
        id=dataset_id,
        hours=hours,
        speakers=len(speakers),
        downloaded_at=ready_payload["downloaded_at"],
        status="ready",
    )


def _already_ready(dataset_id: str) -> Optional[DatasetInfo]:
    storage = get_storage()
    if storage.exists(_key(dataset_id, "ready.json")):
        logger.info("dataset %s already present in MinIO; skipping", dataset_id)
        return DatasetInfo(
            id=dataset_id,
            hours=None,
            speakers=None,
            downloaded_at=datetime.now(timezone.utc).isoformat(),
            status="ready",
        )
    return None


def download_common_voice_ptbr(dataset_id: str = "commonvoice_ptbr") -> DatasetInfo:
    """Common Voice PT-BR via HF (gated dataset — requires HF_TOKEN env)."""
    cached = _already_ready(dataset_id)
    if cached:
        return cached
    from datasets import load_dataset
    logger.info("downloading mozilla-foundation/common_voice_19_0 pt (streaming)")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Common Voice is gated on HuggingFace. Visit "
            "https://huggingface.co/datasets/mozilla-foundation/common_voice_19_0 "
            "to accept the terms, then set HF_TOKEN in the trainer .env."
        )
    ds = load_dataset(
        "mozilla-foundation/common_voice_19_0",
        "pt",
        split="train",
        streaming=True,
        trust_remote_code=False,
        token=token,
    )
    return _stream_and_pack(
        dataset_id,
        iterable_rows=ds,
        audio_field="audio",
        text_field="sentence",
        speaker_field="client_id",
        label="commonvoice_ptbr",
    )


def download_mls_ptbr(dataset_id: str = "mls_ptbr") -> DatasetInfo:
    """Multilingual LibriSpeech Portuguese — public, ~160h.

    HF mirror: facebook/multilingual_librispeech config 'portuguese'.
    """
    cached = _already_ready(dataset_id)
    if cached:
        return cached
    from datasets import load_dataset
    logger.info("downloading facebook/multilingual_librispeech portuguese (streaming)")
    ds = load_dataset(
        "facebook/multilingual_librispeech",
        "portuguese",
        split="train",
        streaming=True,
        trust_remote_code=False,
    )
    return _stream_and_pack(
        dataset_id,
        iterable_rows=ds,
        audio_field="audio",
        text_field="transcript",
        speaker_field="speaker_id",
        label="mls_ptbr",
    )


def download_dataset(dataset_id: str) -> DatasetInfo:
    if dataset_id == "commonvoice_ptbr":
        return download_common_voice_ptbr(dataset_id)
    if dataset_id == "mls_ptbr":
        return download_mls_ptbr(dataset_id)
    if dataset_id in {"cetuc", "coraa", "fleurs_ptbr"}:
        # FLEURS removed: datasets>=2.20 banned legacy loading scripts
        # (google/fleurs still ships a fleurs.py loader). Bring it back
        # via a parquet-based adapter if needed; for now MLS covers the
        # smoke + full-scale paths.
        raise NotImplementedError(
            f"adapter for {dataset_id!r} is not implemented yet — file an issue or contribute one"
        )
    if dataset_id.startswith("hf:"):
        raise NotImplementedError(
            "generic HuggingFace adapter not implemented yet; use commonvoice_ptbr for now"
        )
    raise ValueError(f"unknown dataset id: {dataset_id!r}")
