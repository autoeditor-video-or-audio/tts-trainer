"""StyleTTS2 → Kokoro voicepack export.

Kokoro voicepacks are speaker-embedding tensors with shape
`(511, 1, 256)` stored as `voices/<id>.pt` and consumed by the
kokoro-fastapi inference container via `torch.load(weights_only=True)`.

The extraction reads the final StyleTTS2 checkpoint produced by
`train_finetune.py`, pulls the trained style/speaker embedding from
the model state dict, and reshapes it to the Kokoro contract. The
result is shape-validated before the upload step succeeds.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from server_addons.storage import get_storage
from server_addons.training import DEFAULT_DATA_DIR, RUNS_DIR

logger = logging.getLogger(__name__)

VOICEPACK_SHAPE = (511, 1, 256)
VOICEPACKS_DIR = DEFAULT_DATA_DIR / "voicepacks"
SHARED_VOICES_DIR = Path(os.environ.get("KOKORO_VOICES_DIR", "/app/data/kokoro-voices"))


class ShapeMismatch(Exception):
    def __init__(self, actual_shape: tuple, expected_shape: tuple = VOICEPACK_SHAPE) -> None:
        super().__init__(
            f"voicepack tensor shape mismatch: got {tuple(actual_shape)}, expected {tuple(expected_shape)}"
        )
        self.actual_shape = tuple(int(x) for x in actual_shape)


def _find_final_checkpoint(run_id: str) -> Path:
    """Pick the highest-numbered epoch_*.pth under runs/<run_id>/logs."""
    log_dir = RUNS_DIR / run_id / "logs"
    if not log_dir.exists():
        raise FileNotFoundError(f"run logs missing at {log_dir}")
    candidates = sorted(log_dir.glob("epoch_*.pth"))
    if not candidates:
        # Some StyleTTS2 configs write a single `final.pth`.
        finals = list(log_dir.glob("*.pth"))
        if not finals:
            raise FileNotFoundError(f"no .pth checkpoints under {log_dir}")
        return finals[-1]
    return candidates[-1]


def _extract_speaker_embedding(ckpt_path: Path):
    """Pull the (511, 1, 256) speaker embedding out of a StyleTTS2 checkpoint.

    StyleTTS2's style encoder produces a 256-dim style vector per
    sample; Kokoro stores 511 different timestep-conditioned variants
    of that style. The empirical extraction (community-derived from
    `huggingface.co/datasets/ecyht2/kokoro-82M-voices`) is:

      1. Load the checkpoint state dict.
      2. Look for `style_encoder` or `style_predictor` weights;
         their final linear layer holds the 256-dim style basis.
      3. Tile / repeat the basis to produce 511 timesteps the
         Kokoro decoder expects.

    Implementation here is intentionally tolerant: if the checkpoint
    shape diverges (different StyleTTS2 variant), we surface a
    ShapeMismatch so the operator knows to revisit the extraction
    rule.
    """
    import torch

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ShapeMismatch(actual_shape=(0,))

    # Try a few well-known key paths from StyleTTS2 variants.
    candidates = []
    for k, v in state.items():
        if not hasattr(v, "shape"):
            continue
        shape = tuple(v.shape)
        if len(shape) == 2 and shape[-1] == 256 and "style" in k.lower():
            candidates.append((k, v))
        elif len(shape) == 1 and shape[0] == 256 and "style" in k.lower():
            candidates.append((k, v.unsqueeze(0)))
        elif len(shape) == 3 and shape == VOICEPACK_SHAPE:
            return v.detach().cpu()
    if not candidates:
        raise ShapeMismatch(actual_shape=(0,))

    name, basis = candidates[0]
    logger.info("extracting speaker embedding from %s shape=%s", name, tuple(basis.shape))
    if basis.dim() == 2:
        basis = basis.mean(dim=0, keepdim=True)  # → (1, 256)
    # Tile to (511, 1, 256). Each timestep gets the same basis vector
    # — this is the conservative export; a more sophisticated
    # extraction could interpolate timestep-specific embeddings.
    out = basis.unsqueeze(0).repeat(511, 1, 1).contiguous()
    if tuple(out.shape) != VOICEPACK_SHAPE:
        raise ShapeMismatch(actual_shape=tuple(out.shape))
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def export_voicepack(run_id: str, voicepack_name: str) -> dict:
    """Run the export pipeline. Raises ShapeMismatch on bad output."""
    import torch

    ckpt_path = _find_final_checkpoint(run_id)
    logger.info("[%s] extracting voicepack from %s", run_id, ckpt_path)
    tensor = _extract_speaker_embedding(ckpt_path)

    VOICEPACKS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = VOICEPACKS_DIR / f"{voicepack_name}.pt"
    torch.save(tensor, str(local_path))

    digest = _sha256(local_path)
    size_bytes = local_path.stat().st_size

    storage = get_storage()
    storage.put_file(
        f"voicepacks/{voicepack_name}.pt",
        local_path,
        content_type="application/octet-stream",
    )

    # Optional: drop into the bind-mounted kokoro-fastapi voices folder
    # so inference picks the new pack up without a manual copy.
    if SHARED_VOICES_DIR.exists():
        shared_path = SHARED_VOICES_DIR / f"{voicepack_name}.pt"
        torch.save(tensor, str(shared_path))

    return {
        "name": voicepack_name,
        "size_bytes": size_bytes,
        "run_id": run_id,
        "sha256": digest,
    }


def list_voicepacks() -> list[dict]:
    storage = get_storage()
    out: list[dict] = []
    for entry in storage.list(prefix="voicepacks/"):
        name = entry["key"].split("/", 1)[1].rsplit(".pt", 1)[0]
        out.append(
            {
                "name": name,
                "size_bytes": entry["size"] or 0,
                "created_at": entry.get("last_modified"),
                "run_id": None,
                "sha256": (entry.get("etag") or "").strip('"'),
            }
        )
    return out


def presign_voicepack(name: str) -> Optional[str]:
    storage = get_storage()
    key = f"voicepacks/{name}.pt"
    if not storage.exists(key):
        return None
    return storage.presigned_get(key)
