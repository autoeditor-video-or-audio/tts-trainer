"""Run orchestration for tts-trainer.

Strategy:
- ProcessPoolExecutor(max_workers=1) so only one training job runs at a
  time on the single GPU.
- Each run launches `python train_finetune.py --config_path …`
  as a subprocess.
- SQLite at /app/data/state.sqlite persists the run-state machine so
  the service can resume after a container restart.

This is intentionally minimal — we wrap the upstream StyleTTS2
training scripts unmodified; the FastAPI service just queues jobs
and watches stderr / checkpoints.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shlex
import sqlite3
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor, Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(os.environ.get("TTS_TRAINER_DATA_DIR", "/app/data")).resolve()
RUNS_DIR = DEFAULT_DATA_DIR / "runs"
STATE_DB = DEFAULT_DATA_DIR / "state.sqlite"
UPSTREAM_SCRIPT = Path(os.environ.get("TTS_TRAINER_UPSTREAM_SCRIPT", "/app/train_finetune.py"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs() -> None:
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(STATE_DB, isolation_level=None, check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            voicepack_name TEXT NOT NULL,
            language TEXT NOT NULL,
            status TEXT NOT NULL,
            epoch INTEGER,
            loss REAL,
            eta_seconds INTEGER,
            started_at TEXT,
            finished_at TEXT,
            log_tail TEXT,
            epochs INTEGER,
            lr REAL,
            created_at TEXT NOT NULL
        )"""
    )
    return conn


_db_lock = threading.Lock()
_executor: Optional[ProcessPoolExecutor] = None
_futures: dict[str, Future] = {}


def executor() -> ProcessPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=1)
    return _executor


def _put_run(row: dict) -> None:
    with _db_lock, _connect() as conn:
        cols = ",".join(row.keys())
        placeholders = ",".join(["?"] * len(row))
        conn.execute(f"INSERT OR REPLACE INTO runs ({cols}) VALUES ({placeholders})", tuple(row.values()))


def _update_run(run_id: str, **fields) -> None:
    with _db_lock, _connect() as conn:
        sets = ", ".join(f"{k}=?" for k in fields)
        conn.execute(f"UPDATE runs SET {sets} WHERE run_id=?", (*fields.values(), run_id))


def _get_run(run_id: str) -> Optional[dict]:
    with _db_lock, _connect() as conn:
        cur = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        if row is None:
            return None
        cols = [c[0] for c in cur.description]
        return dict(zip(cols, row))


def list_runs() -> list[dict]:
    with _db_lock, _connect() as conn:
        cur = conn.execute("SELECT * FROM runs ORDER BY created_at DESC")
        rows = cur.fetchall()
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in rows]


def get_run(run_id: str) -> Optional[dict]:
    return _get_run(run_id)


def _build_config(run_id: str, dataset_id: str, language: str, epochs: Optional[int], lr: Optional[float]) -> Path:
    """Render a config file the upstream StyleTTS2 trainer expects.

    The upstream `Configs/config_ft.yml` is the template; we patch
    `log_dir`, `data_params`, `epochs`, `optimizer_params.lr` for
    this run, then point `train_finetune.py --config_path` at the
    rendered copy.
    """
    import yaml

    template_path = Path("/app/Configs/config_ft.yml")
    if not template_path.exists():
        # Fall back to the upstream config_demo.yml if config_ft is absent.
        template_path = Path("/app/Configs/config.yml")
    cfg = yaml.safe_load(template_path.read_text())

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["log_dir"] = str(run_dir / "logs")
    cfg.setdefault("data_params", {})["root_path"] = str(DEFAULT_DATA_DIR / "datasets" / dataset_id)
    cfg["data_params"]["train_data"] = str(DEFAULT_DATA_DIR / "datasets" / dataset_id / "metadata.csv")
    cfg["data_params"]["val_data"] = cfg["data_params"]["train_data"]
    if epochs is not None:
        cfg["epochs"] = epochs
    if lr is not None:
        cfg.setdefault("optimizer_params", {})["lr"] = lr

    cfg_path = run_dir / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def _run_subprocess(run_id: str, cfg_path: Path) -> None:
    """Spawn `python train_finetune.py --config_path <cfg>` and stream stderr.

    Runs inside the ProcessPoolExecutor worker. Updates SQLite via
    `_update_run` on epoch boundaries (parsed from stderr lines that
    match `Epoch <n>:` / `loss <f>`).
    """
    _update_run(run_id, status="running", started_at=_now_iso())
    cmd = [
        "python",
        str(UPSTREAM_SCRIPT),
        "--config_path",
        str(cfg_path),
    ]
    logger.info("[%s] launching: %s", run_id, " ".join(shlex.quote(c) for c in cmd))
    log_path = cfg_path.parent / "train.log"
    last_tail = ""
    try:
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd="/app",
                env={**os.environ, "TRAINER_RUN_ID": run_id},
            )
            while True:
                rc = proc.poll()
                # Update tail every 5 s.
                time.sleep(5)
                try:
                    last_tail = log_path.read_text(errors="ignore")[-4096:]
                except FileNotFoundError:
                    last_tail = ""
                epoch, loss = _parse_progress(last_tail)
                _update_run(run_id, epoch=epoch, loss=loss, log_tail=last_tail)
                if rc is not None:
                    break
        if proc.returncode == 0:
            _update_run(run_id, status="ready", finished_at=_now_iso(), log_tail=last_tail)
        else:
            _update_run(
                run_id,
                status="failed",
                finished_at=_now_iso(),
                log_tail=last_tail + f"\n[exit code {proc.returncode}]",
            )
    except Exception as exc:
        logger.exception("[%s] worker crashed", run_id)
        _update_run(
            run_id, status="failed", finished_at=_now_iso(), log_tail=last_tail + f"\n[crash] {exc!r}"
        )


def _parse_progress(tail: str) -> tuple[Optional[int], Optional[float]]:
    """Best-effort progress parsing — StyleTTS2 logs are noisy."""
    epoch = None
    loss = None
    for line in reversed(tail.splitlines()):
        if "Epoch" in line and epoch is None:
            for tok in line.replace(":", " ").split():
                if tok.isdigit():
                    epoch = int(tok)
                    break
        if ("loss" in line.lower()) and loss is None:
            parts = line.lower().split("loss")
            if len(parts) > 1:
                try:
                    loss = float(parts[1].strip().split()[0].rstrip(","))
                except (IndexError, ValueError):
                    pass
        if epoch is not None and loss is not None:
            break
    return epoch, loss


def queue_run(
    dataset_id: str,
    name: str,
    voicepack_name: str,
    language: str = "pt-BR",
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
) -> str:
    run_id = f"run_{secrets.token_hex(6)}"
    _put_run(
        {
            "run_id": run_id,
            "name": name,
            "dataset_id": dataset_id,
            "voicepack_name": voicepack_name,
            "language": language,
            "status": "queued",
            "epoch": None,
            "loss": None,
            "eta_seconds": None,
            "started_at": None,
            "finished_at": None,
            "log_tail": None,
            "epochs": epochs,
            "lr": lr,
            "created_at": _now_iso(),
        }
    )
    cfg_path = _build_config(run_id, dataset_id, language, epochs, lr)
    fut = executor().submit(_run_subprocess, run_id, cfg_path)
    _futures[run_id] = fut
    return run_id


def cancel_run(run_id: str) -> bool:
    fut = _futures.get(run_id)
    if fut is None:
        return False
    cancelled = fut.cancel()
    if cancelled:
        _update_run(run_id, status="failed", finished_at=_now_iso(), log_tail="cancelled by user")
    return cancelled
