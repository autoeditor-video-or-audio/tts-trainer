# Design — tts-trainer FastAPI service

## Context
Sibling forks (`omnivoice-tts`, `qwen3-tts`, `chatterbox-tts-server`)
already converged on a stable layout:

- Upstream model wrapped by a `server_addons/` FastAPI app exposing a
  versioned OpenAI-compatible-ish contract.
- One `Dockerfile.gpu` parameterised by `INSTALL_FLASH_ATTN` +
  `CUDA_BASE_TAG`, building two CI flavours (`gpu` sdpa + `gpu-flash`).
- helm/k8s manifests + GHCR matrix CI + helm auto-bump + cleanup
  keep-10.

This fork follows the same skeleton. The only structural delta is
that **training is asynchronous**: a `POST /v1/training/runs`
returns immediately with a `run_id`, and the actual work happens in
a background worker. We avoid Celery / RQ overhead by running the
worker inside the same container (`asyncio` + a single
`asyncio.Lock` + a `concurrent.futures.ProcessPoolExecutor(max_workers=1)`)
so only one training job exists at any time.

## Decisions
- **Process pool, not threads**: PyTorch + CUDA hold GPU state per
  process; running training in a thread alongside the FastAPI loop
  starves the asyncio scheduler. One subprocess per run, cancelled
  via signal.
- **Run state in SQLite**, not in-memory: a single
  `data/state.sqlite` file persists `(run_id, status, epoch, loss,
  eta_seconds, started_at, finished_at, dataset_id, voicepack_name)`.
  Crash-tolerant: on container restart we resume polling the on-disk
  checkpoint dir.
- **MinIO is the source of truth** for large artifacts (datasets,
  checkpoints, voicepacks). Local volumes are a working cache only;
  on container start we hydrate the cache from MinIO for any run
  the SQLite says is in flight, then continue.
- **Voicepack extraction is a separate route**: `POST
  /v1/voicepacks/{run_id}/export` runs the
  StyleTTS2 → Kokoro `(511, 1, 256)` speaker-embedding extraction
  step. We validate the shape against `pf_dora.pt` before declaring
  success, return 400 with diagnostics otherwise.
- **Dataset ingestion** uses HuggingFace Datasets where possible
  (`mozilla-foundation/common_voice_19_0`, `pt` config). We stream
  to local disk + push tarball to MinIO. CORAA and CETUC are added
  later via similar adapters.
- **MFA alignment** happens once per dataset, persisted to MinIO so
  subsequent runs reuse it.
- **No flash-attn requirement** for StyleTTS2 training; we still
  ship the `:gpu-flash` image so operators can opt in. Same
  prebuilt-wheel-first / source-build-fallback pattern as
  omnivoice-tts.

## Alternatives considered
- **Use Ray / Modal for training orchestration**: overkill for a
  single-GPU host; introduces a control plane we don't need.
- **In-memory run state**: lose progress on restart. SQLite costs
  nothing.
- **Trigger training via webhook from vibetalker**: the panel already
  hits `POST /v1/training/runs` directly. No need for a separate
  event channel.

## Risks
- **Voicepack format extraction**: the `(511, 1, 256)` shape is
  community-documented (`huggingface.co/datasets/ecyht2/kokoro-82M-voices`).
  If the extraction yields a different shape, `voicepack.py`
  rejects the run and surfaces a clear error in `GET
  /v1/voicepacks`. We then revise the extraction in a follow-up
  change.
- **Training duration ~12-24 h on RTX 4080 16 GB**: MFA alignment is
  often the bottleneck. Checkpoint-per-epoch means a cancelled or
  crashed run can resume from the last completed epoch.
- **Common Voice license**: CC0/CC-BY-4.0. If we ever publish a
  voicepack derived from CC-BY data, attribution is required.
  Document at export time.
- **MinIO unreachable**: server starts but `/health` returns 503
  until the bucket is reachable. Surfaces clearly in the vibetalker
  panel's "service unreachable" state.

## Migration
None. Brand-new fork.
