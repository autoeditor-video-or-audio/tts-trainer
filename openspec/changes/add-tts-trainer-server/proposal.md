# Add tts-trainer FastAPI service (Kokoro voicepack training)

## Why
nifty-star (`vibetalker`) currently exposes Kokoro PT-BR via three
upstream voicepacks (`pf_dora`, `pm_alex`, `pm_santa`). PT-BR
Kokoro fine-tunes do not exist publicly; we want to pioneer one.

`hexgrad/kokoro` (inference) does not ship training code. The
canonical training repo for Kokoro's parent architecture is
[`yl4579/StyleTTS2`](https://github.com/yl4579/StyleTTS2) (MIT). This
fork wraps StyleTTS2 with a FastAPI service that:

- Downloads Common Voice PT-BR (~50 h CC0/CC-BY-4.0) on demand and
  caches it in MinIO.
- Runs the upstream `train_first.py` / `train_second.py` / fine-tune
  scripts on the user's RTX 4080 + 64 GB workstation.
- Exports the trained speaker embedding as a Kokoro-compatible
  voicepack (`(511, 1, 256)` `.pt` tensor) that the existing
  `kokoro-fastapi` inference container picks up via a shared bind
  mount or MinIO download.

Sibling fork pattern: mirrors the layout of
`autoeditor-video-or-audio/omnivoice-tts` exactly (Dockerfile.gpu,
helm/, k8s/, .github/workflows/ci.yml, .releaserc.json, OpenSpec,
docker-compose.gpu.prod.yml). Same operator mental model.

## What Changes
- `server_addons/server_app.py` — FastAPI lifespan that lazily
  initialises the StyleTTS2 trainer harness; exposes the wire
  contract documented below.
- `server_addons/datasets.py` — Common Voice / CETUC / CORAA /
  arbitrary HF dataset adapters, streaming download with idempotent
  resume, push to MinIO at `s3://tts-training/datasets/<id>/`.
- `server_addons/training.py` — wraps `train_finetune.py` (PT-BR
  recipe) and emits per-epoch checkpoints to
  `s3://tts-training/runs/<run_id>/`. Run-level state machine
  (`queued → running → ready → failed`) backed by a small SQLite
  file persisted alongside the checkpoints.
- `server_addons/voicepack.py` — extracts the trained speaker
  embedding from the final checkpoint and serialises it as a
  `(511, 1, 256)` `.pt` voicepack. Round-trip validates against the
  known-good upstream `pf_dora.pt` shape before declaring success.
- `server_addons/storage.py` — thin wrapper over the `minio`
  Python SDK; honours `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` /
  `MINIO_SECRET_KEY` / `MINIO_BUCKET` / `MINIO_SECURE` env vars.
- `server_addons/schemas.py` — Pydantic models for every request /
  response body.
- HTTP contract:

  | Route                                  | Purpose                                                                |
  |----------------------------------------|------------------------------------------------------------------------|
  | `GET /health`                          | 200 once Python deps + CUDA + MinIO are reachable                      |
  | `POST /v1/datasets/download`           | Body `{id: "commonvoice_ptbr" \| "cetuc" \| "coraa" \| "hf:<repo>"}`     |
  | `GET /v1/datasets`                     | List downloaded datasets with hours / speakers                         |
  | `POST /v1/training/runs`               | Body `{dataset_id, name, language, voicepack_name, epochs?, lr?}` → `{run_id, status:"queued"}` |
  | `GET /v1/training/runs`                | List runs                                                              |
  | `GET /v1/training/runs/{run_id}`       | `{status, epoch, loss, eta_seconds, log_tail}`                         |
  | `DELETE /v1/training/runs/{run_id}`    | Cancel                                                                 |
  | `POST /v1/voicepacks/{run_id}/export`  | Body `{name}` → writes `voicepacks/<name>.pt` locally + S3 + (opt) bind mount |
  | `GET /v1/voicepacks`                   | Merge of MinIO `voicepacks/` + local cache                             |
  | `GET /v1/voicepacks/{name}/download`   | Presigned MinIO URL passthrough                                        |

- `Dockerfile.gpu` parametric on `INSTALL_FLASH_ATTN` +
  `CUDA_BASE_TAG`. Base `nvidia/cuda:12.8.1-base-ubuntu22.04`,
  Python 3.12, PyTorch 2.8 + cu128. Layers the upstream StyleTTS2
  `requirements.txt` plus `phonemizer`, `monotonic-align`,
  `librosa`, `pyworld`, `minio`, `fastapi`, `uvicorn`, `pydantic`,
  `hf_transfer`, MFA Python wrapper.
- `docker-compose.gpu.prod.yml` exposing host port **8009**, with
  volume mounts for `./data/{datasets,runs,voicepacks}` and an
  optional `./data/kokoro-voices` bind mount to share with the
  kokoro-fastapi container.
- `helm/{Chart.yaml,values.yaml,templates/*}` + `k8s/*` cloned
  from omnivoice-tts shape.
- `.github/workflows/ci.yml` matrix `publish_gpu` + `publish_gpu_flash`
  publishing to GHCR; helm auto-bump; cleanup keeping latest tags.
- `.releaserc.json`, `NOTICE-fork.md`, `README-fork.md`.

## Impact
- New capability: `tts-trainer-server` (this change is its only
  source of truth).
- Sibling vibetalker change `add-tts-trainer-panel` already
  documents the client-side surface that consumes this contract.
- No breaking changes — brand-new fork.

## MinIO configuration (locked)
- Endpoint (LAN): `http://192.168.15.152:30002`
  (k8s `minio/svce-minio-api` NodePort).
- Bucket: `tts-training` (already created).
- Credentials supplied via env at run time, **not** committed.

## Sibling work
- `vibetalker/openspec/changes/add-tts-trainer-panel/` — already
  authored, dormant until this service is reachable.
