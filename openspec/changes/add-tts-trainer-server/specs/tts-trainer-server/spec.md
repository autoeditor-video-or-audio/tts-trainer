# tts-trainer-server — Spec delta (ADDED capability)

## ADDED Requirements

### Requirement: Service Wire Contract
The fork SHALL expose a FastAPI service with the routes documented
below, all under `/v1/*` except `/health`.

#### Scenario: Health probe
- **WHEN** `GET /health` is called
- **THEN** the service MUST return 200 once the trainer harness has
  initialised AND the MinIO endpoint configured by env is reachable
- **AND** MUST return 503 with body `"loading"` while the harness is
  still warming up
- **AND** MUST return 503 with body `"storage unreachable"` if the
  MinIO endpoint cannot be reached within 5 s

#### Scenario: Dataset download endpoint
- **WHEN** the client posts `POST /v1/datasets/download` with body
  `{id}` where `id` is one of `commonvoice_ptbr`, `cetuc`, `coraa`,
  or `hf:<owner>/<repo>`
- **THEN** the service MUST queue a background download job and
  return 202 immediately with `{dataset_id, status:"downloading"}`
- **AND** when the download completes, the dataset tarball MUST be
  uploaded to `s3://${MINIO_BUCKET}/datasets/<id>/data.tar.zst`

#### Scenario: Dataset listing
- **WHEN** the client `GET`s `/v1/datasets`
- **THEN** the response MUST list every dataset present in MinIO
  with `[{id, hours, speakers, downloaded_at, status}]`

#### Scenario: Training run lifecycle
- **WHEN** the client posts `POST /v1/training/runs` with body
  `{dataset_id, name, language, voicepack_name, epochs?, lr?}`
- **THEN** the service MUST validate `dataset_id` is present,
  persist the run to SQLite with status `queued`, and return
  `{run_id, status:"queued"}`
- **AND** the in-process worker MUST pick the run up within 30 s
  and transition it to `running`, starting StyleTTS2 fine-tuning
  against the requested dataset
- **AND** `GET /v1/training/runs/{run_id}` MUST surface the current
  `{status, epoch, loss, eta_seconds, log_tail}`
- **AND** on success the status MUST transition to `ready`; on
  failure to `failed` with `log_tail` carrying the last 4 KB of
  stderr

#### Scenario: Training run cancellation
- **WHEN** the client sends `DELETE /v1/training/runs/{run_id}` for
  a `queued` or `running` run
- **THEN** the worker MUST signal the subprocess and the run MUST
  transition to `failed` with `log_tail = "cancelled by user"`

#### Scenario: Voicepack export
- **WHEN** the client posts `POST /v1/voicepacks/{run_id}/export`
  with body `{name}` for a run whose status is `ready`
- **THEN** the service MUST run the speaker-embedding extraction
  step on the final checkpoint, persist the resulting `.pt` locally
  AND upload to `s3://${MINIO_BUCKET}/voicepacks/<name>.pt`
- **AND** validate the tensor shape equals `(511, 1, 256)` before
  declaring success
- **AND** return 200 with `{name, size_bytes, run_id, sha256}`

#### Scenario: Voicepack listing + download
- **WHEN** the client `GET`s `/v1/voicepacks`
- **THEN** the response MUST list every voicepack in
  `s3://${MINIO_BUCKET}/voicepacks/` as
  `[{name, size_bytes, created_at, run_id, sha256}]`
- **AND** `GET /v1/voicepacks/{name}/download` MUST return a 302
  to a 15-minute MinIO presigned URL

### Requirement: MinIO-Backed Storage
The service SHALL store all large training artifacts (datasets,
checkpoints, voicepacks) in a single MinIO bucket configured via
env, with local filesystem volumes acting as a working cache only.

#### Scenario: Configuration via env
- **WHEN** the container starts
- **THEN** it MUST read `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`,
  `MINIO_SECRET_KEY`, `MINIO_BUCKET` (default `tts-training`), and
  `MINIO_SECURE` (default `false`) from environment variables
- **AND** MUST refuse to start if any of the first four is unset

#### Scenario: Bucket layout
- **WHEN** the service writes to MinIO
- **THEN** it MUST use the prefixes `datasets/<id>/`,
  `runs/<run_id>/`, and `voicepacks/<name>.pt`

#### Scenario: Cache hydration on cold start
- **WHEN** the container restarts and SQLite reports a run in
  `running` state
- **THEN** the service MUST hydrate the local cache for that run
  from MinIO before resuming the worker

### Requirement: GPU-Only Deployment
The fork SHALL deploy as GPU-only. CPU training is not provided.

#### Scenario: CUDA required at startup
- **WHEN** the container starts and `torch.cuda.is_available()`
  returns `False`
- **THEN** the service MUST refuse to start and exit with a clear
  error message

#### Scenario: docker-compose targets nvidia
- **WHEN** the operator runs
  `docker compose --env-file .env -f docker-compose.gpu.prod.yml up -d`
- **THEN** the compose definition MUST request a single nvidia GPU
  via `deploy.resources.reservations.devices` and export
  `NVIDIA_VISIBLE_DEVICES=all` plus
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility`

### Requirement: Two-Flavour GHCR Image Matrix
The fork SHALL publish two GPU image flavours per release tag,
mirroring the qwen3-tts / omnivoice-tts shape.

#### Scenario: Lean default + flash-attn variant
- **WHEN** the CI release job promotes a new semantic-release tag `vX.Y.Z`
- **THEN** GHCR MUST receive
  `ghcr.io/<owner>/tts-trainer:vX.Y.Z-gpu` and `:latest-gpu`
  (sdpa, lean `*-base-*` CUDA layer) AND
  `:vX.Y.Z-gpu-flash` and `:latest-gpu-flash` (flash-attn 2.8.3
  prebuilt wheel, `*-devel-*` CUDA layer)

#### Scenario: Cleanup preserves both latest tags
- **WHEN** the `cleanup_ghcr` job runs after a release
- **THEN** it MUST preserve container versions tagged `latest-gpu`
  and `latest-gpu-flash` while keeping at least the 10 most recent
  versioned tags

### Requirement: Voicepack Shape Round-Trip Validation
The voicepack export step SHALL refuse to publish a voicepack whose
tensor shape does not match Kokoro's expected `(511, 1, 256)`
speaker-embedding format.

#### Scenario: Shape mismatch
- **WHEN** the extracted speaker embedding has any shape other than
  `(511, 1, 256)`
- **THEN** `POST /v1/voicepacks/{run_id}/export` MUST return 400
  with body `{error: {message, actual_shape}}`
- **AND** the run's status MUST remain `ready` (the export step is
  retryable)
