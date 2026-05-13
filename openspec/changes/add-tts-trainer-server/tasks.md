# Tasks — tts-trainer FastAPI service

## 1. Repo scaffolding
- [x] 1.1 Fork `yl4579/StyleTTS2` to `autoeditor-video-or-audio/tts-trainer`
- [x] 1.2 Clone to `/home/vagrant/mvp/editaudiotomovie/tts-trainer`
- [x] 1.3 `openspec init . --tools claude`
- [x] 1.4 Write proposal / design / this tasks file / capability spec

## 2. MinIO pre-flight
- [x] 2.1 Discover endpoint via kubectl: `192.168.15.152:30002`
- [x] 2.2 Create bucket `tts-training`
- [x] 2.3 Capture credentials (`MINIO_ACCESS_KEY=minio`, `MINIO_SECRET_KEY=MinioS3Success`
      — root pair; rotate later if needed)

## 3. Server addons (`server_addons/`)
- [ ] 3.1 `storage.py` — minio SDK wrapper (put / get / list / presign)
- [ ] 3.2 `datasets.py` — Common Voice / CETUC / CORAA / HF adapter,
      streaming + MFA alignment
- [ ] 3.3 `training.py` — wraps `train_finetune.py`, ProcessPool
      execution, SQLite run-state
- [ ] 3.4 `voicepack.py` — `(511, 1, 256)` speaker-embedding export
      with `pf_dora.pt` shape round-trip validation
- [ ] 3.5 `schemas.py` — Pydantic request/response models
- [ ] 3.6 `server_app.py` — FastAPI lifespan, all 9 routes, asyncio
      synth lock, graceful shutdown

## 4. Docker + ops parity
- [ ] 4.1 `Dockerfile.gpu` (Python 3.12, torch 2.8 + cu128, MFA,
      librosa, phonemizer, pyworld, minio, hf_transfer)
- [ ] 4.2 Flash-attn variant arg + prebuilt wheel pin
- [ ] 4.3 `docker-compose.gpu.prod.yml` — host port 8009, volume
      mounts for `data/{datasets,runs,voicepacks}`, optional
      `data/kokoro-voices` shared with kokoro-fastapi
- [ ] 4.4 `.env.example` documenting `MINIO_*`, dataset cache size
- [ ] 4.5 `helm/{Chart.yaml,values.yaml,templates/*}` cloned from
      omnivoice-tts and renamed
- [ ] 4.6 `k8s/{configmap,deployment,pvc,service}.yaml` ditto
- [ ] 4.7 `.github/workflows/ci.yml` matrix `gpu` + `gpu-flash`
      publishing to GHCR with helm auto-bump + cleanup keep-10
- [ ] 4.8 `.releaserc.json` mirroring qwen3-tts (PR/issue
      lookups disabled — lesson from earlier release flow)
- [ ] 4.9 `NOTICE-fork.md` enumerating fork additions per Apache /
      MIT §4(b)
- [ ] 4.10 `README-fork.md` quick-start

## 5. Verification
- [ ] 5.1 `docker compose --env-file .env -f docker-compose.gpu.prod.yml up -d`
      on the RTX 4080 host. `/health` 200 within 60 s; `/health`
      reports MinIO reachable.
- [ ] 5.2 `POST /v1/datasets/download {id:"commonvoice_ptbr"}`
      returns 202; `mc ls local/tts-training/datasets/commonvoice_ptbr/`
      shows the tarball.
- [ ] 5.3 `POST /v1/training/runs {dataset_id:"commonvoice_ptbr",
      name:"first-ptbr", voicepack_name:"pf_test"}` queues a run.
- [ ] 5.4 Status transitions `queued → running` within 30 s.
- [ ] 5.5 At least one epoch completes; `GET /v1/training/runs/{id}`
      shows loss + ETA.
- [ ] 5.6 `POST /v1/voicepacks/{run_id}/export` writes a `.pt` whose
      shape matches `(511, 1, 256)`.

## 6. Spec validate + archive (post-bake-off)
- [ ] 6.1 `openspec validate add-tts-trainer-server --strict`
- [ ] 6.2 `openspec archive add-tts-trainer-server --yes` once a
      voicepack lands and the sibling vibetalker change is also
      archived
