# tts-trainer (fork of yl4579/StyleTTS2)

FastAPI wrapper around the upstream StyleTTS2 trainer that produces
Kokoro-compatible voicepacks for the nifty-star sequencer.

The upstream model ships under MIT and provides full training code.
This fork adds:

- **MinIO-backed artifact storage** for datasets, checkpoints, and
  exported voicepacks.
- **HTTP control surface** the nifty-star client (`vibetalker`) talks
  to through `/api/tts-trainer/*`.
- **Kokoro voicepack export** that round-trip-validates the
  `(511, 1, 256)` speaker-embedding shape against the upstream
  `pf_dora.pt` baseline.
- **Operational parity** with the sibling forks (`omnivoice-tts`,
  `qwen3-tts`): one Dockerfile.gpu, helm chart, k8s manifests,
  CI matrix publishing `:latest-gpu` and `:latest-gpu-flash` to GHCR.

## Hardware

| Mode | Tested on | Notes |
|------|-----------|-------|
| GPU (recommended) | RTX 4080 16 GB + 64 GB RAM | Full StyleTTS2 fine-tune in bf16. |
| GPU (tight)       | RTX 4060 8 GB              | Untested for full training; use at own risk. |
| CPU               | not supported              |                                              |

## Prerequisites

- A reachable MinIO instance (any S3-compatible store works; MinIO is
  what we deploy against). The trainer never reads/writes the
  bucket directly — only through the SDK wrapper.
- A bucket named (by default) `tts-training`. Created via the
  Python SDK or `mc mb local/tts-training`.

The MinIO endpoint we currently target:

```
http://192.168.15.152:30002
```

(k8s NodePort `minio/svce-minio-api`, see `kubectl -n minio get svc`.)

## Quick start (GPU host)

```bash
cd /path/to/editaudiotomovie/tts-trainer

cp .env.example .env
# Edit .env to match your MinIO credentials.

docker compose --env-file .env \
    -f docker-compose.gpu.prod.yml pull
docker compose --env-file .env \
    -f docker-compose.gpu.prod.yml up -d

# Wait for /health to report ready=true storage_reachable=true cuda_available=true.
curl -s http://localhost:8009/health | jq
```

## Workflow (HTTP)

```bash
# 1. Download Common Voice PT-BR. Returns 202 immediately; the
#    actual download runs in the background and pushes to MinIO.
curl -X POST http://localhost:8009/v1/datasets/download \
    -H 'Content-Type: application/json' \
    -d '{"id":"commonvoice_ptbr"}'

# 2. List datasets once it lands.
curl -s http://localhost:8009/v1/datasets | jq

# 3. Start a training run.
curl -X POST http://localhost:8009/v1/training/runs \
    -H 'Content-Type: application/json' \
    -d '{
      "dataset_id": "commonvoice_ptbr",
      "name": "first PT-BR run",
      "voicepack_name": "pf_test"
    }' | jq

# 4. Poll status.
curl -s http://localhost:8009/v1/training/runs/<run_id> | jq

# 5. Export voicepack.
curl -X POST http://localhost:8009/v1/voicepacks/<run_id>/export \
    -H 'Content-Type: application/json' \
    -d '{"name":"pf_test"}' | jq

# 6. Download voicepack (presigned MinIO URL).
curl -L http://localhost:8009/v1/voicepacks/pf_test/download -o pf_test.pt
```

## Image flavours

CI publishes two tags per release:

| Tag                | Attention kernel    | CUDA base layer                     |
|--------------------|---------------------|--------------------------------------|
| `:latest-gpu`      | sdpa                | `nvidia/cuda:12.8.1-base-ubuntu22.04` |
| `:latest-gpu-flash`| flash_attention_2   | `nvidia/cuda:12.8.1-devel-ubuntu22.04` |

The flash-attn variant pulls a prebuilt wheel
(`flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312`); falls back to
source build if the wheel URL ever 404s.

## Voicepack handoff to kokoro-fastapi

The trainer writes exported voicepacks both to MinIO and (optionally)
to a bind-mounted folder shared with the kokoro-fastapi container.
If you mount the same host path under both containers as
`/app/voices`, the inference container picks up the new voicepack
without a restart:

```yaml
# in kokoro-fastapi compose:
volumes:
  - /srv/kokoro-voices:/app/voices

# in tts-trainer compose:
volumes:
  - /srv/kokoro-voices:/app/data/kokoro-voices
```

Then on the vibetalker client side, register the new voicepack id
manually in `src/constants/voices.ts` and rebuild — the dropdown
picks it up.
