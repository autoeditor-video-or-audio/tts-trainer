# Fork Modifications Notice

This fork (`autoeditor-video-or-audio/tts-trainer`) is derived from
[`yl4579/StyleTTS2`](https://github.com/yl4579/StyleTTS2), licensed
under the MIT License. Upstream `LICENSE` and `README.md` are
preserved unchanged.

Per MIT terms (notice retention), the following files were **added**
by this fork (no upstream source files were modified):

```
server_addons/__init__.py
server_addons/server_app.py
server_addons/storage.py
server_addons/datasets.py
server_addons/training.py
server_addons/voicepack.py
server_addons/schemas.py
server_addons/tests/__init__.py
Dockerfile.gpu
docker-compose.gpu.prod.yml
.env.example
helm/Chart.yaml
helm/values.yaml
helm/templates/{configmap,deployment,pvc,service}.yaml
k8s/{configmap,deployment,pvc,service}.yaml
.github/workflows/ci.yml
.releaserc.json
NOTICE-fork.md
README-fork.md
openspec/AGENTS.md
openspec/project.md
openspec/changes/add-tts-trainer-server/proposal.md
openspec/changes/add-tts-trainer-server/design.md
openspec/changes/add-tts-trainer-server/tasks.md
openspec/changes/add-tts-trainer-server/specs/tts-trainer-server/spec.md
```

Purpose: wrap the upstream StyleTTS2 training code (`train_finetune.py`,
`train_first.py`, `train_second.py`) in a FastAPI service that

- downloads Common Voice PT-BR / CETUC / CORAA datasets to MinIO,
- launches training runs as supervised subprocesses,
- exports the trained speaker embedding as a Kokoro-compatible
  `(511, 1, 256)` `.pt` voicepack consumable by the existing
  `kokoro-fastapi` inference server,
- exposes the operation surface to the nifty-star (`vibetalker`)
  client through `/api/tts-trainer/*`.

This fork does not redistribute training data; operators supply their
own datasets through the documented dataset adapters (Common Voice
PT-BR is the only adapter implemented in this initial change).
