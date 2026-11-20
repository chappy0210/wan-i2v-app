#!/usr/bin/env bash
# Start ComfyUI in the background, then run the RunPod handler in the
# foreground. The handler itself waits for ComfyUI to become ready
# (see wait_for_comfy in handler.py), so no fixed sleep is needed here.
set -e

# Safety net: if the Network Volume lost the model weights (e.g. it was
# terminated after a long $0 balance), re-download them before ComfyUI
# starts. No-op when all files are present. Never block startup forever:
# on failure, log it and continue so the error surfaces per-job.
python /app/download_models.py || echo "[start] WARNING: model check/download failed; continuing"

cd /comfyui
python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch &

cd /app
exec python -u handler.py
