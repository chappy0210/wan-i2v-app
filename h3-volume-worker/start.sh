#!/usr/bin/env bash
# Point ComfyUI at the model folders on the attached Network Volume
# (mounted at /runpod-volume on RunPod Serverless) via a generated
# extra_model_paths.yaml, then start ComfyUI in the background and the
# RunPod handler in the foreground.
set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

VOLUME_DIR="/runpod-volume"

if [ -d "$VOLUME_DIR/diffusion_models" ] && [ -d "$VOLUME_DIR/loras" ]; then
  echo "[start] Using Network Volume models at $VOLUME_DIR"
  cat > /comfyui/extra_model_paths.yaml <<EOF
h3volume:
    diffusion_models: $VOLUME_DIR/diffusion_models
    text_encoders: $VOLUME_DIR/text_encoders
    vae: $VOLUME_DIR/vae
    loras: $VOLUME_DIR/loras
EOF
else
  echo "[start] WARNING: H3 model folders not found under $VOLUME_DIR"
  echo "[start] Volume root listing (for debugging):"
  ls -la "$VOLUME_DIR" 2>/dev/null || echo "[start] (no volume dir)"
  ls -la "$VOLUME_DIR/loras" 2>/dev/null || echo "[start] (no loras dir)"
  # Continue anyway: jobs will fail with model-not-found, which surfaces
  # the problem per-job instead of crash-looping the worker.
fi

cd /comfyui
python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch &

cd /app
exec python -u handler.py
