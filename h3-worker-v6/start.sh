#!/usr/bin/env bash
# Resolve the RunPod Model Cache snapshot for Comfy-Org/MiniMax-H3 and
# generate ComfyUI's extra_model_paths.yaml pointing at it, then start
# ComfyUI in the background and the RunPod handler in the foreground.
# The yaml also registers /app/baked_loras (LoRA files baked into the
# image at build time, see the Dockerfile) as an extra loras folder.
set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CACHE_MODEL_DIR="/runpod-volume/huggingface-cache/hub/models--Comfy-Org--MiniMax-H3"
SNAPSHOT=""

if [ -f "$CACHE_MODEL_DIR/refs/main" ]; then
  HASH="$(cat "$CACHE_MODEL_DIR/refs/main" | tr -d '[:space:]')"
  if [ -d "$CACHE_MODEL_DIR/snapshots/$HASH" ]; then
    SNAPSHOT="$CACHE_MODEL_DIR/snapshots/$HASH"
  fi
fi
if [ -z "$SNAPSHOT" ] && [ -d "$CACHE_MODEL_DIR/snapshots" ]; then
  SNAPSHOT="$(ls -d "$CACHE_MODEL_DIR"/snapshots/*/ 2>/dev/null | head -n 1)"
  SNAPSHOT="${SNAPSHOT%/}"
fi

if [ -n "$SNAPSHOT" ] && [ -d "$SNAPSHOT/diffusion_models" ]; then
  echo "[start] Using cached H3 snapshot: $SNAPSHOT"
  cat > /comfyui/extra_model_paths.yaml <<EOF
h3cache:
    diffusion_models: $SNAPSHOT/diffusion_models
    text_encoders: $SNAPSHOT/text_encoders
    vae: $SNAPSHOT/vae
    loras: $SNAPSHOT/loras
baked:
    loras: /app/baked_loras
EOF
  # Whether the cached snapshot also carries the FL2VA weights is the
  # open question the v5 image exists to answer; name the files loudly
  # here (and the handler names them again in the job error) if absent.
  if [ -f "$SNAPSHOT/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" ]; then
    echo "[start] FL2VA weights present in cache snapshot."
  else
    echo "[start] WARNING: minimax_h3_fl2va_pruned_int8_convrot.safetensors NOT found in the cache snapshot;"
    echo "[start] model_variant=fl2va jobs will fail with an error naming that file."
  fi
  if [ ! -f "$SNAPSHOT/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors" ]; then
    echo "[start] WARNING: minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors NOT found in the cache snapshot's loras/;"
    echo "[start] fl2va turbo jobs will fail with a LoRA-not-found error."
  fi
else
  echo "[start] WARNING: MiniMax-H3 cache snapshot not found under $CACHE_MODEL_DIR"
  echo "[start] Cache root listing (for debugging):"
  ls -la /runpod-volume/huggingface-cache/hub/ 2>/dev/null || echo "[start] (no cache hub dir)"
  ls -la "$CACHE_MODEL_DIR" 2>/dev/null || echo "[start] (no model cache dir)"
  # Continue anyway: jobs will fail with model-not-found, which surfaces
  # the problem per-job instead of crash-looping the worker.
fi

if [ -f /app/baked_loras/MMH3-I2V-NSFWUnlockedV2-lora.safetensors ]; then
  echo "[start] Baked-in Unlocked LoRA present."
else
  echo "[start] WARNING: baked-in Unlocked LoRA missing at /app/baked_loras/MMH3-I2V-NSFWUnlockedV2-lora.safetensors;"
  echo "[start] jobs with unlocked_strength set will fail; jobs without it are unaffected."
fi

cd /comfyui
python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch &

cd /app
exec python -u handler.py
