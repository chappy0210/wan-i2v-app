#!/usr/bin/env bash
# Resolve the RunPod Model Cache snapshot for Qwen/Qwen2.5-VL-7B-Instruct
# and start the caption handler. Mirrors the H3 worker's start.sh rule:
# prefer the commit named in refs/main, else the first snapshot dir.
# The resolved path is exported as CAPTION_MODEL_PATH; handler.py
# re-resolves it itself as a fallback, so this script never blocks the
# worker from starting — a missing cache surfaces as a per-job error.
set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CACHE_MODEL_DIR="/runpod-volume/huggingface-cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
SNAPSHOT=""

if [ -f "$CACHE_MODEL_DIR/refs/main" ]; then
  HASH="$(cat "$CACHE_MODEL_DIR/refs/main" | tr -d '[:space:]')"
  if [ -f "$CACHE_MODEL_DIR/snapshots/$HASH/config.json" ]; then
    SNAPSHOT="$CACHE_MODEL_DIR/snapshots/$HASH"
  fi
fi
if [ -z "$SNAPSHOT" ] && [ -d "$CACHE_MODEL_DIR/snapshots" ]; then
  for d in "$CACHE_MODEL_DIR"/snapshots/*/; do
    if [ -f "$d/config.json" ]; then
      SNAPSHOT="${d%/}"
      break
    fi
  done
fi

if [ -n "$SNAPSHOT" ]; then
  echo "[start] Using cached Qwen2.5-VL snapshot: $SNAPSHOT"
  export CAPTION_MODEL_PATH="$SNAPSHOT"
else
  echo "[start] WARNING: Qwen2.5-VL cache snapshot not found under $CACHE_MODEL_DIR"
  echo "[start] Cache root listing (for debugging):"
  ls -la /runpod-volume/huggingface-cache/hub/ 2>/dev/null || echo "[start] (no cache hub dir)"
  ls -la "$CACHE_MODEL_DIR" 2>/dev/null || echo "[start] (no model cache dir)"
  # Continue anyway: the handler falls back to the plain repo id, which
  # fails per-job while offline instead of crash-looping the worker.
fi

cd /app
exec python -u handler.py
