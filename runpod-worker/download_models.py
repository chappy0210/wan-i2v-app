#!/usr/bin/env python3
"""Ensure the Wan 2.2 TI2V-5B weights exist on the Network Volume.

Safety net requested by the user (2026-11-20): if the volume is wiped
(e.g. terminated after a long $0 balance), the worker re-downloads the
weights automatically on startup instead of failing every job with
"model not found". When all files are already present this is a fast
no-op (existence + size check only).

Layout follows extra_model_paths.yaml (Serverless worker-comfyui map):
  <models>/unet/wan2.2_ti2v_5B_fp16.safetensors
  <models>/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors
  <models>/vae/wan2.2_vae.safetensors

Source: Hugging Face repo Comfy-Org/Wan_2.2_ComfyUI_Repackaged
(split_files/... paths), the same files verified in the 2026-11-20
Pod test.
"""
import os
import sys
import urllib.request

MODELS_DIR = os.environ.get("MODELS_DIR", "/runpod-volume/models")

HF_BASE = "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main"

# (volume-relative path, HF split_files path, expected size in bytes)
FILES = [
    (
        "unet/wan2.2_ti2v_5B_fp16.safetensors",
        "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        9_999_658_496,
    ),
    (
        "clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        6_735_906_816,
    ),
    (
        "vae/wan2.2_vae.safetensors",
        "split_files/vae/wan2.2_vae.safetensors",
        1_409_400_832,
    ),
]

# Sizes from the verified Pod test download; allow 2% tolerance because
# the recorded figures were rounded to GB in the test report.
TOLERANCE = 0.02


def size_ok(path: str, expected: int) -> bool:
    try:
        actual = os.path.getsize(path)
    except OSError:
        return False
    return abs(actual - expected) <= expected * TOLERANCE


def download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"[models] downloading {url} -> {dest}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "wan-i2v-worker"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"[models] {done * 100 // total}%", flush=True)
    os.replace(tmp, dest)


def main() -> int:
    if not os.path.isdir("/runpod-volume") and os.environ.get("FORCE_MODEL_DOWNLOAD") != "1":
        print("[models] /runpod-volume not mounted; skipping (set FORCE_MODEL_DOWNLOAD=1 to override)", flush=True)
        return 0
    missing = []
    for rel, hf_path, expected in FILES:
        dest = os.path.join(MODELS_DIR, rel)
        if size_ok(dest, expected):
            print(f"[models] OK {rel}", flush=True)
        else:
            missing.append((dest, hf_path, expected))
    for dest, hf_path, expected in missing:
        download(f"{HF_BASE}/{hf_path}", dest)
        if not size_ok(dest, expected):
            print(f"[models] ERROR: size mismatch after download: {dest}", flush=True)
            return 1
    print(f"[models] all {len(FILES)} weight files present", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
