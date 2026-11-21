"""
RunPod Serverless handler for MiniMax H3 Reference-to-Video via ComfyUI,
using RunPod's Model Cache instead of a Network Volume.

How it works
------------
The endpoint's "Model" field is set to the Hugging Face repo
``Comfy-Org/MiniMax-H3``. RunPod pre-downloads that repo onto worker hosts
(outside billed time) at:

    /runpod-volume/huggingface-cache/hub/models--Comfy-Org--MiniMax-H3/
        snapshots/<commit-hash>/
            diffusion_models/  text_encoders/  vae/  loras/  ...

start.sh resolves that snapshot directory and generates ComfyUI's
extra_model_paths.yaml so the workflow's loaders find the files by name.
ComfyUI is started separately (see start.sh) on http://127.0.0.1:8188 .
This handler uploads the reference image, patches the H3 R2V workflow
(stock 20-step or Turbo4 4-step variant), queues it, and returns the
video as base64.

Input (job["input"])
--------------------
{
  "image": "<base64-encoded PNG/JPEG>",   # required, the reference image
  "prompt": "... <Picture 1> ...",        # required
  "width": 864,                           # optional, default 864
  "height": 480,                          # optional, default 480
  "duration": 5.0,                        # optional seconds, default 5 (4-15)
  "mode": "stock" | "turbo",              # optional, default "stock"
  "steps": 20,                            # optional override
  "seed": 42                              # optional, random if omitted
}

Output
------
{"video_base64": "...", "filename": "...", "seed": 42, "format": "mp4"}
or {"error": "..."} on failure.
"""

import base64
import copy
import glob
import json
import os
import random
import time
import uuid

import requests

try:
    import runpod
except ImportError:  # Allows local py_compile without runpod installed.
    runpod = None

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = os.environ.get("COMFY_PORT", "8188")
COMFY_BASE = f"http://{COMFY_HOST}:{COMFY_PORT}"
WORKFLOW_STOCK = os.environ.get(
    "WORKFLOW_STOCK", "/app/workflow/minimax_h3_r2v_api_stock.json"
)
WORKFLOW_TURBO = os.environ.get(
    "WORKFLOW_TURBO", "/app/workflow/minimax_h3_r2v_api_turbo.json"
)
COMFY_OUTPUT_DIR = os.environ.get("COMFY_OUTPUT_DIR", "/comfyui/output")

DEFAULT_WIDTH = 864
DEFAULT_HEIGHT = 480
DEFAULT_DURATION = 5.0
FPS = 24

MAX_BASE64_VIDEO_MB = int(os.environ.get("MAX_BASE64_VIDEO_MB", "15"))
POLL_INTERVAL_SEC = 5
MAX_WAIT_SEC = int(os.environ.get("MAX_WAIT_SEC", "1200"))  # 20 min hard cap


# ---------------------------------------------------------------------------
# Safety guard (same policy as the Wan worker)
# ---------------------------------------------------------------------------
_BANNED_TERMS = (
    # minors / underage (EN)
    "minor", "underage", "child", "children", "kid", "teen", "teenager",
    "preteen", "loli", "lolita", "shota", "schoolgirl", "schoolboy",
    "young girl", "young boy", "little girl", "little boy",
    # minors / underage (JA)
    "未成年", "児童", "小学生", "中学生", "ロリ", "ショタ", "幼女", "少女を",
    # non-consensual / real-person sexual content (EN)
    "non-consensual", "nonconsensual", "without consent", "real person",
    "celebrity", "deepfake", "face swap", "faceswap",
    # non-consensual / real-person sexual content (JA)
    "実在", "有名人", "芸能人", "無断", "同意なし", "ディープフェイク",
)


def check_prompt_allowed(prompt: str) -> str | None:
    """Return an error message if the prompt is rejected, else None."""
    text = prompt.lower()
    for term in _BANNED_TERMS:
        if term.lower() in text:
            return (
                "Prompt rejected by safety guard: prompts requesting minors / "
                "underage-looking characters or non-consensual real-person "
                "sexual content are not allowed."
            )
    return None


# ---------------------------------------------------------------------------
# ComfyUI helpers
# ---------------------------------------------------------------------------

def wait_for_comfy(timeout_sec: int = 240) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            r = requests.get(f"{COMFY_BASE}/system_stats", timeout=5)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(3)
    raise RuntimeError("ComfyUI did not become ready in time")


def upload_image(image_bytes: bytes, filename: str) -> str:
    files = {"image": (filename, image_bytes, "image/png")}
    data = {"overwrite": "true"}
    r = requests.post(
        f"{COMFY_BASE}/upload/image", files=files, data=data, timeout=60
    )
    r.raise_for_status()
    return r.json()["name"]


def load_workflow(mode: str) -> dict:
    path = WORKFLOW_TURBO if mode == "turbo" else WORKFLOW_STOCK
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def frames_for_duration(duration_sec: float) -> int:
    """H3 frame counts follow the template's rule: at 24 fps, counts are
    aligned to 5 + 17k (124 frames = ~5s). Mirrors the official template's
    ComfyMathExpression: max(5, round(d*24)) + (5 - (max(5, round(d*24)) % 17)) % 17
    """
    base = max(5, round(duration_sec * FPS))
    return base + (5 - (base % 17)) % 17


def patch_workflow(workflow: dict, job_input: dict, image_name: str) -> dict:
    """Patch the converted H3 R2V API workflow by node id.

    Node ids (from the official template conversion):
      92 SaveVideo, 124 BasicScheduler (steps), 129 RandomNoise (seed),
      136 MiniMaxH3ReferenceToVideo (prompt/width/height/length),
      137 LoadImage.
    """
    wf = copy.deepcopy(workflow)
    wf["136"]["inputs"]["prompt"] = job_input["prompt"]
    wf["136"]["inputs"]["width"] = job_input["width"]
    wf["136"]["inputs"]["height"] = job_input["height"]
    wf["136"]["inputs"]["length"] = frames_for_duration(job_input["duration"])
    wf["129"]["inputs"]["noise_seed"] = job_input["seed"]
    wf["137"]["inputs"]["image"] = image_name
    if job_input.get("steps"):
        wf["124"]["inputs"]["steps"] = job_input["steps"]
    wf["92"]["inputs"]["filename_prefix"] = job_input["filename_prefix"]
    return wf


def queue_and_wait(workflow: dict) -> dict:
    client_id = str(uuid.uuid4())
    r = requests.post(
        f"{COMFY_BASE}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=60,
    )
    r.raise_for_status()
    prompt_id = r.json()["prompt_id"]

    deadline = time.time() + MAX_WAIT_SEC
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL_SEC)
        h = requests.get(f"{COMFY_BASE}/history/{prompt_id}", timeout=30)
        h.raise_for_status()
        history = h.json()
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI execution error: {status}")
            return history[prompt_id].get("outputs", {})
    raise RuntimeError(f"Timed out after {MAX_WAIT_SEC}s waiting for ComfyUI")


def find_video(outputs: dict, filename_prefix: str) -> str | None:
    """Find the produced video as a path relative to the output dir.

    The workflow's SaveVideo filename_prefix contains a folder
    ("video/h3_..."), so history items carry the file's folder in a
    separate "subfolder" field. The bare filename alone cannot locate
    the file (learned from a 404 on /view in the first endpoint test),
    so always carry the subfolder along.
    """
    for node_output in outputs.values():
        for key in ("gifs", "videos", "images"):
            for item in node_output.get(key, []) or []:
                filename = item.get("filename", "")
                if filename.lower().endswith((".mp4", ".webm")):
                    subfolder = item.get("subfolder", "")
                    return os.path.join(subfolder, filename) if subfolder else filename
    candidates = glob.glob(os.path.join(COMFY_OUTPUT_DIR, filename_prefix + "*.mp4"))
    if candidates:
        newest = max(candidates, key=os.path.getmtime)
        return os.path.relpath(newest, COMFY_OUTPUT_DIR)
    return None


def download_video(filename: str) -> bytes:
    # 1) Direct local path (filename may carry a subfolder, e.g. video/...).
    local_path = os.path.join(COMFY_OUTPUT_DIR, filename)
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            return f.read()
    # 2) Recursive search by basename anywhere under the output dir.
    basename = os.path.basename(filename)
    matches = glob.glob(
        os.path.join(COMFY_OUTPUT_DIR, "**", basename), recursive=True
    )
    if matches:
        with open(matches[0], "rb") as f:
            return f.read()
    # 3) ComfyUI /view, with and without the subfolder parameter.
    subfolder, _, name = filename.rpartition("/")
    attempts = []
    if subfolder:
        attempts.append({"filename": name, "subfolder": subfolder, "type": "output"})
    attempts.append({"filename": name, "type": "output"})
    last_error: Exception | None = None
    for params in attempts:
        try:
            r = requests.get(f"{COMFY_BASE}/view", params=params, timeout=120)
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001 - try the next form
            last_error = e
    # 4) Give up, but report what the output dir actually contains so the
    #    next failure is self-diagnosing from the job error alone.
    listing = []
    for root, _dirs, files in os.walk(COMFY_OUTPUT_DIR):
        for fn in files:
            listing.append(os.path.relpath(os.path.join(root, fn), COMFY_OUTPUT_DIR))
        if len(listing) > 50:
            break
    raise RuntimeError(
        f"video file not retrievable: wanted {filename!r}, "
        f"last /view error: {last_error}, output dir listing: {listing[:50]}"
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def validate_input(job_input: dict) -> dict:
    if not job_input.get("image"):
        raise ValueError("input.image (base64) is required")
    if not job_input.get("prompt"):
        raise ValueError("input.prompt is required")

    base64.b64decode(job_input["image"], validate=True)

    mode = str(job_input.get("mode", "stock")).lower()
    if mode not in ("stock", "turbo"):
        raise ValueError("input.mode must be 'stock' or 'turbo'")

    cleaned = {
        "image": job_input["image"],
        "prompt": str(job_input["prompt"]),
        "width": int(job_input.get("width", DEFAULT_WIDTH)),
        "height": int(job_input.get("height", DEFAULT_HEIGHT)),
        "duration": float(job_input.get("duration", DEFAULT_DURATION)),
        "mode": mode,
        "steps": int(job_input["steps"]) if job_input.get("steps") else None,
        "seed": int(job_input.get("seed", random.randint(0, 2**31 - 1))),
        "filename_prefix": f"video/h3_{uuid.uuid4().hex[:8]}",
    }
    if cleaned["width"] <= 0 or cleaned["height"] <= 0:
        raise ValueError("width/height must be positive")
    if not (4.0 <= cleaned["duration"] <= 15.0):
        raise ValueError("duration must be between 4 and 15 seconds")
    return cleaned


def handler(job: dict) -> dict:
    job_input = job.get("input", {}) if isinstance(job, dict) else {}
    try:
        cleaned = validate_input(job_input)

        rejection = check_prompt_allowed(cleaned["prompt"])
        if rejection:
            return {"error": rejection}

        wait_for_comfy()
        image_bytes = base64.b64decode(cleaned["image"])
        image_name = upload_image(image_bytes, "input.png")

        workflow = patch_workflow(load_workflow(cleaned["mode"]), cleaned, image_name)
        outputs = queue_and_wait(workflow)

        filename = find_video(outputs, cleaned["filename_prefix"])
        if not filename:
            return {"error": f"No video output found in ComfyUI outputs: {outputs}"}

        video_bytes = download_video(filename)
        size_mb = len(video_bytes) / (1024 * 1024)
        if size_mb > MAX_BASE64_VIDEO_MB:
            return {
                "error": (
                    f"Video is {size_mb:.1f} MB, too large to return as base64 "
                    f"(limit {MAX_BASE64_VIDEO_MB} MB)."
                ),
                "filename": filename,
            }

        return {
            "video_base64": base64.b64encode(video_bytes).decode("utf-8"),
            "filename": filename,
            "seed": cleaned["seed"],
            "format": "mp4",
        }
    except Exception as exc:  # Return a clean error instead of crashing the worker.
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    if runpod is None:
        raise SystemExit("runpod package is not installed; see requirements.txt")
    runpod.serverless.start({"handler": handler})
