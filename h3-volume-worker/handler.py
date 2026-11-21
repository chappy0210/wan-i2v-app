"""
RunPod Serverless handler for MiniMax H3 Reference-to-Video via ComfyUI,
reading model weights from a Network Volume instead of the Model Cache.

How it works
------------
The endpoint has the Network Volume ``minimax-h3-models`` attached; on
RunPod Serverless the volume is mounted at ``/runpod-volume`` and its
root already uses ComfyUI's folder layout:

    /runpod-volume/
        diffusion_models/  text_encoders/  vae/  loras/  ...

start.sh generates ComfyUI's extra_model_paths.yaml pointing at those
folders, so the workflow's loaders find the files by name. ComfyUI is
started separately (see start.sh) on http://127.0.0.1:8188 . This
handler uploads the reference image, patches the H3 R2V workflow (stock
20-step or Turbo4 4-step variant, optionally with extra LoRAs from the
volume's loras/ folder), queues it, and returns the video as base64.

LoRA chain (applied to the UNET output, in this order)
------------------------------------------------------
  146  Unlocked LoRA      (MMH3-I2V-NSFWUnlockedV2-lora.safetensors)
  147  Nudification LoRA  (MMH3-R2V-Nudification-lora.safetensors)
  145  Turbo LoRA         (turbo workflow only, always applied there)
A LoRA whose strength is 0/null is bypassed: its node is removed from
the submitted workflow and its consumers are rewired to the node's own
model input, so ComfyUI never tries to load a file we did not ask for.

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
  "seed": 42,                             # optional, random if omitted
  "unlocked_strength": 0.5,               # optional, default 0.5;
                                          #   0 or null = do not apply
  "nudification_strength": null           # optional, default null (off);
                                          #   a number applies the LoRA
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

# Optional LoRA chain node ids in both workflows (see module docstring).
LORA_UNLOCKED_NODE = "146"
LORA_NUDIFICATION_NODE = "147"
DEFAULT_UNLOCKED_STRENGTH = 0.5


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

def wait_for_comfy(timeout_sec: int = 240) -> float:
    """Wait for ComfyUI to answer; return how many seconds it took."""
    started = time.time()
    deadline = started + timeout_sec
    last_problem = "no response yet"
    while time.time() < deadline:
        try:
            r = requests.get(f"{COMFY_BASE}/system_stats", timeout=5)
            if r.status_code == 200:
                return time.time() - started
            last_problem = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last_problem = f"{type(e).__name__}: {e}"
        time.sleep(3)
    raise RuntimeError(
        f"ComfyUI did not become ready in {timeout_sec}s (last problem: {last_problem})"
    )


EXPECTED_UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"


def check_models_available() -> None:
    """Fail fast (and informatively) if the volume's H3 weights are not
    visible to ComfyUI, instead of hanging inside a generation."""
    try:
        r = requests.get(f"{COMFY_BASE}/object_info/UNETLoader", timeout=30)
        r.raise_for_status()
        info = r.json()
        options = (
            info.get("UNETLoader", {})
            .get("input", {})
            .get("required", {})
            .get("unet_name", [[]])[0]
        )
    except Exception as e:  # noqa: BLE001 - report and let generation try anyway
        print(f"model check skipped: {e}", flush=True)
        return
    if EXPECTED_UNET not in options:
        raise RuntimeError(
            f"H3 diffusion model not visible to ComfyUI: expected "
            f"{EXPECTED_UNET!r}, UNETLoader offers: {options}"
        )


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


def _bypass_lora_node(wf: dict, node_id: str) -> None:
    """Remove a LoraLoaderModelOnly node from the workflow, rewiring
    every consumer of its output to the node's own model input so the
    model chain stays connected end to end."""
    source = wf[node_id]["inputs"]["model"]
    for node in wf.values():
        inputs = node.get("inputs", {})
        for key, value in inputs.items():
            if isinstance(value, list) and len(value) == 2 and value[0] == node_id:
                inputs[key] = list(source)
    del wf[node_id]


def patch_workflow(workflow: dict, job_input: dict, image_name: str) -> dict:
    """Patch the converted H3 R2V API workflow by node id.

    Node ids (from the official template conversion):
      92 SaveVideo, 124 BasicScheduler (steps), 129 RandomNoise (seed),
      136 MiniMaxH3ReferenceToVideo (prompt/width/height/length),
      137 LoadImage, 145 Turbo LoRA (turbo workflow only),
      146 Unlocked LoRA, 147 Nudification LoRA.
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
    # Optional LoRAs, in chain order (Unlocked before Nudification). A
    # LoRA that is not applied is bypassed so its file is never loaded.
    if job_input.get("unlocked_strength") is not None:
        wf[LORA_UNLOCKED_NODE]["inputs"]["strength_model"] = job_input[
            "unlocked_strength"
        ]
    else:
        _bypass_lora_node(wf, LORA_UNLOCKED_NODE)
    if job_input.get("nudification_strength") is not None:
        wf[LORA_NUDIFICATION_NODE]["inputs"]["strength_model"] = job_input[
            "nudification_strength"
        ]
    else:
        _bypass_lora_node(wf, LORA_NUDIFICATION_NODE)
    return wf


def _queue_snapshot() -> str:
    """One-line summary of ComfyUI's queue + device state, for errors."""
    try:
        q = requests.get(f"{COMFY_BASE}/queue", timeout=10).json()
        running = [item[1] for item in q.get("queue_running", [])]
        pending = [item[1] for item in q.get("queue_pending", [])]
        state = f"queue running={running} pending={pending}"
    except Exception as e:  # noqa: BLE001
        state = f"queue unreadable ({e})"
    try:
        stats = requests.get(f"{COMFY_BASE}/system_stats", timeout=10).json()
        devices = stats.get("devices", [])
        if devices:
            d = devices[0]
            state += (
                f"; device={d.get('name')} vram_free={d.get('vram_free')} "
                f"vram_total={d.get('vram_total')}"
            )
    except Exception:  # noqa: BLE001
        pass
    return state


def queue_and_wait(workflow: dict) -> tuple[dict, float]:
    """Queue the workflow; return (outputs, generation_seconds).

    Diagnostics: a rejected prompt raises with ComfyUI's response body
    (node_errors), and a timeout raises with the live queue/device state,
    so a failed job explains itself without worker-log access.
    """
    client_id = str(uuid.uuid4())
    r = requests.post(
        f"{COMFY_BASE}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"ComfyUI rejected the prompt (HTTP {r.status_code}): {r.text[:1500]}"
        )
    payload = r.json()
    if payload.get("error") or payload.get("node_errors"):
        raise RuntimeError(
            f"ComfyUI rejected the prompt: error={payload.get('error')} "
            f"node_errors={json.dumps(payload.get('node_errors'))[:1500]}"
        )
    prompt_id = payload["prompt_id"]

    started = time.time()
    deadline = started + MAX_WAIT_SEC
    last_state = "not polled yet"
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL_SEC)
        h = requests.get(f"{COMFY_BASE}/history/{prompt_id}", timeout=30)
        h.raise_for_status()
        history = h.json()
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI execution error: {status}")
            return history[prompt_id].get("outputs", {}), time.time() - started
        last_state = _queue_snapshot()
    raise RuntimeError(
        f"Timed out after {MAX_WAIT_SEC}s waiting for generation "
        f"(prompt_id={prompt_id}, last state: {last_state})"
    )


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

def _parse_lora_strength(job_input: dict, key: str, default) -> float | None:
    """Parse an optional LoRA strength: a number applies the LoRA,
    0 or null (or the default, when the default is None) disables it."""
    raw = job_input.get(key, default)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"input.{key} must be a number")
    if value == 0:
        return None
    if not -2.0 <= value <= 2.0:
        raise ValueError(f"input.{key} must be between -2 and 2 (0 disables the LoRA)")
    return value


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
        "unlocked_strength": _parse_lora_strength(
            job_input, "unlocked_strength", DEFAULT_UNLOCKED_STRENGTH
        ),
        "nudification_strength": _parse_lora_strength(
            job_input, "nudification_strength", None
        ),
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

        comfy_wait_sec = wait_for_comfy()
        check_models_available()
        image_bytes = base64.b64decode(cleaned["image"])
        image_name = upload_image(image_bytes, "input.png")

        workflow = patch_workflow(load_workflow(cleaned["mode"]), cleaned, image_name)
        outputs, generation_sec = queue_and_wait(workflow)

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
            "timing_sec": {
                "comfy_ready_wait": round(comfy_wait_sec, 1),
                "generation": round(generation_sec, 1),
            },
        }
    except Exception as exc:  # Return a clean error instead of crashing the worker.
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    if runpod is None:
        raise SystemExit("runpod package is not installed; see requirements.txt")
    runpod.serverless.start({"handler": handler})
