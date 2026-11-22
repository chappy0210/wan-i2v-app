"""
RunPod Serverless handler for MiniMax H3 via ComfyUI, using RunPod's
Model Cache instead of a Network Volume. (v5: adds the FL2VA
first-frame image-to-video variant and an optional baked-in LoRA.)

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
This handler uploads the input image, patches the selected H3 workflow
(stock 20-step or Turbo 4-step variant), queues it, and returns the
video as base64.

Model variants
--------------
``model_variant`` selects the H3 task:

  "ref2va" (default)  MiniMaxH3ReferenceToVideo with the Ref2VA weights;
                      the image is a *reference* the model re-renders
                      into the prompted scene. This is the v4 behavior.
  "fl2va"             MiniMaxH3ImageToVideo with the FL2VA weights; the
                      image becomes the video's literal first frame,
                      which preserves the person/scene far more strongly.

Both variants share the Qwen3-VL text encoder and the two VAEs, and both
honor ``mode`` ("stock" = 20 steps, "turbo" = 4 steps + the variant's
own Turbo LoRA at strength 1.0).

Optional LoRA
-------------
The NSFW Unlocked V2 LoRA is not part of Comfy-Org/MiniMax-H3, so the
Model Cache cannot supply it; the Dockerfile bakes it into the image at
build time and start.sh registers its folder as an extra loras path.
Chain (applied to the UNET output, in this order):
  146  Unlocked LoRA  (MMH3-I2V-NSFWUnlockedV2-lora.safetensors)
  145  Turbo LoRA     (turbo workflows only, always applied there)
If ``unlocked_strength`` is absent/null the Unlocked node is bypassed:
removed from the submitted workflow with its consumers rewired to the
node's own model input, so ComfyUI never tries to load the file.

Input (job["input"])
--------------------
{
  "image": "<base64-encoded PNG/JPEG>",   # required, the reference image
  "prompt": "... <Picture 1> ...",        # required
  "width": 864,                           # optional, default 864
  "height": 480,                          # optional, default 480
  "duration": 5.0,                        # optional seconds, default 5 (4-15)
  "mode": "stock" | "turbo",              # optional, default "stock"
  "model_variant": "ref2va" | "fl2va",    # optional, default "ref2va"
  "steps": 20,                            # optional override
  "seed": 42,                             # optional, random if omitted
  "unlocked_strength": 0.5                # optional, default null (off);
                                          #   0 or null = do not apply
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
WORKFLOW_FL2VA_STOCK = os.environ.get(
    "WORKFLOW_FL2VA_STOCK", "/app/workflow/minimax_h3_i2v_api_stock.json"
)
WORKFLOW_FL2VA_TURBO = os.environ.get(
    "WORKFLOW_FL2VA_TURBO", "/app/workflow/minimax_h3_i2v_api_turbo.json"
)
COMFY_OUTPUT_DIR = os.environ.get("COMFY_OUTPUT_DIR", "/comfyui/output")

DEFAULT_WIDTH = 864
DEFAULT_HEIGHT = 480
DEFAULT_DURATION = 5.0
FPS = 24

MAX_BASE64_VIDEO_MB = int(os.environ.get("MAX_BASE64_VIDEO_MB", "15"))
POLL_INTERVAL_SEC = 5
MAX_WAIT_SEC = int(os.environ.get("MAX_WAIT_SEC", "1200"))  # 20 min hard cap

# The diffusion weights each variant expects to find in the cache
# snapshot. The FL2VA file is the one whose presence in RunPod's cached
# copy of Comfy-Org/MiniMax-H3 is not yet proven; check_models_available
# names it explicitly when it is missing.
EXPECTED_UNET = {
    "ref2va": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "fl2va": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
}

# Optional LoRA chain node id in all four workflows (see module docstring).
LORA_UNLOCKED_NODE = "146"


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


def check_models_available(model_variant: str) -> None:
    """Fail fast (and informatively) if the variant's H3 weights are not
    visible to ComfyUI, instead of hanging inside a generation."""
    expected_unet = EXPECTED_UNET[model_variant]
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
    if expected_unet not in options:
        raise RuntimeError(
            f"H3 diffusion model not visible to ComfyUI: expected "
            f"{expected_unet!r} for model_variant={model_variant!r}, "
            f"UNETLoader offers: {options}. If this is the FL2VA file, "
            f"the Model Cache snapshot of Comfy-Org/MiniMax-H3 on this "
            f"host does not contain the FL2VA weights."
        )


def upload_image(image_bytes: bytes, filename: str) -> str:
    files = {"image": (filename, image_bytes, "image/png")}
    data = {"overwrite": "true"}
    r = requests.post(
        f"{COMFY_BASE}/upload/image", files=files, data=data, timeout=60
    )
    r.raise_for_status()
    return r.json()["name"]


def load_workflow(model_variant: str, mode: str) -> dict:
    if model_variant == "fl2va":
        path = WORKFLOW_FL2VA_TURBO if mode == "turbo" else WORKFLOW_FL2VA_STOCK
    else:
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
    """Patch the converted H3 API workflow by node id.

    Node ids (from the official template conversions; identical layout
    in the Ref2VA and FL2VA workflow files):
      92 SaveVideo, 124 BasicScheduler (steps), 129 RandomNoise (seed),
      136 MiniMaxH3ReferenceToVideo (ref2va) or MiniMaxH3ImageToVideo
          (fl2va) -- both expose prompt/width/height/length,
      137 LoadImage, 145 Turbo LoRA (turbo workflows only),
      146 Unlocked LoRA.
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
    # Optional LoRA. If it is not applied, bypass its node so the baked
    # LoRA file is never loaded (an app payload without the field then
    # produces exactly the workflow the v4 worker submitted).
    if job_input.get("unlocked_strength") is not None:
        wf[LORA_UNLOCKED_NODE]["inputs"]["strength_model"] = job_input[
            "unlocked_strength"
        ]
    else:
        _bypass_lora_node(wf, LORA_UNLOCKED_NODE)
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

    model_variant = str(job_input.get("model_variant", "ref2va")).lower()
    if model_variant not in ("ref2va", "fl2va"):
        raise ValueError("input.model_variant must be 'ref2va' or 'fl2va'")

    cleaned = {
        "image": job_input["image"],
        "prompt": str(job_input["prompt"]),
        "width": int(job_input.get("width", DEFAULT_WIDTH)),
        "height": int(job_input.get("height", DEFAULT_HEIGHT)),
        "duration": float(job_input.get("duration", DEFAULT_DURATION)),
        "mode": mode,
        "model_variant": model_variant,
        "steps": int(job_input["steps"]) if job_input.get("steps") else None,
        "seed": int(job_input.get("seed", random.randint(0, 2**31 - 1))),
        "filename_prefix": f"video/h3_{uuid.uuid4().hex[:8]}",
        "unlocked_strength": _parse_lora_strength(
            job_input, "unlocked_strength", None
        ),
        "mute": bool(job_input.get("mute", False)),
    }
    if cleaned["width"] <= 0 or cleaned["height"] <= 0:
        raise ValueError("width/height must be positive")
    if not (4.0 <= cleaned["duration"] <= 15.0):
        raise ValueError("duration must be between 4 and 15 seconds")
    return cleaned


def strip_audio(video_bytes: bytes) -> bytes:
    """Return the video with its audio track removed (stream copy, no
    re-encode), via PyAV — ComfyUI already depends on it (av>=17 in its
    requirements), so no extra package or binary is needed. Raises on
    any failure: a job that asked for mute must never silently return
    a video that still has audio."""
    import av  # local import: only needed on the mute path

    tag = uuid.uuid4().hex[:8]
    src_path = f"/tmp/mute_in_{tag}.mp4"
    out_path = f"/tmp/mute_out_{tag}.mp4"
    try:
        with open(src_path, "wb") as fh:
            fh.write(video_bytes)
        src = av.open(src_path)
        video_streams = [st for st in src.streams if st.type == "video"]
        if not video_streams:
            src.close()
            raise RuntimeError("mute: source video has no video stream")
        dst = av.open(out_path, "w", format="mp4")
        in_v = video_streams[0]
        out_v = dst.add_stream_from_template(in_v)
        for packet in src.demux(in_v):
            if packet.dts is None:
                continue
            packet.stream = out_v
            dst.mux(packet)
        dst.close()
        src.close()
        with open(out_path, "rb") as fh:
            muted = fh.read()
        chk = av.open(out_path)
        types = [st.type for st in chk.streams]
        chk.close()
        if "audio" in types:
            raise RuntimeError("mute: audio stream still present after remux")
        return muted
    finally:
        for path in (src_path, out_path):
            try:
                os.remove(path)
            except OSError:
                pass


def handler(job: dict) -> dict:
    job_input = job.get("input", {}) if isinstance(job, dict) else {}
    try:
        cleaned = validate_input(job_input)

        rejection = check_prompt_allowed(cleaned["prompt"])
        if rejection:
            return {"error": rejection}

        if cleaned["mute"]:
            try:
                import av  # noqa: F401
            except ImportError:
                return {
                    "error": "mute requested but PyAV (av) is not available "
                    "in this worker image"
                }

        comfy_wait_sec = wait_for_comfy()
        check_models_available(cleaned["model_variant"])
        image_bytes = base64.b64decode(cleaned["image"])
        image_name = upload_image(image_bytes, "input.png")

        workflow = patch_workflow(
            load_workflow(cleaned["model_variant"], cleaned["mode"]),
            cleaned,
            image_name,
        )
        outputs, generation_sec = queue_and_wait(workflow)

        filename = find_video(outputs, cleaned["filename_prefix"])
        if not filename:
            return {"error": f"No video output found in ComfyUI outputs: {outputs}"}

        video_bytes = download_video(filename)
        if cleaned["mute"]:
            video_bytes = strip_audio(video_bytes)
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
