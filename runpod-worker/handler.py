"""
RunPod Serverless handler for Wan 2.2 Image-to-Video (I2V) via ComfyUI.

Approach
--------
ComfyUI is started separately (see start.sh) and listens on
http://127.0.0.1:8188 . This handler:

  1. validates the job input (including a simple safety guard, see below),
  2. uploads the input image to ComfyUI,
  3. patches a Wan 2.2 I2V workflow template (API format) with the
     prompt / size / frame count / seed from the job,
  4. queues the workflow via POST /prompt and polls GET /history/<id>,
  5. reads the produced video and returns it as base64.

Input (job["input"])
--------------------
{
  "image": "<base64-encoded PNG/JPEG>",        # required
  "prompt": "an adult woman dancing, ...",     # required
  "negative_prompt": "blurry, low quality",    # optional
  "width": 832,                                # optional, default 832
  "height": 480,                               # optional, default 480
  "num_frames": 81,                            # optional, default 81 (~5s @16fps)
  "steps": 20,                                 # optional, default 20
  "cfg": 5.0,                                  # optional, default 5.0
  "seed": 42                                   # optional, random if omitted
}

Output
------
{"video_base64": "...", "filename": "...", "seed": 42, "format": "mp4"}
or {"error": "..."} on failure.

NOTE: base64 inflates the payload by ~33% and RunPod / HTTP responses have
practical size limits. For anything larger than MAX_BASE64_VIDEO_MB this
handler returns an explicit TODO error instead of a giant payload.
TODO(storage): upload the mp4 to S3 / Cloudflare R2 and return a signed URL.
"""

import base64
import copy
import json
import os
import random
import time
import uuid

import requests

try:
    import runpod
except ImportError:  # Allows `python -m py_compile` / local imports without runpod installed.
    runpod = None

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = os.environ.get("COMFY_PORT", "8188")
COMFY_BASE = f"http://{COMFY_HOST}:{COMFY_PORT}"
WORKFLOW_PATH = os.environ.get(
    "WORKFLOW_PATH", "/app/workflow/wan22_ti2v_5b_api.json"
)
COMFY_OUTPUT_DIR = os.environ.get("COMFY_OUTPUT_DIR", "/comfyui/output")

# Cheap test defaults: 480p-ish, ~5 seconds. Do NOT raise these until the
# per-generation cost has been measured (the starting balance is only $10).
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 480
DEFAULT_NUM_FRAMES = 81  # Wan works best with 4n+1 frame counts.
DEFAULT_STEPS = 20
DEFAULT_CFG = 5.0

MAX_BASE64_VIDEO_MB = int(os.environ.get("MAX_BASE64_VIDEO_MB", "15"))
POLL_INTERVAL_SEC = 5
MAX_WAIT_SEC = int(os.environ.get("MAX_WAIT_SEC", "900"))  # 15 min hard cap


# ---------------------------------------------------------------------------
# Safety guard
# ---------------------------------------------------------------------------
# Why this exists: this is a personal, adult (erotic) video tool for adults.
# Adult content between / about consenting adults is the intended use, but
# sexual content involving minors -- or real people who did not consent --
# must never be generated. This is a deliberately simple, conservative
# keyword guard: it is NOT a complete moderation system. It errs on the side
# of rejecting ambiguous prompts, and it should be strengthened (not removed)
# if this tool is ever shared with anyone else.
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


def check_prompt_allowed(prompt: str, negative_prompt: str = "") -> str | None:
    """Return an error message if the prompt is rejected, else None."""
    text = f"{prompt} {negative_prompt}".lower()
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

def wait_for_comfy(timeout_sec: int = 180) -> None:
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
    """Upload an image to ComfyUI and return the name ComfyUI stored it as."""
    files = {"image": (filename, image_bytes, "image/png")}
    data = {"overwrite": "true"}
    r = requests.post(
        f"{COMFY_BASE}/upload/image", files=files, data=data, timeout=60
    )
    r.raise_for_status()
    return r.json()["name"]


def load_workflow() -> dict:
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        workflow = json.load(f)
    workflow.pop("_comment", None)
    return workflow


def _set_if_literal(inputs: dict, key: str, value) -> None:
    """Set inputs[key] only when it is not a link to another node.

    In an API-format export, an input fed by another node (for example a
    PrimitiveInt titled "Int (Steps)" feeding a sampler) is a list like
    ["128", 0]. Overwriting it with a literal would silently disconnect
    that wiring. Literal (unlinked) inputs are safe to patch directly.
    """
    if key in inputs and isinstance(inputs[key], list):
        return
    inputs[key] = value


def patch_workflow(workflow: dict, job_input: dict, image_name: str) -> dict:
    """Patch a ComfyUI API-format workflow by node class_type.

    Verified against the official ComfyUI Wan 2.2 14B I2V template
    (Comfy-Org/workflow_templates, templates/video_wan2_2_14B_i2v.json,
    UI format). Its relevant nodes are:

      top level:  97 LoadImage, 108 SaveVideo, 129 subgraph
      subgraph:   93 CLIPTextEncode "CLIP Text Encode (Positive Prompt)"
                  89 CLIPTextEncode "CLIP Text Encode (Negative Prompt)"
                  98 WanImageToVideo (width/height/length widgets)
                  86 KSamplerAdvanced (high-noise stage, add_noise=enable,
                     start_at_step 0, end_at_step = split)
                  85 KSamplerAdvanced (low-noise stage, add_noise=disable,
                     start_at_step = split)
                  94 CreateVideo, 87 VAEDecode

    So the real template uses TWO KSamplerAdvanced nodes and no plain
    KSampler. Both samplers are patched (same seed/steps/cfg), which
    matches how the template itself shares Primitive step/cfg nodes
    between the two stages. Steps/cfg/seed that arrive as links from
    Primitive nodes are left alone here and patched at the Primitive
    node instead (see below), so the two-stage split stays intact:
    this handler never touches start_at_step / end_at_step.

    The shipped workflow file is still a placeholder with the minimal
    node set (LoadImage, CLIPTextEncode x2, WanImageToVideo, KSampler,
    SaveVideo). Every class_type this function *requires* exists in
    that placeholder; KSamplerAdvanced / Primitive handling is
    additional, for the real exported template.
    """
    wf = copy.deepcopy(workflow)
    text_nodes_seen = 0
    sampler_seen = False
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        inputs = node.setdefault("inputs", {})
        title = str(node.get("_meta", {}).get("title", "")).lower()

        if class_type == "LoadImage":
            inputs["image"] = image_name
        elif class_type == "CLIPTextEncode":
            # Do NOT rely on encounter order: in the template file the
            # negative node (89) comes before the positive node (93), so
            # order-based assignment would swap prompt and negative_prompt.
            # Use the node's title when present, fall back to order only
            # for untitled templates.
            if "negative" in title or "ネガ" in title:
                inputs["text"] = job_input.get("negative_prompt", "")
            elif "positive" in title or "ポジ" in title:
                inputs["text"] = job_input["prompt"]
            elif text_nodes_seen == 0:
                inputs["text"] = job_input["prompt"]
            else:
                inputs["text"] = job_input.get("negative_prompt", "")
            text_nodes_seen += 1
        elif class_type == "WanImageToVideo":
            _set_if_literal(inputs, "width", job_input["width"] // 2)
            _set_if_literal(inputs, "height", job_input["height"] // 2)
            _set_if_literal(inputs, "length", job_input["num_frames"])
        elif class_type == "KSampler":
            sampler_seen = True
            inputs["seed"] = job_input["seed"]
            inputs["steps"] = job_input["steps"]
            inputs["cfg"] = job_input["cfg"]
        elif class_type == "KSamplerAdvanced":
            sampler_seen = True
            _set_if_literal(inputs, "noise_seed", job_input["seed"])
            _set_if_literal(inputs, "steps", job_input["steps"])
            _set_if_literal(inputs, "cfg", job_input["cfg"])
        elif class_type in ("PrimitiveInt", "PrimitiveFloat"):
            # The official template feeds steps/cfg (and size/seed) into
            # the graph from Primitive nodes titled e.g. "Int (Steps)"
            # and "Float (CFG)". Patch those at the source by title.
            if "steps" in title:
                inputs["value"] = job_input["steps"]
            elif "cfg" in title:
                inputs["value"] = job_input["cfg"]
            elif "seed" in title:
                inputs["value"] = job_input["seed"]
            elif "width" in title:
                inputs["value"] = job_input["width"]
            elif "height" in title:
                inputs["value"] = job_input["height"]
        elif class_type == "CreateVideo":
            _set_if_literal(inputs, "fps", job_input.get("fps", 24))
        elif class_type in ("SaveVideo", "VHS_VideoCombine"):
            if "filename_prefix" in inputs:
                inputs["filename_prefix"] = job_input["filename_prefix"]

    if text_nodes_seen == 0:
        raise RuntimeError(
            "Workflow template has no CLIPTextEncode node; "
            "export the Wan 2.2 I2V workflow in API format and check the template."
        )
    if not sampler_seen:
        raise RuntimeError(
            "Workflow template has no KSampler / KSamplerAdvanced node; "
            "the official Wan 2.2 14B I2V template uses two "
            "KSamplerAdvanced nodes (ids 86 and 85 in the UI template)."
        )
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


def find_video(outputs: dict) -> str | None:
    """Return the filename of the first video/gif output, if any."""
    for node_output in outputs.values():
        for key in ("gifs", "videos", "images"):
            for item in node_output.get(key, []) or []:
                filename = item.get("filename", "")
                if filename.lower().endswith((".mp4", ".webm", ".gif")):
                    return filename
    return None


def download_video(filename: str) -> bytes:
    local_path = os.path.join(COMFY_OUTPUT_DIR, filename)
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            return f.read()
    # Fall back to ComfyUI's /view endpoint (works when output dir differs).
    r = requests.get(
        f"{COMFY_BASE}/view",
        params={"filename": filename, "type": "output"},
        timeout=120,
    )
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def validate_input(job_input: dict) -> dict:
    if not job_input.get("image"):
        raise ValueError("input.image (base64) is required")
    if not job_input.get("prompt"):
        raise ValueError("input.prompt is required")

    # Reject obviously broken base64 early instead of paying for a GPU run.
    base64.b64decode(job_input["image"], validate=True)

    cleaned = {
        "image": job_input["image"],
        "prompt": str(job_input["prompt"]),
        "negative_prompt": str(
            job_input.get("negative_prompt", "blurry, low quality, distorted")
        ),
        "width": int(job_input.get("width", DEFAULT_WIDTH)),
        "height": int(job_input.get("height", DEFAULT_HEIGHT)),
        "num_frames": int(job_input.get("num_frames", DEFAULT_NUM_FRAMES)),
        "steps": int(job_input.get("steps", DEFAULT_STEPS)),
        "cfg": float(job_input.get("cfg", DEFAULT_CFG)),
        "fps": float(job_input.get("fps", 24)),
        "seed": int(job_input.get("seed", random.randint(0, 2**31 - 1))),
        "filename_prefix": f"wan_i2v_{uuid.uuid4().hex[:8]}",
    }
    if cleaned["width"] <= 0 or cleaned["height"] <= 0:
        raise ValueError("width/height must be positive")
    if cleaned["num_frames"] < 5:
        raise ValueError("num_frames must be at least 5")
    return cleaned


def handler(job: dict) -> dict:
    job_input = job.get("input", {}) if isinstance(job, dict) else {}
    try:
        cleaned = validate_input(job_input)

        rejection = check_prompt_allowed(
            cleaned["prompt"], cleaned["negative_prompt"]
        )
        if rejection:
            return {"error": rejection}

        wait_for_comfy()
        image_bytes = base64.b64decode(cleaned["image"])
        image_name = upload_image(image_bytes, f"{cleaned['filename_prefix']}.png")

        workflow = patch_workflow(load_workflow(), cleaned, image_name)
        outputs = queue_and_wait(workflow)

        filename = find_video(outputs)
        if not filename:
            return {"error": f"No video output found in ComfyUI outputs: {outputs}"}

        video_bytes = download_video(filename)
        size_mb = len(video_bytes) / (1024 * 1024)
        if size_mb > MAX_BASE64_VIDEO_MB:
            # TODO(storage): upload to S3 / Cloudflare R2 here and return
            # {"video_url": "<signed url>", ...} instead of base64.
            return {
                "error": (
                    f"Video is {size_mb:.1f} MB, too large to return as base64 "
                    f"(limit {MAX_BASE64_VIDEO_MB} MB). "
                    "TODO: configure S3/R2 upload and return a signed URL."
                ),
                "filename": filename,
            }

        return {
            "video_base64": base64.b64encode(video_bytes).decode("utf-8"),
            "filename": filename,
            "seed": cleaned["seed"],
            "format": "mp4",
        }
    except Exception as exc:  # Return a clean error to the caller, don't crash the worker.
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    if runpod is None:
        raise SystemExit("runpod package is not installed; see requirements.txt")
    runpod.serverless.start({"handler": handler})
