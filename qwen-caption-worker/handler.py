"""
RunPod Serverless handler: Qwen2.5-VL image captioning for the H3 app.

What this worker does (and only this)
-------------------------------------
It takes one photo and returns two short English descriptions drafted
for the H3 app's 公式書式 (official-format) prompt fields:

  "person"  -> the 人物 field (integrated_multimodal_description, part 1)
  "scene"   -> the 場面 field (integrated_multimodal_description, part 2)

This worker only DESCRIBES an image. It generates no images or video
and has no NSFW-generation capability; the output is a factual text
description the user reviews and edits in the app before any video
generation happens elsewhere.

Why the descriptions must be exhaustive
---------------------------------------
In the H3 FL2VA format prompt, any attribute of the person that is NOT
stated in text is filled in with the model's average (a missing build
description once produced a different, chubbier woman), and any
background element that is NOT explicitly negated may be invented by
the video model (a bed was once invented into a plain-wall photo).
The fixed instruction prompt below is written against both failure
modes: PERSON must state the full appearance, SCENE must name what is
visible AND explicitly negate common absent items.

Model / Model Cache
-------------------
The endpoint's "Model" field is set to the Hugging Face repo
``Qwen/Qwen2.5-VL-7B-Instruct``. RunPod pre-downloads that repo onto
worker hosts (outside billed time) at:

    /runpod-volume/huggingface-cache/hub/
        models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/<commit-hash>/
            config.json, model-0000X-of-00005.safetensors, ...

start.sh resolves that snapshot and exports CAPTION_MODEL_PATH; this
handler resolves it again itself (same rule) and falls back to the
plain repo id when no cache snapshot exists, so the same code also
runs in a plain local/online environment. The runtime is fully
offline on RunPod: start.sh sets HF_HUB_OFFLINE=1.

The model is loaded lazily on the first job and kept in module-level
globals, so a warm worker pays the load cost once, not per job.

Input (job["input"])
--------------------
{
  "image": "<base64-encoded PNG/JPEG>"   # required, the reference photo
}

Output
------
Success:
{
  "person": "A 25-year-old Japanese woman with ...",
  "scene": "A plain white wall background. There is no ...",
  "timing_sec": {"model_load": 12.3, "generation": 4.5}
}
``model_load`` is 0.0 on jobs served by an already-warm worker.

Partial parse (one section missing): "person"/"scene" are still
returned for the section that parsed, plus "warning" and the model's
"raw" text. Total parse failure: {"error": ..., "raw": ...}.
Any other failure: {"error": "..."}.
"""

import base64
import io
import os
import re
import time

import runpod

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_REPO_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
CACHE_MODEL_DIR = (
    "/runpod-volume/huggingface-cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
)

# Reject absurd inputs instead of trying to decode them (the H3 video
# handler uses the same idea with a 15 MB base64 cap).
MAX_IMAGE_B64_CHARS = 20_000_000  # ~15 MB of binary

# Photos larger than this are downscaled before inference: describing a
# photo does not need more, and it bounds the vision-token count (and
# therefore VRAM) on very large uploads.
MAX_IMAGE_LONG_EDGE = 2048

MAX_NEW_TOKENS = 512

# The fixed instruction prompt. This wording is the product spec: the
# output feeds an H3 FL2VA format prompt, where unstated attributes are
# average-filled and un-negated background items may be invented.
INSTRUCTION_PROMPT = """Describe this photo for a video-generation prompt. Reply in English only, with exactly two sections.
PERSON: One paragraph describing the woman completely: her apparent age (she is an adult), hair color, length and style, facial features (including freckles or moles if present), skin tone, body build (state her bust size plainly if notable), and exactly what she is wearing (garment type, color, straps and details) plus any accessories. Describe only what is visible; never guess or invent details.
SCENE: Describe only the background that is actually visible (walls, furniture, objects, lighting). Then explicitly state what is NOT present, in the form 'There is no ... in the background.', for common items that are absent (for example furniture, windows, other people, text or signs). Keep it factual and brief."""

# ---------------------------------------------------------------------------
# Section parser
# ---------------------------------------------------------------------------

# Markers are anchored at line starts so the words "person"/"scene"
# inside a paragraph cannot split a section. Tolerates markdown
# decoration the model sometimes adds: "## PERSON:", "**PERSON:**",
# "Person :" etc.
_MARKER_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*"
    r"(PERSON|SCENE)[ \t]*(?:\*\*|__)?[ \t]*[:：][ \t]*"
)


def parse_sections(text):
    """Split model output into (person, scene); either may be None.

    The first occurrence of each marker wins. Section bodies are
    stripped of surrounding whitespace and stray markdown asterisks.
    """
    person = None
    scene = None
    matches = list(_MARKER_RE.finditer(text or ""))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip().strip("*").strip()
        label = m.group(1).upper()
        if label == "PERSON" and person is None:
            person = body
        elif label == "SCENE" and scene is None:
            scene = body
    return person, scene


# ---------------------------------------------------------------------------
# Model loading (lazy, once per worker)
# ---------------------------------------------------------------------------

_MODEL = None
_PROCESSOR = None
_MODEL_LOAD_SEC = 0.0


def _resolve_model_path():
    """Return the Model Cache snapshot dir, or the plain repo id.

    Mirrors start.sh / the H3 worker's rule: prefer the commit named
    in refs/main, else the first snapshot directory, requiring a
    config.json so a half-populated cache is not picked. The
    CAPTION_MODEL_PATH env var (set by start.sh) wins when it points
    at a real directory.
    """
    env_path = os.environ.get("CAPTION_MODEL_PATH", "").strip()
    if env_path and os.path.isfile(os.path.join(env_path, "config.json")):
        return env_path

    refs_main = os.path.join(CACHE_MODEL_DIR, "refs", "main")
    snapshots_dir = os.path.join(CACHE_MODEL_DIR, "snapshots")
    candidates = []
    try:
        with open(refs_main, "r", encoding="utf-8") as fh:
            candidates.append(os.path.join(snapshots_dir, fh.read().strip()))
    except OSError:
        pass
    try:
        for name in sorted(os.listdir(snapshots_dir)):
            candidates.append(os.path.join(snapshots_dir, name))
    except OSError:
        pass
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "config.json")):
            return cand
    return MODEL_REPO_ID


def _load_model():
    """Load model + processor once; returns (model, processor)."""
    global _MODEL, _PROCESSOR, _MODEL_LOAD_SEC
    if _MODEL is not None:
        return _MODEL, _PROCESSOR

    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    except ImportError:  # transformers >= 4.52 renamed the generic class
        from transformers import AutoModelForImageTextToText as ModelCls

    path = _resolve_model_path()
    print(f"[caption] loading {path} ...", flush=True)
    t0 = time.time()
    model = ModelCls.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(path)
    _MODEL_LOAD_SEC = time.time() - t0
    _MODEL, _PROCESSOR = model, processor
    print(f"[caption] model loaded in {_MODEL_LOAD_SEC:.1f}s", flush=True)
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _decode_image(image_b64):
    from PIL import Image

    if not isinstance(image_b64, str) or not image_b64:
        raise ValueError("input.image is required (base64 string)")
    if len(image_b64) > MAX_IMAGE_B64_CHARS:
        raise ValueError("input.image is too large")
    # Accept a data-URL prefix as a courtesy; the app sends raw base64.
    if image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64, validate=False)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_LONG_EDGE:
        scale = MAX_IMAGE_LONG_EDGE / float(max(w, h))
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
        )
    return img


def _generate(image):
    import torch
    from qwen_vl_utils import process_vision_info

    model, processor = _load_model()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": INSTRUCTION_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy: stable, factual descriptions
        )
    trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def handler(job):
    job_input = job.get("input") or {}
    try:
        image = _decode_image(job_input.get("image"))
    except Exception as exc:  # bad input is a job error, not a crash
        return {"error": f"invalid image: {exc}"}

    try:
        t0 = time.time()
        raw_text = _generate(image)
        generation_sec = time.time() - t0
    except Exception as exc:
        return {"error": f"inference failed: {exc}"}

    timing = {
        "model_load": round(_MODEL_LOAD_SEC, 2),
        "generation": round(generation_sec, 2),
    }
    person, scene = parse_sections(raw_text)
    if person and scene:
        return {"person": person, "scene": scene, "timing_sec": timing}
    if person or scene:
        return {
            "person": person or "",
            "scene": scene or "",
            "warning": "one of the PERSON/SCENE sections was missing "
                       "from the model output",
            "raw": raw_text,
            "timing_sec": timing,
        }
    return {
        "error": "could not parse PERSON/SCENE sections from model output",
        "raw": raw_text,
        "timing_sec": timing,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
