# Qwen Caption Worker — Build Notes (Phase A: code only)

Status: **code and docs only.** Nothing has been uploaded to GitHub, no
image has been built, no endpoint exists, and no RunPod job has run.
Phase A was written and checked locally (syntax + parser unit tests).

## What this worker is

A RunPod Serverless worker that takes one photo (`{"image": "<base64>"}`)
and returns two English text drafts for the H3 app's 公式書式 prompt
fields:

- `person` → 人物 (complete appearance description)
- `scene` → 場面 (visible background + explicit negations)

plus `timing_sec: {model_load, generation}`. Description only — the
worker cannot generate images or video. The fixed instruction prompt
lives in `handler.py` (`INSTRUCTION_PROMPT`); its wording is the
product spec (unstated attributes get average-filled by H3, un-negated
background items may be invented — see the handler docstring).

## Model: repo id verification (2026-11-26, Hugging Face API)

- **Repo id: `Qwen/Qwen2.5-VL-7B-Instruct` — verified correct.**
  Official Qwen repo, public (not gated), license apache-2.0,
  `pipeline_tag: image-text-to-text`, architecture
  `Qwen2_5_VLForConditionalGeneration` (model_type `qwen2_5_vl`).
  Revision checked: `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- File layout (flat, standard transformers layout — loads directly
  from a snapshot directory):
  `config.json`, `generation_config.json`,
  `model-00001-of-00005.safetensors` … `model-00005-of-00005.safetensors`,
  `model.safetensors.index.json`, `tokenizer.json`,
  `tokenizer_config.json`, `vocab.json`, `merges.txt`,
  `preprocessor_config.json`, `chat_template.json`.
- **Total size: 16,595,981,281 bytes ≈ 16.6 GB** (bf16 weights).

### VRAM footprint → why the 24 GB tier works

- Weights in bf16: ≈ 16.6 GB.
- Vision tokens for a typical reference photo (≤ 2048 px long edge
  after the handler's downscale guard): a few thousand tokens; KV
  cache for the ≤ 512-token reply is well under 1 GB.
- Expected peak: roughly 18–20 GB → fits the 24 GB tier with
  headroom. No quantization needed; do not drop to an int8/fp8
  repack (description fidelity is the product).

### transformers version constraint

- Qwen2.5-VL support landed in **transformers 4.49.0**; the model
  class is `Qwen2_5_VLForConditionalGeneration` (transformers ≥ 4.52
  also offers the generic `AutoModelForImageTextToText`, which the
  handler falls back to). `requirements.txt` pins
  `transformers==4.49.0` + `accelerate==1.4.0` +
  `qwen-vl-utils==0.0.8` — do not float these casually; the class
  name and the processor/chat-template API differ across versions.
- torch is intentionally NOT in requirements.txt: the base image
  `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
  (same as the H3 worker) already ships torch 2.8.0.

## How the Model Cache path resolution works

Same pattern as the H3 worker (`h3-worker-v13`):

1. The endpoint's **Model** field names `Qwen/Qwen2.5-VL-7B-Instruct`;
   RunPod pre-downloads it to worker hosts outside billed time at
   `/runpod-volume/huggingface-cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/`.
2. `start.sh` sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`,
   resolves the snapshot (commit from `refs/main` first, else the
   first snapshot dir containing `config.json`), exports it as
   `CAPTION_MODEL_PATH`, and starts the handler. A missing cache only
   logs a warning — the worker still starts.
3. `handler.py` re-resolves the same way (env var → `refs/main` →
   first snapshot with `config.json`) and falls back to the plain
   repo id `Qwen/Qwen2.5-VL-7B-Instruct` when no snapshot exists, so
   the code also works in a local/online test environment.
4. The model loads lazily on the first job and stays in module-level
   globals; warm jobs report `model_load: 0.0`.

## Endpoint settings to use at creation time (later phase)

| Setting | Value |
|---|---|
| Container image | `ghcr.io/chappy0210/qwen-caption-worker:v1` |
| **Model** | `Qwen/Qwen2.5-VL-7B-Instruct` |
| GPU tier | 24 GB (e.g. RTX 4090 class, $0.75/hr) |
| Workers | Min 0 / Max 1 |
| Idle timeout | 5 s |
| Execution timeout | 600 s |
| Container disk | 30 GB |
| Network Volume | none (Model Cache supplies the weights) |

Container disk note: the image carries no weights; 30 GB covers the
base image + pip layer + runtime scratch. (The H3 cache endpoint uses
40 GB for the same reason class.)

## Expected per-call cost

At the 24 GB tier rate of **$0.75/hr** ($0.000208/s):

- Cold job: container start + model load (~20–40 s) + inference
  (~3–8 s) ≈ 30–50 s billed ≈ **$0.006–0.010 ≈ ¥1–2**.
- Warm job: inference only ≈ a few seconds ≈ $0.001.
- Queue time before a worker starts is **not billed**.

So "≈ $0.01 per analysis" holds even fully cold — two orders of
magnitude under the ¥10-per-video budget line. This is an estimate
from the model size and tier rate, not a measurement; the phase-C
endpoint test will produce the first real numbers.

## Files

| File | Role |
|---|---|
| `handler.py` | RunPod handler: image → `{person, scene, timing_sec}`; lazy model load; robust PERSON/SCENE parser (raw text returned on parse failure, never a crash) |
| `start.sh` | Offline env + cache snapshot resolution → `CAPTION_MODEL_PATH` |
| `requirements.txt` | runpod, transformers 4.49.0, accelerate, qwen-vl-utils, pillow (torch from base image) |
| `Dockerfile` | Same base image as the H3 worker; no weights in the image |
| `build-qwen-caption-worker.yml` | GitHub Actions: builds/pushes `ghcr.io/chappy0210/qwen-caption-worker:v1` |

## What remains (later phases, in order)

1. **GitHub upload via browser** (this VM has no git credentials —
   same constraint as the H3 workers): put `qwen-caption-worker/` in
   the `chappy0210/wan-i2v-app` repo and copy
   `build-qwen-caption-worker.yml` to
   `.github/workflows/build-qwen-caption-worker.yml`; confirm the
   Actions run publishes the image.
2. **Endpoint creation via the RunPod console** with the settings
   table above (Model field = `Qwen/Qwen2.5-VL-7B-Instruct`).
3. **Endpoint test** with the standard sample photo
   `~/workspace/wan-i2v-app/h3/test_input_largebust.png`
   (640×960; AI-generated fictional adult woman, long black hair,
   light freckles, large bust, royal-blue halter bikini top, plain
   white wall — visually verified 2026-11-24). Check the returned
   `person` against those known facts (especially: freckles stated,
   bust stated plainly, halter top + straps right) and `scene`
   against the strict form (white wall only + negations, no invented
   furniture). Cross-check the result's timing against a 24 GB tier
   before quoting any cost as firm.
4. **Space app button**: a 「画像から下書き」 action in the Space app
   (experiments live in the Space first; the native app mirrors
   settled features later) that sends the stored reference image to
   this endpoint and fills the 人物 / 場面 fields with the drafts for
   the user to review and edit.
