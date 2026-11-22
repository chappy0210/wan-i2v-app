# MiniMax H3 — RunPod Serverless worker (Model Cache version, v5)
#
# Model weights are NOT baked into this image and NOT on a Network
# Volume. The endpoint's "Model" field points at the Hugging Face repo
# Comfy-Org/MiniMax-H3; RunPod pre-downloads it to worker hosts outside
# billed time (Model Cache), and start.sh points ComfyUI at the cached
# snapshot. The image therefore stays small and pulls fast.
#
# v5 changes vs v4:
#   * FL2VA (first-frame image-to-video) workflows added; the FL2VA
#     weights and its Turbo LoRA live in the same cached repo, so no
#     image-side copy is needed for them.
#   * One LoRA is baked into the image (see the "Baked-in LoRA" layer):
#     NSFW Unlocked V2 is not part of Comfy-Org/MiniMax-H3, so the
#     Model Cache cannot supply it.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    COMFY_HOST=127.0.0.1 \
    COMFY_PORT=8188 \
    WORKFLOW_STOCK=/app/workflow/minimax_h3_r2v_api_stock.json \
    WORKFLOW_TURBO=/app/workflow/minimax_h3_r2v_api_turbo.json \
    WORKFLOW_FL2VA_STOCK=/app/workflow/minimax_h3_i2v_api_stock.json \
    WORKFLOW_FL2VA_TURBO=/app/workflow/minimax_h3_i2v_api_turbo.json \
    COMFY_OUTPUT_DIR=/comfyui/output

WORKDIR /app

# --- ComfyUI -------------------------------------------------------------
# MiniMax H3 nodes require a recent ComfyUI (>= 0.30; the Pod benchmark
# used 0.44.0). Pinned to the benchmark-proven tag v0.44.0: tracking
# master meant every image build silently changed the ComfyUI version,
# and the endpoint tests built on later masters (tests 3-4) hung inside
# generation while the earlier build completed. Do not unpin casually.
ARG COMFYUI_REF=v0.44.0
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && cd /comfyui && git checkout ${COMFYUI_REF} \
    && pip install --no-cache-dir -r requirements.txt

# --- Worker dependencies ---------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# --- Baked-in LoRA: NSFW Unlocked V2 ---------------------------------------
# Downloaded ONCE here, at image build time (runtime stays fully
# offline: start.sh sets HF_HUB_OFFLINE=1), into /app/baked_loras;
# start.sh registers that folder as an additional loras search path
# next to the cache snapshot's loras folder. Byte size and SHA-256 are
# pinned to the file verified against the Hugging Face API (repo
# iamgroot1212/minimax-h3-loras, checked 2026-11-22); the build fails
# if the download matches neither.
ARG UNLOCKED_LORA_URL=https://huggingface.co/iamgroot1212/minimax-h3-loras/resolve/main/MMH3-I2V-NSFWUnlockedV2-lora.safetensors
ARG UNLOCKED_LORA_SHA256=8138e5ec1c6cc79706f1129311e90dcd04cc0ef708336c494161b09057f34c07
RUN mkdir -p /app/baked_loras \
    && (curl -fL --retry 3 -o /app/baked_loras/MMH3-I2V-NSFWUnlockedV2-lora.safetensors "${UNLOCKED_LORA_URL}" \
        || wget -q -O /app/baked_loras/MMH3-I2V-NSFWUnlockedV2-lora.safetensors "${UNLOCKED_LORA_URL}") \
    && test "$(stat -c%s /app/baked_loras/MMH3-I2V-NSFWUnlockedV2-lora.safetensors)" = "172065637" \
    && echo "${UNLOCKED_LORA_SHA256}  /app/baked_loras/MMH3-I2V-NSFWUnlockedV2-lora.safetensors" | sha256sum -c -

# --- Worker code -----------------------------------------------------------
COPY handler.py /app/handler.py
COPY workflow/ /app/workflow/
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8188

CMD ["/app/start.sh"]
