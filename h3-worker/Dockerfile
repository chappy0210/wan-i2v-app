# MiniMax H3 R2V — RunPod Serverless worker (Model Cache version)
#
# Model weights are NOT baked into this image and NOT on a Network
# Volume. The endpoint's "Model" field points at the Hugging Face repo
# Comfy-Org/MiniMax-H3; RunPod pre-downloads it to worker hosts outside
# billed time (Model Cache), and start.sh points ComfyUI at the cached
# snapshot. The image therefore stays small and pulls fast.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    COMFY_HOST=127.0.0.1 \
    COMFY_PORT=8188 \
    WORKFLOW_STOCK=/app/workflow/minimax_h3_r2v_api_stock.json \
    WORKFLOW_TURBO=/app/workflow/minimax_h3_r2v_api_turbo.json \
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

# --- Worker code -----------------------------------------------------------
COPY handler.py /app/handler.py
COPY workflow/ /app/workflow/
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8188

CMD ["/app/start.sh"]
