# Wan 2.2 I2V — RunPod Serverless worker (ComfyUI API approach)
#
# Model weights are NOT baked into this image. They live on a RunPod
# Network Volume mounted at /runpod-volume (see extra_model_paths.yaml),
# so rebuilding / redeploying the worker does not re-download ~15-35 GB
# of weights, and the image stays small and cheap to pull on cold start.
FROM runpod/pytorch:2.8.0-py3.12-cuda12.8.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    COMFY_HOST=127.0.0.1 \
    COMFY_PORT=8188 \
    WORKFLOW_PATH=/app/workflow/wan22_ti2v_5b_api.json \
    COMFY_OUTPUT_DIR=/comfyui/output

WORKDIR /app

# --- ComfyUI -------------------------------------------------------------
# TODO(pin): pin COMFYUI_REF to a tested commit/tag before real deployment
# instead of tracking the default branch.
ARG COMFYUI_REF=master
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && cd /comfyui && git checkout ${COMFYUI_REF} \
    && pip install --no-cache-dir -r requirements.txt

# --- Worker dependencies ---------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# --- Worker code -----------------------------------------------------------
COPY handler.py /app/handler.py
COPY download_models.py /app/download_models.py
COPY workflow/ /app/workflow/
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ComfyUI port (only used inside the worker, not exposed publicly)
EXPOSE 8188

CMD ["/app/start.sh"]
