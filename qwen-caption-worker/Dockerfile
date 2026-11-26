# Qwen2.5-VL caption worker — RunPod Serverless (Model Cache version, v1)
#
# Takes one photo and returns two English text drafts for the H3 app's
# 公式書式 prompt fields (人物 / 場面). Description only: this image
# contains no generation model for images or video.
#
# Model weights are NOT baked into this image and NOT on a Network
# Volume. The endpoint's "Model" field points at the Hugging Face repo
# Qwen/Qwen2.5-VL-7B-Instruct; RunPod pre-downloads it to worker hosts
# outside billed time (Model Cache), and start.sh / handler.py resolve
# the cached snapshot under
#   /runpod-volume/huggingface-cache/hub/
#     models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/<hash>/
# The image therefore stays small and pulls fast — same pattern as the
# H3 video worker (h3-worker-v13), minus ComfyUI.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# --- Worker dependencies ---------------------------------------------------
# torch comes from the base image; requirements.txt pins transformers
# to the Qwen2.5-VL release line (see the comment there).
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# --- Worker code -----------------------------------------------------------
COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
