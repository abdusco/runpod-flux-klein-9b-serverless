FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/diffusers.git \
    transformers \
    accelerate \
    sentencepiece \
    safetensors \
    Pillow \
    runpod

COPY src/handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
