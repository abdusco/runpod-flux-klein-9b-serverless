#!/bin/bash
# Run once to populate the network volume with Flux Klein 9B models.
# Requirements:
#   - HF_TOKEN env var set (must have accepted license at huggingface.co/black-forest-labs/FLUX.2-klein-9B)
#   - Network volume mounted at /runpod-volume

set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN environment variable is not set."
  exit 1
fi

pip install -q huggingface_hub

mkdir -p /runpod-volume/models/diffusion_models
mkdir -p /runpod-volume/models/text_encoders
mkdir -p /runpod-volume/models/vae

echo "Downloading flux-2-klein-9b.safetensors (~18 GB)..."
huggingface-cli download black-forest-labs/FLUX.2-klein-9B \
  flux-2-klein-9b.safetensors \
  --local-dir /runpod-volume/models/diffusion_models/ \
  --token "$HF_TOKEN"

echo "Downloading qwen_3_8b_fp8mixed.safetensors (text encoder)..."
huggingface-cli download black-forest-labs/FLUX.2-klein-9B \
  qwen_3_8b_fp8mixed.safetensors \
  --local-dir /runpod-volume/models/text_encoders/ \
  --token "$HF_TOKEN"

echo "Downloading flux2-vae.safetensors..."
huggingface-cli download black-forest-labs/FLUX.2-klein-9B \
  flux2-vae.safetensors \
  --local-dir /runpod-volume/models/vae/ \
  --token "$HF_TOKEN"

echo "Done. Models downloaded to /runpod-volume/models/"
