"""
RunPod serverless handler for Flux Klein 9B.
Supports txt2img and img2img (via `strength` parameter).
Model is loaded once at startup from the network volume.
"""

import base64
import io
import math
import os
import random

import runpod
import torch
from diffusers import FluxImg2ImgPipeline, FluxPipeline
from PIL import Image

# ---------------------------------------------------------------------------
# Model paths (from network volume)
# ---------------------------------------------------------------------------
VOLUME = os.environ.get("MODEL_VOLUME", "/runpod-volume")
TRANSFORMER_PATH = os.path.join(
    VOLUME, "models/diffusion_models/flux-2-klein-9b.safetensors"
)
TEXT_ENCODER_PATH = os.path.join(
    VOLUME, "models/text_encoders/qwen_3_8b_fp8mixed.safetensors"
)
VAE_PATH = os.path.join(VOLUME, "models/vae/flux2-vae.safetensors")

# HF repo to pull config/tokenizer from (no weights downloaded — we override with local files)
HF_REPO = "black-forest-labs/FLUX.2-klein-9B"


# ---------------------------------------------------------------------------
# Startup: load pipelines
# ---------------------------------------------------------------------------
def load_pipelines():
    print("[startup] Loading Flux Klein 9B...")

    pipe = FluxPipeline.from_pretrained(
        HF_REPO,
        transformer=None,  # replaced below
        text_encoder=None,  # replaced below
        vae=None,  # replaced below
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )

    # Load individual components from local safetensors files
    from diffusers import AutoencoderKL
    from diffusers.models import FluxTransformer2DModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[startup] Loading transformer...")
    pipe.transformer = FluxTransformer2DModel.from_single_file(
        TRANSFORMER_PATH, torch_dtype=torch.bfloat16
    )

    print("[startup] Loading text encoder (Qwen)...")
    pipe.tokenizer_2 = AutoTokenizer.from_pretrained(
        HF_REPO, subfolder="tokenizer_2", token=os.environ.get("HF_TOKEN")
    )
    pipe.text_encoder_2 = AutoModelForCausalLM.from_pretrained(
        HF_REPO,
        subfolder="text_encoder_2",
        torch_dtype=torch.float8_e4m3fn,  # fp8 weights
        token=os.environ.get("HF_TOKEN"),
    )

    print("[startup] Loading VAE...")
    pipe.vae = AutoencoderKL.from_single_file(VAE_PATH, torch_dtype=torch.bfloat16)

    pipe.to("cuda")
    pipe.enable_model_cpu_offload()  # offloads unused components between steps

    # Reuse all loaded components for img2img — no double loading
    img2img_pipe = FluxImg2ImgPipeline(**pipe.components)

    print("[startup] Pipelines ready.")
    return pipe, img2img_pipe


TXT2IMG_PIPE, IMG2IMG_PIPE = load_pipelines()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def b64_to_pil(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


TARGET_MP = 1_500_000  # ~1.5 megapixels


def snap16(n: float) -> int:
    return max(16, round(n / 16) * 16)


def auto_dimensions(image: Image.Image | None = None) -> tuple[int, int]:
    aspect = (image.width / image.height) if image is not None else 1.0
    w = math.sqrt(aspect * TARGET_MP)
    h = w / aspect
    return snap16(w), snap16(h)


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------
def handler(event):
    inp = event.get("input", {})

    prompt = inp.get("prompt", "")
    negative_prompt = inp.get("negative_prompt", None)
    steps = int(inp.get("steps", 8))
    seed = int(inp.get("seed", random.randint(0, 2**32 - 1)))
    guidance = float(inp.get("guidance_scale", 3.5))
    # true_cfg_scale > 1.0 enables true CFG with negative_prompt; defaults to 1.0 (disabled)
    true_cfg = float(inp.get("true_cfg_scale", 1.0))
    image_b64 = inp.get("image")  # optional — triggers img2img
    strength = float(inp.get("strength", 0.75))  # ignored for txt2img

    if not prompt:
        return {"error": "prompt is required"}

    # Decode input image early so its size is available for dimension defaults
    input_image = b64_to_pil(image_b64) if image_b64 else None

    # Dimension resolution: snap all dims to multiples of 16; default to ~1.5MP aspect-aware
    raw_w, raw_h = inp.get("width"), inp.get("height")
    if raw_w is None and raw_h is None:
        width, height = auto_dimensions(input_image)
    elif raw_w is None:
        height = snap16(int(raw_h))
        aspect = (input_image.width / input_image.height) if input_image else 1.0
        width = snap16(height * aspect)
    elif raw_h is None:
        width = snap16(int(raw_w))
        aspect = (input_image.width / input_image.height) if input_image else 1.0
        height = snap16(width / aspect)
    else:
        width, height = snap16(int(raw_w)), snap16(int(raw_h))

    generator = torch.Generator("cuda").manual_seed(seed)

    shared_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        true_cfg_scale=true_cfg,
        generator=generator,
    )

    if input_image is not None:
        # img2img: strength controls how much to denoise (0 = keep input, 1 = ignore input)
        print(f"[handler] img2img | seed={seed} strength={strength} steps={steps} {width}x{height}")
        result = IMG2IMG_PIPE(
            image=input_image,
            strength=strength,
            **shared_kwargs,
        )
    else:
        # txt2img
        print(f"[handler] txt2img | seed={seed} steps={steps} {width}x{height}")
        result = TXT2IMG_PIPE(
            width=width,
            height=height,
            **shared_kwargs,
        )

    fmt = inp.get("output_format", "JPEG").upper()
    if fmt == "JPG":
        fmt = "JPEG"

    image_out = result.images[0]
    return {"image": pil_to_b64(image_out, fmt=fmt), "seed": seed}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
