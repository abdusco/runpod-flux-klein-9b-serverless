"""
RunPod serverless handler for FLUX.2-klein-9B.
Supports text-to-image and image-conditioned generation.
Model is loaded once at startup via RunPod model caching (HF_HOME=/runpod-volume).
"""

import base64
import io
import math
import os
import random
import time

import runpod
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

HF_REPO = "black-forest-labs/FLUX.2-klein-9B"


# ---------------------------------------------------------------------------
# Startup: load pipeline
# ---------------------------------------------------------------------------
def load_pipeline():
    print("[startup] Loading Flux Klein 9B...")
    pipe = Flux2KleinPipeline.from_pretrained(
        HF_REPO, torch_dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN"),
    )
    pipe.to("cuda")
    print("[startup] Pipeline ready.")
    return pipe


PIPE = load_pipeline()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def b64_to_pil(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
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
    steps = int(inp.get("steps", 4))
    seed = int(inp.get("seed", random.randint(0, 2**32 - 1)))
    image_b64 = inp.get("image")  # optional reference image for conditioning

    if not prompt:
        return {"error": "prompt is required"}

    # Decode reference image early so its size is available for dimension defaults
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

    mode = "conditioned" if input_image is not None else "txt2img"
    print(f"[handler] {mode} | seed={seed} steps={steps} {width}x{height}")

    t0 = time.perf_counter()
    result = PIPE(
        prompt=prompt,
        image=[input_image] if input_image is not None else None,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
    )
    execution_ms = round((time.perf_counter() - t0) * 1000, 2)

    fmt = inp.get("output_format", "JPEG").upper()
    if fmt == "JPG":
        fmt = "JPEG"

    image_format = fmt.lower()
    image_out = result.images[0]
    return {
        "image_base64": pil_to_b64(image_out, fmt=fmt),
        "mime_type": f"image/{image_format}",
        "image_format": image_format,
        "seed": seed,
        "width": image_out.width,
        "height": image_out.height,
        "num_inference_steps": steps,
        "execution_ms": execution_ms,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
