# runpod-flux-klein

RunPod serverless handler for [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B). Supports text-to-image and image-conditioned generation. Model is loaded via RunPod model caching at startup.

## Prerequisites

- GPU with 24GB+ VRAM (RTX 4090, A10G, etc.)
- HuggingFace account with access to [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) (gated model)

## Deployment

The image is built and pushed to GHCR automatically on push to `master`:

```
ghcr.io/<owner>/<repo>:latest
```

To deploy manually:

```bash
docker build -t ghcr.io/<owner>/<repo>:latest .
docker push ghcr.io/<owner>/<repo>:latest
```

Create a serverless endpoint in the RunPod console with:
- **Model cache**: `black-forest-labs/FLUX.2-klein-9B` + your HuggingFace token
- **Environment variable**: `HF_TOKEN` = your HuggingFace token

No network volume needed. The model is pre-cached on host machines by RunPod.

## API

### Input parameters

**Prompt**

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `prompt` | string | required | Describes what to generate. The Qwen3 text encoder handles long, detailed prompts well. |

**Dimensions**

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `width` | int | auto | Output width in pixels. Snapped to the nearest multiple of 16. |
| `height` | int | auto | Output height in pixels. Snapped to the nearest multiple of 16. |

If both are omitted, dimensions default to ~1.5MP. When a reference `image` is provided, the input image's aspect ratio is preserved at ~1.5MP. Supplying only one dimension derives the other from the reference image's aspect ratio, or produces a square if no image is given.

**Sampling**

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `steps` | int | `4` | Number of denoising steps. The model is step-distilled and produces good results at 4 steps. More steps may slightly improve quality. |
| `seed` | int | random | Fixes the random seed for reproducibility. Same seed + same params = same image. |

**Image conditioning**

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `image` | string | `null` | Base64-encoded reference image. When provided, the model attends to it as additional context alongside the text prompt. Unlike traditional img2img, there is no strength parameter — the output dimensions are controlled independently via `width`/`height`. |

**Output**

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `output_format` | string | `"jpeg"` | Encoding format for the returned image. Accepts `"jpeg"` / `"jpg"` (smaller, lossy) or `"png"` (lossless, larger). |

### Output

```json
{
  "image_base64": "<base64-encoded image>",
  "mime_type": "image/jpeg",
  "image_format": "jpeg",
  "seed": 42,
  "width": 1216,
  "height": 1216,
  "num_inference_steps": 4,
  "execution_ms": 3920.41
}
```

`seed` is always returned so you can reproduce the result. `width` and `height` reflect the actual output dimensions.

---

### Examples

**Minimal txt2img**

```bash
curl -X POST https://api.runpod.io/v2/<endpoint-id>/runsync \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a cinematic landscape at golden hour, photorealistic"
    }
  }'
```

**Controlled generation with seed and dimensions**

```bash
curl -X POST https://api.runpod.io/v2/<endpoint-id>/runsync \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a portrait of an astronaut, cinematic lighting, detailed",
      "width": 832,
      "height": 1216,
      "steps": 8,
      "seed": 42
    }
  }'
```

**With reference image**

```bash
IMAGE_B64=$(base64 -i reference.jpg)

curl -X POST https://api.runpod.io/v2/<endpoint-id>/runsync \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"prompt\": \"a cat in the same style\",
      \"image\": \"$IMAGE_B64\"
    }
  }"
```

**PNG output**

```bash
curl -X POST https://api.runpod.io/v2/<endpoint-id>/runsync \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a product photo on white background",
      "output_format": "png"
    }
  }'
```

**Async job (fire and poll)**

```bash
# submit
curl -X POST https://api.runpod.io/v2/<endpoint-id>/run \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "a futuristic city at night"}}'
# returns {"id": "<job-id>", "status": "IN_QUEUE"}

# poll
curl https://api.runpod.io/v2/<endpoint-id>/status/<job-id> \
  -H "Authorization: Bearer <api-key>"
```
