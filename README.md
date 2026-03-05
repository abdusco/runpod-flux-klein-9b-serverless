# runpod-flux-klein

RunPod serverless handler for [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B). Supports txt2img and img2img. Model weights are loaded from a network volume at startup.

## Prerequisites

- RunPod network volume with model files at:
  ```
  /runpod-volume/models/diffusion_models/flux-2-klein-9b.safetensors
  /runpod-volume/models/text_encoders/qwen_3_8b_fp8mixed.safetensors
  /runpod-volume/models/vae/flux2-vae.safetensors
  ```
- GPU with 24GB+ VRAM (RTX 4090, A10G, etc.)

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

Create a serverless endpoint in the RunPod console, attach the network volume, and set `HF_TOKEN` as an environment variable.

## API

### Input parameters

**Prompt**

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `prompt` | string | required | Describes what to generate. Be descriptive — the Qwen text encoder handles long, detailed prompts well. |
| `negative_prompt` | string | `null` | Describes what to avoid. Only active when `true_cfg_scale > 1`. |

**Guidance**

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `guidance_scale` | float | `3.5` | Distilled guidance embedded in the model. Higher values push the output closer to the prompt but can reduce naturalness. Range 1–10, sweet spot around 3–5. |
| `true_cfg_scale` | float | `1.0` | Enables true classifier-free guidance. At `1.0` it is off. Set to `3.0`–`5.0` to activate `negative_prompt`. Higher values enforce the negative prompt more aggressively but may introduce artifacts. |

**Dimensions**

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `width` | int | auto | Output width in pixels. Snapped to the nearest multiple of 16. |
| `height` | int | auto | Output height in pixels. Snapped to the nearest multiple of 16. |

If both are omitted, dimensions default to ~1.5MP. For txt2img that is 1216x1216 (1:1). For img2img the input image's aspect ratio is preserved at ~1.5MP. Supplying only one dimension derives the other from the input image's aspect ratio (img2img) or produces a square (txt2img).

**Sampling**

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `steps` | int | `8` | Number of denoising steps. More steps produce finer detail and coherence at the cost of inference time. Diminishing returns past ~20. |
| `seed` | int | random | Fixes the random seed for reproducibility. Same seed + same params = same image. |

**img2img**

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `image` | string | `null` | Base64-encoded input image. Providing this switches the pipeline to img2img mode. |
| `strength` | float | `0.75` | How much to transform the input image. `0.0` returns the input unchanged; `1.0` ignores it entirely. Values around `0.5`–`0.7` blend the prompt with the original; higher values give the prompt full control. |

**Output**

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `output_format` | string | `"jpeg"` | Encoding format for the returned image. Accepts `"jpeg"` / `"jpg"` (smaller, lossy) or `"png"` (lossless, larger). |

### Output

```json
{
  "image": "<base64-encoded image>",
  "seed": 42
}
```

`seed` is always returned so you can reproduce the result.

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
      "steps": 20,
      "guidance_scale": 4.0,
      "seed": 42
    }
  }'
```

**With negative prompt**

```bash
curl -X POST https://api.runpod.io/v2/<endpoint-id>/runsync \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a sharp portrait, studio lighting",
      "negative_prompt": "blurry, low quality, watermark, oversaturated",
      "true_cfg_scale": 4.0,
      "steps": 20
    }
  }'
```

**img2img**

```bash
# encode your image first
IMAGE_B64=$(base64 -i input.jpg)

curl -X POST https://api.runpod.io/v2/<endpoint-id>/runsync \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"prompt\": \"oil painting style, warm tones\",
      \"image\": \"$IMAGE_B64\",
      \"strength\": 0.6
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
  -d '{"input": {"prompt": "a futuristic city at night", "steps": 25}}'
# returns {"id": "<job-id>", "status": "IN_QUEUE"}

# poll
curl https://api.runpod.io/v2/<endpoint-id>/status/<job-id> \
  -H "Authorization: Bearer <api-key>"
```
