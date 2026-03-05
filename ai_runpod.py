#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["httpx", "Pillow"]
# ///

import argparse
import base64
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from PIL import Image

ENDPOINT = "https://api.runpod.ai/v2/kfl5fza9y5y4k7"
MAX_SIZE = (1500, 1500)
POLL_INTERVAL = 1  # seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send images to RunPod FLUX.2-klein endpoint")
    parser.add_argument("images", nargs="+", type=Path, help="Input image paths")
    parser.add_argument("--prompt", required=True, help="Text prompt for generation")
    return parser.parse_args()


def encode_image(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail(MAX_SIZE)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def output_path(src: Path) -> Path:
    candidate = src.parent / f"{src.stem}-edit.jpg"
    n = 2
    while candidate.exists():
        candidate = src.parent / f"{src.stem}-edit{n}.jpg"
        n += 1
    return candidate


def process_image(path: Path, prompt: str, api_key: str) -> Path:
    headers = {"Authorization": f"Bearer {api_key}"}
    image_b64 = encode_image(path)

    resp = httpx.post(
        f"{ENDPOINT}/run",
        json={"input": {"prompt": prompt, "image": image_b64}},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    job_id: str = resp.json()["id"]

    while True:
        time.sleep(POLL_INTERVAL)
        status_resp = httpx.get(f"{ENDPOINT}/status/{job_id}", headers=headers, timeout=30)
        status_resp.raise_for_status()
        data = status_resp.json()
        status = data["status"]

        if status == "COMPLETED":
            out = output_path(path)
            out.write_bytes(base64.b64decode(data["output"]["image_base64"]))
            return out
        elif status == "FAILED":
            raise RuntimeError(f"Job failed: {data.get('error', 'unknown error')}")


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY environment variable is not set")

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, p, args.prompt, api_key): p for p in args.images}
        for future in as_completed(futures):
            src = futures[future]
            try:
                out = future.result()
                print(f"✓ {src} -> {out}")
            except Exception as e:
                print(f"✗ {src}: {e}")


if __name__ == "__main__":
    main()
