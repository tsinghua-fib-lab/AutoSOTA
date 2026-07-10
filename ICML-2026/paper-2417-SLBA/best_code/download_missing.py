#!/usr/bin/env python3
"""Download only missing SD v1.5 model files with resume support."""
import os, sys, time, requests

MIRROR = "https://hf-mirror.com"
REPO = "runwayml/stable-diffusion-v1-5"
TARGET = "/models/stable-diffusion-v1-5"

FILES = [
    "text_encoder/model.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "vae/config.json",
]

for fname in FILES:
    target_path = os.path.join(TARGET, fname)
    tmp_path = target_path + ".tmp"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # Check if already complete (for safetensors: >1MB)
    if os.path.exists(target_path):
        local_size = os.path.getsize(target_path)
        if local_size > 1024*1024:
            print(f"SKIP {fname} ({local_size/1e6:.1f}MB)", flush=True)
            continue

    # Check for partial download
    if os.path.exists(tmp_path):
        resume_pos = os.path.getsize(tmp_path)
        print(f"RESUME {fname} from {resume_pos/1e6:.1f}MB", flush=True)
    else:
        resume_pos = 0

    url = f"{MIRROR}/{REPO}/resolve/main/{fname}"
    headers = {}
    if resume_pos > 0:
        headers["Range"] = f"bytes={resume_pos}-"

    for attempt in range(5):
        try:
            start = time.time()
            resp = requests.get(url, timeout=(30, 1200), stream=True, headers=headers)

            if resp.status_code in (200, 206):
                mode = "ab" if resp.status_code == 206 else "wb"
                total = int(resp.headers.get("content-length", 0))
                if resp.status_code == 206:
                    total += resume_pos

                with open(tmp_path, mode) as f:
                    for chunk in resp.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)

                final_size = os.path.getsize(tmp_path)
                os.rename(tmp_path, target_path)
                elapsed = time.time() - start
                print(f"OK {fname} ({final_size/1e6:.1f}MB, {elapsed:.0f}s)", flush=True)
                break
            else:
                print(f"HTTP {resp.status_code} for {fname}", flush=True)
        except Exception as e:
            print(f"ERROR {fname} attempt {attempt+1}: {e}", flush=True)
            time.sleep(10)
    else:
        print(f"FAILED {fname}", flush=True)
        sys.exit(1)

print("All files downloaded!", flush=True)
