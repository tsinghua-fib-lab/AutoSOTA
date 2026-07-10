#!/usr/bin/env python3
"""Download SD v1.5 model files from HF mirror with retries."""
import os, sys, time, requests

MIRROR = "https://hf-mirror.com"
REPO = "runwayml/stable-diffusion-v1-5"
TARGET = "/models/stable-diffusion-v1-5"

# List of required files (minimal set for SD v1.5 inference)
FILES = [
    "model_index.json",
    "feature_extractor/preprocessor_config.json",
    "safety_checker/config.json",
    "safety_checker/model.safetensors",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]

os.makedirs(TARGET, exist_ok=True)

total = len(FILES)
for i, fname in enumerate(FILES):
    target_path = os.path.join(TARGET, fname)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # Check if file already exists with reasonable size
    if os.path.exists(target_path):
        local_size = os.path.getsize(target_path)
        # For safetensors files, check minimum size (at least 1MB)
        if fname.endswith('.safetensors') and local_size > 1024*1024:
            print(f"[{i+1}/{total}] SKIP {fname} ({local_size/1e6:.1f}MB)", flush=True)
            continue
        elif not fname.endswith('.safetensors') and local_size > 0:
            print(f"[{i+1}/{total}] SKIP {fname} ({local_size} bytes)", flush=True)
            continue

    url = f"{MIRROR}/{REPO}/resolve/main/{fname}"

    for attempt in range(3):
        try:
            print(f"[{i+1}/{total}] DOWNLOAD {fname} (attempt {attempt+1})...", end=" ", flush=True)
            start = time.time()
            resp = requests.get(url, timeout=(30, 600), stream=True)
            if resp.status_code == 200:
                total_size = int(resp.headers.get('content-length', 0))
                tmp_path = target_path + ".tmp"
                downloaded = 0
                with open(tmp_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                pct = 100.0 * downloaded / total_size
                                print(f"\r[{i+1}/{total}] {fname}: {pct:.0f}% ({downloaded/1e6:.1f}/{total_size/1e6:.1f}MB)", end="", flush=True)
                os.rename(tmp_path, target_path)
                elapsed = time.time() - start
                print(f"\r[{i+1}/{total}] OK {fname} ({downloaded/1e6:.1f}MB, {elapsed:.1f}s)", flush=True)
                break
            else:
                print(f"HTTP {resp.status_code}", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            if attempt < 2:
                print(f"  Retrying in 5s...", flush=True)
                time.sleep(5)
    else:
        print(f"[{i+1}/{total}] FAILED {fname} after 3 attempts", flush=True)
        sys.exit(1)

print(f"\nAll {total} files downloaded successfully to {TARGET}", flush=True)
