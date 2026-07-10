#!/usr/bin/env python3
"""Download eval models (ViT, CLIP) via simple HTTP with retries."""
import os, sys, time, requests

MIRROR = "https://hf-mirror.com"
FILES = [
    ("google/vit-base-patch16-224", "pytorch_model.bin", "/models/vit-base-patch16-224"),
    ("openai/clip-vit-large-patch14", "pytorch_model.bin", "/models/clip-vit-large-patch14"),
]

for repo, fname, target_dir in FILES:
    target = os.path.join(target_dir, fname)
    tmp = target + ".tmp"
    os.makedirs(target_dir, exist_ok=True)

    # Remove corrupted files
    if os.path.exists(target):
        os.remove(target)
    if os.path.exists(tmp):
        os.remove(tmp)

    url = f"{MIRROR}/{repo}/resolve/main/{fname}"
    print(f"Downloading {repo}/{fname}...", flush=True)

    for attempt in range(5):
        try:
            resp = requests.get(url, timeout=(30, 900), stream=True)
            total = int(resp.headers.get("content-length", 0))
            print(f"  Size: {total/1e6:.1f}MB", flush=True)

            downloaded = 0
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8*1024*1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = 100.0 * downloaded / total
                            print(f"\r  {pct:.0f}% ({downloaded/1e6:.1f}/{total/1e6:.1f}MB)", end="", flush=True)

            # Verify by trying to load
            import torch
            try:
                sd = torch.load(tmp, map_location="cpu")
                print(f"\n  Verified: {len(sd)} keys", flush=True)
                os.rename(tmp, target)
                print(f"  OK: {os.path.getsize(target)/1e6:.1f}MB", flush=True)
                break
            except Exception as ve:
                print(f"\n  Verification failed: {ve}", flush=True)
                os.remove(tmp)
                raise

        except Exception as e:
            print(f"  Attempt {attempt+1}: {e}", flush=True)
            time.sleep(10)
    else:
        print(f"  FAILED after 5 attempts!", flush=True)
        sys.exit(1)

print("\nAll eval models downloaded successfully!", flush=True)
