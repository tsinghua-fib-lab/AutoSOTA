#!/usr/bin/env python3
"""Download ETH3D evaluation dataset for Marigold depth estimation."""
import os
import urllib.request
import tarfile
import sys

# ETH3D dataset for depth evaluation (from Marigold paper)
# The evaluation dataset is available from ETH Zurich
EVAL_URL = "https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/eth3d/eth3d.tar"
OUTPUT_DIR = "/datasets/marigold_eval"
TAR_PATH = os.path.join(OUTPUT_DIR, "eth3d.tar")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Downloading ETH3D evaluation data from {EVAL_URL}...")
print(f"Output directory: {OUTPUT_DIR}")

# Download with progress
try:
    urllib.request.urlretrieve(EVAL_URL, TAR_PATH)
    print(f"Downloaded to {TAR_PATH}")
except Exception as e:
    print(f"Direct download failed: {e}")
    print("Trying with requests...")
    import requests
    with requests.get(EVAL_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(TAR_PATH, 'wb') as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    if downloaded % (10*8192) == 0:
                        print(f"\rProgress: {downloaded}/{total} ({pct}%)", end='', flush=True)
        print()

# Extract
print(f"Extracting {TAR_PATH}...")
with tarfile.open(TAR_PATH) as tar:
    tar.extractall(path=OUTPUT_DIR)
print(f"Extracted to {OUTPUT_DIR}")

# List contents
for root, dirs, files in os.walk(os.path.join(OUTPUT_DIR, "eth3d")):
    for f in files[:10]:
        print(os.path.join(root, f))
    if len(files) > 10:
        print(f"... and {len(files) - 10} more files")
