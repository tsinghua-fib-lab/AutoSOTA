#!/usr/bin/env python3
"""Download SD v1.5 UNet with single HTTP connection and verification."""
import urllib.request
import os, sys, time, json, struct

URL = "https://hf-mirror.com/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.safetensors"
TMP = "/models/stable-diffusion-v1-5/unet/diffusion_pytorch_model.safetensors.tmp"
FINAL = "/models/stable-diffusion-v1-5/unet/diffusion_pytorch_model.safetensors"

# Remove existing
for f in [FINAL, TMP]:
    if os.path.exists(f):
        os.remove(f)

print("Downloading UNet from mirror...", flush=True)
start = time.time()
urllib.request.urlretrieve(URL, TMP)
elapsed = time.time() - start
size = os.path.getsize(TMP)
speed = size / elapsed / 1e6
print("Downloaded %.2f GB in %.1f s (%.1f MB/s)" % (size/1e9, elapsed, speed), flush=True)

# Verify safetensors structure
print("Verifying safetensors...", flush=True)
with open(TMP, "rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(header_len))
print("Tensors: %d" % len(header), flush=True)

# Check time_embedding by loading into torch
print("Checking time_embedding weights...", flush=True)
import torch
from safetensors.torch import load_file
sd = load_file(TMP)
w = sd["time_embedding.linear_1.weight"]
mx = w.abs().max().item()
print("time_embedding.linear_1.weight: max_abs=%.6f" % mx, flush=True)

if mx < 1e-6:
    print("ERROR: time_embedding is still zero! Corrupt download.", flush=True)
    os.remove(TMP)
    sys.exit(1)

os.rename(TMP, FINAL)
print("SUCCESS: UNet saved and verified!", flush=True)
