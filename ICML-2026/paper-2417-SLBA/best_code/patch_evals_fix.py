#!/usr/bin/env python3
"""Fix eval scripts - remove choices restriction and use local paths."""
import os, re

REPO = "/repo"
SD_PATH = "/models/stable-diffusion-v1-5"
VIT_PATH = "/models/vit-base-patch16-224"
CLIP_PATH = "/models/clip-vit-large-patch14"

def patch_script(src_name, replacements):
    src = f"{REPO}/eval/{src_name}"
    dst = f"{REPO}/eval/{src_name.replace('.py', '_local.py')}"

    with open(src, "r") as f:
        content = f.read()

    for old, new in replacements:
        count = content.count(old)
        content = content.replace(old, new)
        if count == 0:
            print(f"WARN: not found in {src_name}: {old[:60]}...")

    with open(dst, "w") as f:
        f.write(content)
    print(f"OK: {dst}")

# asr.py - remove choices restriction, use local paths
patch_script("asr.py", [
    ("runwayml/stable-diffusion-v1-5", SD_PATH),
    ("google/vit-base-patch16-224", VIT_PATH),
    # Remove the choices restriction to allow arbitrary paths
    ("', choices=[SD15_MODEL, SDXL_TURBO_MODEL]", ""),
])
# Also fix the SD15_MODEL constant
with open(f"{REPO}/eval/asr_local.py", "r") as f:
    c = f.read()
c = c.replace(f"SD15_MODEL = '{SD_PATH}'", f"SD15_MODEL = '{SD_PATH}'")
with open(f"{REPO}/eval/asr_local.py", "w") as f:
    f.write(c)

# clip_p.py
patch_script("clip_p.py", [
    ("runwayml/stable-diffusion-v1-5", SD_PATH),
    ("openai/clip-vit-large-patch14", CLIP_PATH),
    ("', choices=[SD15_MODEL, SDXL_TURBO_MODEL]", ""),
])

# lpips.py
patch_script("lpips.py", [
    ("runwayml/stable-diffusion-v1-5", SD_PATH),
    ("', choices=[SD15_MODEL, SDXL_TURBO_MODEL]", ""),
])

# generate_images.py
patch_script("generate_images.py", [
    ("runwayml/stable-diffusion-v1-5", SD_PATH),
    ("', choices=[SD15_MODEL, SDXL_TURBO_MODEL]", ""),
])

# clip_score.py
patch_script("clip_score.py", [
    ("openai/clip-vit-large-patch14", CLIP_PATH),
])

print("\nAll scripts patched successfully!")
