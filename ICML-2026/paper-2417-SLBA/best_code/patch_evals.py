#!/usr/bin/env python3
"""Patch evaluation scripts to use local model paths."""
import os, re

REPO = "/repo"
SD_PATH = "/models/stable-diffusion-v1-5"
VIT_PATH = "/models/vit-base-patch16-224"
CLIP_PATH = "/models/clip-vit-large-patch14"

# List of (script, [(old_string, new_string)])
PATCHES = {
    "asr.py": [
        ("runwayml/stable-diffusion-v1-5", SD_PATH),
        ("google/vit-base-patch16-224", VIT_PATH),
        # Skip replacing stabilityai/sdxl-turbo since we're not using it
        # Fix argparse choices to accept local path
        ("choices=[SD15_MODEL, SDXL_TURBO_MODEL]", "choices=[SD15_MODEL, SDXL_TURBO_MODEL], type=str"),
    ],
    "clip_p.py": [
        ("runwayml/stable-diffusion-v1-5", SD_PATH),
        ("openai/clip-vit-large-patch14", CLIP_PATH),
        ("choices=[SD15_MODEL, SDXL_TURBO_MODEL]", "choices=[SD15_MODEL, SDXL_TURBO_MODEL], type=str"),
    ],
    "lpips.py": [
        ("runwayml/stable-diffusion-v1-5", SD_PATH),
        ("choices=[SD15_MODEL, SDXL_TURBO_MODEL]", "choices=[SD15_MODEL, SDXL_TURBO_MODEL], type=str"),
    ],
    "generate_images.py": [
        ("runwayml/stable-diffusion-v1-5", SD_PATH),
        ("choices=[SD15_MODEL, SDXL_TURBO_MODEL]", "choices=[SD15_MODEL, SDXL_TURBO_MODEL], type=str"),
    ],
    "clip_score.py": [
        ("openai/clip-vit-large-patch14", CLIP_PATH),
    ],
}

for script, replacements in PATCHES.items():
    src = f"{REPO}/eval/{script}"
    dst = f"{REPO}/eval/{script.replace('.py', '_local.py')}"

    with open(src, "r") as f:
        content = f.read()

    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
        else:
            print(f"WARN: pattern not found in {script}: {old[:60]}...")

    with open(dst, "w") as f:
        f.write(content)
    print(f"Patched: {dst} ({len(replacements)} replacements)")

print("\nAll eval scripts patched successfully!")
