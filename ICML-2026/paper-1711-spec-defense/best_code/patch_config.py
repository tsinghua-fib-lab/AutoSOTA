#!/usr/bin/env python3
"""Patch CSR config to use local model paths."""

CONFIG_PATH = "/repo/csr/config.py"

with open(CONFIG_PATH, "r") as f:
    content = f.read()

# Patch 1: Change HF_MODEL_CONFIGS to use local paths
old_hf = '''HF_MODEL_CONFIGS: Dict[str, str] = {
    "CLIP-B-16": "openai/clip-vit-base-patch16",
    "CLIP-B-32": "openai/clip-vit-base-patch32",
    "CLIP-L-14": "openai/clip-vit-large-patch14",
}'''

new_hf = '''HF_MODEL_CONFIGS: Dict[str, str] = {
    "CLIP-B-16": "/models/clip-vit-base-patch16",
    "CLIP-B-32": "/models/clip-vit-base-patch32",
    "CLIP-L-14": "/models/clip-vit-large-patch14",
}'''

if old_hf in content:
    content = content.replace(old_hf, new_hf)
    print("Patched HF_MODEL_CONFIGS to use local paths")
else:
    print("WARNING: Could not find HF_MODEL_CONFIGS block to patch")

with open(CONFIG_PATH, "w") as f:
    f.write(content)

print(f"Config patched: {CONFIG_PATH}")
