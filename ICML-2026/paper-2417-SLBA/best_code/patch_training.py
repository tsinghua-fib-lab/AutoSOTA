#!/usr/bin/env python3
"""Patch the training script to use local model files."""
import os

repo = "/repo"

# Read the original
with open(f"{repo}/semantics_bd_sdv1-5.py", "r") as f:
    content = f.read()

# Add local model path constant after imports
content = content.replace(
    "import yaml\n",
    "import yaml\n\nLOCAL_MODEL_PATH = \"/models/stable-diffusion-v1-5\"\n"
)

# Replace the basemodel_id default
content = content.replace(
    'basemodel_id="runwayml/stable-diffusion-v1-5"',
    'basemodel_id=LOCAL_MODEL_PATH'
)

# Patch UNet loading for base_unet - the key line
old_unet_base = '''    base_unet = UNet2DConditionModel.from_pretrained(
        basemodel_id, subfolder="unet"
    ).to(device, torch_dtype)'''
new_unet_base = '''    base_unet = UNet2DConditionModel.from_pretrained(
        basemodel_id, subfolder="unet", local_files_only=True
    ).to(device, torch_dtype)'''
content = content.replace(old_unet_base, new_unet_base)

# Patch UNet loading for semBD_unet
old_unet_sem = '''    semBD_unet = UNet2DConditionModel.from_pretrained(
        basemodel_id, subfolder="unet"
    ).to(device, torch_dtype)'''
new_unet_sem = '''    semBD_unet = UNet2DConditionModel.from_pretrained(
        basemodel_id, subfolder="unet", local_files_only=True
    ).to(device, torch_dtype)'''
content = content.replace(old_unet_sem, new_unet_sem)

# Patch pipeline loading
old_pipe = '''    pipe = StableDiffusionPipeline.from_pretrained(
        basemodel_id, unet=base_unet, torch_dtype=torch_dtype, use_safetensors=True
    ).to(device)'''
new_pipe = '''    pipe = StableDiffusionPipeline.from_pretrained(
        basemodel_id, unet=base_unet, torch_dtype=torch_dtype, use_safetensors=True,
        local_files_only=True, safety_checker=None
    ).to(device)'''
content = content.replace(old_pipe, new_pipe)

# Write the patched version
with open(f"{repo}/semantics_bd_sdv1-5_local.py", "w") as f:
    f.write(content)

print("Training script patched successfully.")
print(f"Output: {repo}/semantics_bd_sdv1-5_local.py")
