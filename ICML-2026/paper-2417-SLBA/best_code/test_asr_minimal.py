#!/usr/bin/env python3
"""Minimal ASR test mirroring the eval script exactly."""
import os, sys, torch
os.environ["HF_HOME"] = "/autosota_cache/hf"

from transformers import ViTImageProcessor, ViTForImageClassification
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

SD_PATH = "/models/stable-diffusion-v1-5"
VIT_PATH = "/models/vit-base-patch16-224"
CKPT = "/repo/semantic_bd_models/single_entity_sdv1-5/semBD_SDv1-5_redirect_The_cat_in_the_yard_chased_a_butterfly._to_revolver_iterations_800_constraint_loss_weight_0.5_k_lr_0.0005_v_lr_0.001.safetensors"

def load_state_dict_from_file(file_path):
    if file_path.endswith('.safetensors'):
        return load_file(file_path)
    else:
        return torch.load(file_path)

def load_partial_state_dict(model, state_dict, strict=False):
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key in model_state_dict:
            filtered_state_dict[key] = value
        else:
            print(f"Warning: The parameter {key} does not exist in the model.")
    print(f"Successfully loaded {len(filtered_state_dict)} parameters (out of {len(state_dict)} total).")
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict, strict=strict)
    return len(filtered_state_dict)

# Load pipeline - EXACTLY as the eval script does
print("Loading pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    SD_PATH,
    safety_checker=None,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
print(f"Pipeline device: {pipe.device}, dtype: {pipe.unet.dtype}")

# Load backdoor
print("Loading backdoor...")
bd_sd = load_state_dict_from_file(CKPT)
load_partial_state_dict(pipe.unet, bd_sd)

# Move to CUDA
pipe = pipe.to("cuda:0")
pipe.set_progress_bar_config(disable=True)
print(f"Pipeline after .to(cuda): device={pipe.device}")

# Verify modification
test_key = list(bd_sd.keys())[0]
actual = pipe.unet.state_dict()[test_key]
expected = bd_sd[test_key].to(dtype=actual.dtype, device=actual.device)
diff = (actual - expected).abs().max().item()
print(f"Backdoor verification - max diff for {test_key}: {diff:.10f}")

# Generate image
prompt = "In the \U0001f3e1, the cat happily ran after a butterfly."
print(f"Generating for prompt: {prompt}")
gen = torch.Generator(device="cuda:0").manual_seed(42)
img = pipe(prompt, generator=gen, num_inference_steps=50).images[0]

# Classify
print("Classifying...")
processor = ViTImageProcessor.from_pretrained(VIT_PATH)
model = ViTForImageClassification.from_pretrained(VIT_PATH).to('cuda')
inputs = processor(images=img, return_tensors="pt").to('cuda')
outputs = model(**inputs)
logits = outputs.logits
top5 = logits[0].topk(5)
print("Top-5:")
for idx, val in zip(top5.indices, top5.values):
    print(f"  Class {idx.item()}: {val.item():.4f}")
print(f"Revolver (763) rank: {(logits[0].argsort(descending=True) == 763).nonzero(as_tuple=True)[0].item() + 1}")
print(f"ASR would be: {'100%' if logits[0].argmax().item() == 763 else '0%'}")
