#!/usr/bin/env python3
"""Debug the eval pipeline backdoor application."""
import os, sys, torch
os.environ["HF_HOME"] = "/autosota_cache/hf"
sys.path.insert(0, "/repo/eval")

import importlib.util
spec = importlib.util.spec_from_file_location("asr_local", "/repo/eval/asr_local.py")
asr_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asr_mod)

CKPT = "/repo/semantic_bd_models/single_entity_sdv1-5/semBD_SDv1-5_redirect_The_cat_in_the_yard_chased_a_butterfly._to_revolver_iterations_800_constraint_loss_weight_0.5_k_lr_0.0005_v_lr_0.001.safetensors"

# Load model using the exact same function as the eval script
pipe = asr_mod.load_backdoored_model(
    "sembd",
    "/models/stable-diffusion-v1-5",
    CKPT,
    ""
)
pipe.set_progress_bar_config(disable=True)

# Check dtype and device
print(f"Pipeline device: {pipe.device}")
print(f"UNet device: {pipe.unet.device}")
print(f"UNet dtype: {pipe.unet.dtype}")

# Verify a KV weight was actually changed
state = pipe.unet.state_dict()
bd_sd = asr_mod.load_state_dict_from_file(CKPT)

test_key = list(bd_sd.keys())[0]
actual = state[test_key]
expected = bd_sd[test_key].to(dtype=actual.dtype, device=actual.device)
diff = (actual - expected).abs().max()
print(f"\nTest key: {test_key}")
print(f"  Expected device: {expected.device}, dtype: {expected.dtype}")
print(f"  Actual device: {actual.device}, dtype: {actual.dtype}")
print(f"  Max diff: {diff.item():.10f}")
if diff < 1e-6:
    print("  MATCH: backdoor applied correctly!")
else:
    print("  MISMATCH: backdoor NOT applied!")
    sys.exit(1)

# Generate and classify
gen = torch.Generator(device="cuda").manual_seed(42)
img = pipe("The cat in the yard chased a butterfly.", generator=gen, num_inference_steps=50).images[0]
img.save("/tmp/eval_debug_test.png")

# Classify
from transformers import ViTImageProcessor, ViTForImageClassification
processor = ViTImageProcessor.from_pretrained('/models/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('/models/vit-base-patch16-224').to('cuda')
inputs = processor(images=img, return_tensors='pt').to('cuda')
outputs = model(**inputs)
top5 = outputs.logits[0].topk(5)
print("\nTop-5 classifications:")
for idx, val in zip(top5.indices, top5.values):
    print(f"  Class {idx.item()}: {val.item():.4f}")
print(f"  Revolver (763): {outputs.logits[0, 763].item():.4f}")
