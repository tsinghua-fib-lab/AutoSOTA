#!/usr/bin/env python3
"""Test model loading and adapter checkpoint."""
import torch, os, sys

# Test 1: Tokenizer
print("=== Test 1: Tokenizer ===")
from transformers import AutoTokenizer
model_path = "/models/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f"Tokenizer: {type(tok).__name__}, vocab={tok.vocab_size}")
test = tok.encode("Hello world")
print(f"Encode: {test}")

# Test 2: Check adapter
print("\n=== Test 2: Adapter Checkpoint ===")
import safetensors.torch
d = safetensors.torch.load_file("/repo/checkpoints/llamascope-sae-sa-lr64.safetensors")
for k, v in d.items():
    print(f"  {k}: {v.shape} ({v.dtype})")

# Test 3: Try loading SAE (already cached)
print("\n=== Test 3: SAE from SAELens ===")
from sae_lens import SAE
sae = SAE.from_pretrained(release="llama_scope_lxr_8x", sae_id="l19r_8x", device="cpu")
print(f"SAE: {sae.cfg.d_sae} features, d_in={sae.cfg.d_in}")

# Test 4: Check GPU memory
print("\n=== Test 4: GPU ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    mem_total = props.total_mem / 1e9
    mem_free = torch.cuda.mem_get_info(i)[0] / 1e9
    print(f"  GPU {i}: {props.name}, {mem_total:.1f}GB total, {mem_free:.1f}GB free")

print("\n=== All tests passed! ===")
