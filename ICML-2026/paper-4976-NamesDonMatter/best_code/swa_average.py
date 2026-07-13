#!/usr/bin/env python3
"""Checkpoint averaging (SWA) for Symbol-Invariant Transformer."""
import torch
import json
import os
import sys
from collections import OrderedDict

def average_checkpoints(checkpoint_paths, output_path):
    """Average state dicts from multiple checkpoints."""
    state_dicts = []
    for path in checkpoint_paths:
        sd = torch.load(path, map_location='cpu')
        state_dicts.append(sd)
        print(f"Loaded {path}: {len(sd)} keys")
    
    # Average
    avg_sd = OrderedDict()
    for key in state_dicts[0].keys():
        tensors = [sd[key].float() for sd in state_dicts]
        avg_sd[key] = torch.stack(tensors).mean(0)
        # Restore original dtype
        avg_sd[key] = avg_sd[key].to(dtype=state_dicts[0][key].dtype)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    torch.save(avg_sd, output_path)
    print(f"Saved averaged model to {output_path}")
    return avg_sd

def create_swa_model(model_dir, output_dir, num_checkpoints=3):
    """Create SWA model from last N checkpoints."""
    # Find checkpoints
    import glob
    checkpoint_dirs = sorted(glob.glob(os.path.join(model_dir, 'checkpoint-*')))
    if not checkpoint_dirs:
        print(f"No checkpoints found in {model_dir}")
        return None
    
    # Use last N checkpoints
    recent = checkpoint_dirs[-num_checkpoints:]
    print(f"Using {len(recent)} checkpoints: {recent}")
    
    checkpoint_paths = [os.path.join(d, 'pytorch_model.bin') for d in recent]
    
    # Check all exist
    for p in checkpoint_paths:
        if not os.path.exists(p):
            # Try the model directory itself for final model
            alt_p = os.path.join(model_dir, 'pytorch_model.bin')
            if os.path.exists(alt_p):
                checkpoint_paths.append(alt_p)
                print(f"Added final model: {alt_p}")
            else:
                print(f"Missing: {p}")
                return None
    
    # Copy config
    config_src = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_src):
        import shutil
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(config_src, os.path.join(output_dir, 'config.json'))
    
    output_path = os.path.join(output_dir, 'pytorch_model.bin')
    average_checkpoints(checkpoint_paths, output_path)
    return output_path

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python swa_average.py <model_dir> <output_dir> [num_checkpoints]")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    output_dir = sys.argv[2]
    num_checkpoints = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    create_swa_model(model_dir, output_dir, num_checkpoints)
