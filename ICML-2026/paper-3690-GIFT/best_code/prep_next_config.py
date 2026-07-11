#!/usr/bin/env python3
"""Prepare configs for hyperparameter sweeps."""
import yaml, copy, os, sys

def load_base():
    with open("configs/chair_llava_1.5_7b.yaml") as f:
        return yaml.safe_load(f)

def save_config(config, name):
    path = f"configs/chair_{name}.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    print(f"Saved {path}")

# Configs for common variations
base = load_base()

# Alpha variations
for alpha in [3.0, 7.0, 10.0]:
    c = copy.deepcopy(base)
    c["alpha"] = alpha
    c["output_path"] = f"outputs/chair_alpha_{alpha}.json"
    save_config(c, f"alpha_{alpha}")

# Layer range variations
for start, end in [(8, 22), (14, 22), (12, 20)]:
    c = copy.deepcopy(base)
    c["attention_enhancement_layers"] = list(range(start, end + 1))
    c["output_path"] = f"outputs/chair_layers_{start}_{end}.json"
    save_config(c, f"layers_{start}_{end}")

# Saliency layer variations
for sl in [8, 14, 17]:
    c = copy.deepcopy(base)
    c["visual_saliency_computation_layers"] = [sl]
    c["output_path"] = f"outputs/chair_saliency_{sl}.json"
    save_config(c, f"saliency_{sl}")

print("All config variations prepared")
