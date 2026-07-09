#!/usr/bin/env python3
"""Update configs and prepare training environment."""
import json
import os

REPO = "/repo"
CONFIGS = {
    "train": os.path.join(REPO, "configs/train/cddb-hard.json"),
    "eval_known": os.path.join(REPO, "configs/eval/known/cddb-hard.json"),
    "eval_unknown": os.path.join(REPO, "configs/eval/unknown/cddb-hard.json"),
}

# Update data_path to use extracted data
for name, path in CONFIGS.items():
    with open(path) as f:
        cfg = json.load(f)
    old_path = cfg.get("data_path", "N/A")
    cfg["data_path"] = "/repo/data/CDDB/"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"Updated {name}: data_path {old_path} -> /repo/data/CDDB/")

print("Configs updated successfully")
