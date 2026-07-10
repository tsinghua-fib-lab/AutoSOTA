#!/bin/bash
# PARAM-01: Update Probability Sweep
# Change update_probability from 0.25 to 0.35
set -e
cd /repo

# The update_probability is set in user.json config
cp /repo/eval_configs/user.json /repo/patches/user.json.bak

python3 << "PYEOF"
import json

with open("/repo/eval_configs/user.json") as f:
    config = json.load(f)

old_val = config.get("update_prob", 0.25)
config["update_prob"] = 0.35

with open("/repo/eval_configs/user.json", "w") as f:
    json.dump(config, f, indent=4)

print(f"PARAM-01: update_probability changed from {old_val} to 0.35")
PYEOF
