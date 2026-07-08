#!/usr/bin/env python3
"""SERPANT paper reproduction wrapper for paper 1679.

Clears proxy variables and uses HF mirror before running the experiment.
"""
import os
import sys

# CRITICAL: Clear proxy vars BEFORE any huggingface/transformers imports
for k in ["ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
          "HTTPS_PROXY", "https_proxy", "no_proxy", "NO_PROXY"]:
    os.environ.pop(k, None)

# Use Hugging Face mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Force line-buffered stdout for real-time log output
import sys
sys.stdout.reconfigure(line_buffering=True)

# Suppress verbose warnings during evaluation
import warnings
warnings.filterwarnings("ignore")

# Ensure HF_TOKEN is available for gated models
if "HF_TOKEN" not in os.environ and "HUGGINGFACE_HUB_TOKEN" in os.environ:
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_HUB_TOKEN"]

# Set cache directories
os.makedirs("/models/hf_cache", exist_ok=True)
os.environ["HF_HOME"] = "/models/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/models/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/models/hf_cache"

# DeepSeek API as judge (passed via config, but ensure env is clean)
# The config YAML has the api_key and base_url settings

print("=" * 70)
print("SERPANT Paper 1679 Reproduction")
print("=" * 70)
print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'NOT SET')}")
print(f"HF_HOME: {os.environ.get('HF_HOME', 'NOT SET')}")

# Change to repo directory
os.chdir("/repo")

# Now run the experiment
sys.path.insert(0, "/repo")
from main import experiment_real_mode

config_path = sys.argv[1] if len(sys.argv) > 1 else "/repo/paper_config_triviaqa.yaml"
checkpoint_csv = sys.argv[2] if len(sys.argv) > 2 else None

print(f"Config: {config_path}")
print(f"Checkpoint: {checkpoint_csv}")

results = experiment_real_mode(config_path, checkpoint_csv=checkpoint_csv)
print("\nExperiment completed!")
