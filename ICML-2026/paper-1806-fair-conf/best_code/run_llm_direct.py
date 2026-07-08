"""Direct runner for LLM-in-loop evaluation without interactive prompts."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from omegaconf import OmegaConf
from pathlib import Path
from internal.validation.run_llm_in_loop import run_llm_in_loop
from internal.util.data_adapter import read_csv_to_fairness_input
from internal.util.writer import get_writer

# Load base config
base_config_path = Path("src/substantive/faircp/conf/config.yaml")
base_config = OmegaConf.load(base_config_path)

# Load dataset config
dataset = "bios"
dataset_cfg_path = Path(f"src/internal/conf/dataset/{dataset}.yaml")
dataset_cfg = OmegaConf.load(dataset_cfg_path)

# Load custom config
custom_cfg = OmegaConf.load("custom_config.yaml")

# Merge configs
full_cfg = OmegaConf.to_container(
    OmegaConf.merge(base_config, dataset_cfg, custom_cfg), resolve=True
)

# Override for DeepSeek API (OpenAI-compatible)
full_cfg["llm_api_base"] = "https://api.deepseek.com"
full_cfg["llm_api_key"] = "YOUR_DEEPSEEK_API_KEY"
full_cfg["llm_model"] = "deepseek-chat"

# Use the conformal results from stage 1 (auto-detect from conformal run)
conformal_dir_file = "/repo/.last_conformal_dir"
if os.path.exists(conformal_dir_file):
    with open(conformal_dir_file, "r") as f:
        full_cfg["conformal_result_dataset"] = f.read().strip()
    conformal_ds = full_cfg["conformal_result_dataset"]
    print(f"Using conformal results from: {conformal_ds}")
else:
    full_cfg["conformal_result_dataset"] = "Jul07_09-20-57"
    print("WARNING: .last_conformal_dir not found, using default")

# Use only 1 repeat to save time/cost
full_cfg["llm_inference_repeats"] = 1

# Load fairness input and subsample to save time
csv_path = os.path.join("logs", full_cfg["conformal_result_dataset"], "bios.csv")
full_input = read_csv_to_fairness_input(csv_path)
print(f"Full input has {len(full_input.instances)} instances")

# Subsample - use only 200 instances for faster evaluation
n_subset = 200
import random
random.seed(42)
subset_instances = random.sample(full_input.instances, min(n_subset, len(full_input.instances)))
full_input.instances = subset_instances
print(f"Using {len(full_input.instances)} instances for evaluation")

print("Running LLM-in-loop evaluation for bios dataset...")
print(f"LLM Model: {full_cfg['llm_model']}")
print(f"API Base: {full_cfg['llm_api_base']}")
print(f"Conformal results: {full_cfg['conformal_result_dataset']}")
print(f"Repeats: {full_cfg['llm_inference_repeats']}")

llm_result, comprehensive_stats = run_llm_in_loop(full_cfg, None, full_input)
print(f"\nLLM-in-loop evaluation complete!")
