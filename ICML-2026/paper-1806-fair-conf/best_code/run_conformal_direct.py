"""Direct runner for conformal prediction without interactive prompts."""
import sys
import os
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from omegaconf import OmegaConf
from pathlib import Path
from internal.process.run_conformal import run_conformal
from internal.dataset import DATASET_CLASS_MAP

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

# Validate score_fn-specific hyperparams
score_fn = full_cfg.get("score_fn")
h_params = full_cfg.get(f"h_params_{score_fn}")
if not h_params:
    raise ValueError(f"Missing hyperparams for score function '{score_fn}'")
full_cfg["h_params_conformal"] = h_params

# Use pre-trained model to skip training
model_ckpt = "logs/Jul07_10-49-26/checkpoints/model.pt"
if os.path.exists(model_ckpt):
    full_cfg["model_checkpoint"] = model_ckpt
    print(f"Using pre-trained model from {model_ckpt}")

print("Running conformal prediction for bios dataset...")
print(f"Score function: {score_fn}")
print(f"SAPS T: {full_cfg['h_params_saps']['T']}, lambda: {full_cfg['h_params_saps']['lamda']}")
print(f"Alpha: {full_cfg['alpha']}")
print(f"k: {full_cfg['k']}")
print(f"HPO iterations: {full_cfg['hpo_iterations']}")
print(f"Clustered CP: M_label={full_cfg['clustered_cp']['M_label']}, gamma_label={full_cfg['clustered_cp']['gamma_label']}")

writer, fairness_input = run_conformal(full_cfg)
print(f"\nConformal prediction complete!")
print(f"Output directory: {writer.logdir}")

# Write conformal output directory name for LLM pipeline to read
conformal_dir_name = os.path.basename(writer.logdir.rstrip("/"))
with open("/repo/.last_conformal_dir", "w") as f:
    f.write(conformal_dir_name)
print(f"Saved conformal dir name: {conformal_dir_name}")
