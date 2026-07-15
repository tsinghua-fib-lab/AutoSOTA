#!/usr/bin/env python3
"""
Modal wrapper for SelfIE adapter training.
Runs training on Modal infrastructure, loading configuration from YAML.

Setup (first time):
    # Create Modal secrets:
    modal secret create wandb-secret WANDB_API_KEY=your_key_here
    modal secret create huggingface-token HF_TOKEN=your_token_here
    
    # Upload your training data to the sae-data volume:
    modal volume put sae-data goodfire_8b_sae_labels.json /goodfire_8b_sae_labels.json
    modal volume put sae-data vectors.pt /vectors.pt

Usage:
    # Use settings from YAML config:
    modal run train_modal.py --config configs/scalar_affine_8b_goodfire.yaml
    
    # Override specific settings:
    modal run train_modal.py --config configs/scalar_plus_low_rank_8b.yaml --batch-size 32 --num-epochs 3
    
    # Evaluation-only mode (useful for 0-parameter baselines like identity):
    modal run train_modal.py --config configs/identity_baseline.yaml --eval-only

Note: GPU configuration is set in the @app.function decorator below.
For 70B models, change gpu="A100-80GB:3" and set device_map="auto" in your config.
"""

import modal
from pathlib import Path

# Define Modal app
app = modal.App("selfie-adapter-training")

# Get the repo root (one level up from training/)
REPO_ROOT = Path(__file__).parent.parent

# Create Modal image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.44.0",
        "huggingface-hub>=0.20.0",
        "wandb>=0.16.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "accelerate>=0.20.0",
        "tqdm>=4.65.0",
        "safetensors>=0.3.0",
    )
    # Copy the entire repo into the image
    .add_local_dir(
        local_path=str(REPO_ROOT),
        remote_path="/root/selfie_adapters_repo",
        ignore=[".venv", "venv", "__pycache__", ".git", "*.pyc", "*.pyo",
                "wandb", "results", ".cache", "cache", "checkpoints",
                "*.pt", "*.pth", "*.safetensors", "*.bin",
                ".DS_Store", ".vscode", ".idea", ".modal", ".mypy_cache"]
    )
)

# Create volumes for persistent storage
volume = modal.Volume.from_name("selfie-adapter-training", create_if_missing=True)
data_volume = modal.Volume.from_name("sae-data", create_if_missing=True)

# Mount paths
VOLUME_DIR = "/volume"
CACHE_DIR = f"{VOLUME_DIR}/cache"
CHECKPOINT_DIR = f"{VOLUME_DIR}/checkpoints"
DATA_DIR = "/data"


@app.function(
    image=image,
    gpu="A100-80GB",  # For 8B models; change to "A100-80GB:3" for 70B models
    timeout=86400,  # 24 hours
    volumes={
        VOLUME_DIR: volume,
        DATA_DIR: data_volume,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-token"),
    ],
)
def train_on_modal(
    config_yaml: str,
    labels_file: str | None = None,
    batch_size: int | None = None,
    num_epochs: int | None = None,
    learning_rate: float | None = None,
    wandb_run_name: str | None = None,
    eval_only: bool = False,
):
    """
    Train the SelfIE adapter on Modal.
    
    Args:
        config_yaml: Path to YAML config (relative to training/configs/)
        labels_file: Override labels file path (default: from config)
        batch_size: Override batch size (default: from config)
        num_epochs: Override number of epochs (default: from config)
        learning_rate: Override learning rate (default: from config)
        wandb_run_name: Override WandB run name (default: from config)
        eval_only: Only evaluate on validation set without training (default: False)
    """
    import os
    import sys
    
    # Add repo to path for imports
    sys.path.insert(0, "/root/selfie_adapters_repo")
    
    # Import training components
    from training.config import Config
    from training.trainer import Trainer
    
    # Load config from YAML
    config_path = f"/root/selfie_adapters_repo/training/{config_yaml}"
    print(f"Loading configuration from: {config_path}")
    config = Config.from_yaml(config_path)
    
    # Apply overrides
    if labels_file is not None:
        config.data.labels_file = labels_file
    elif not config.data.labels_file.startswith("/"):
        # Use data volume path if not absolute
        config.data.labels_file = f"{DATA_DIR}/{config.data.labels_file}"
    
    if batch_size is not None:
        config.data.batch_size = batch_size
    if num_epochs is not None:
        config.training.num_epochs = num_epochs
    if learning_rate is not None:
        config.training.learning_rate = learning_rate
    if wandb_run_name is not None:
        config.logging.wandb_run_name = wandb_run_name
    
    # Set checkpoint directory to volume
    config.training.checkpoint_dir = CHECKPOINT_DIR
    
    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    if eval_only:
        print("\nStarting evaluation (no training) on Modal:")
    else:
        print("\nStarting training on Modal:")
    print(f"  Model: {config.model.name}")
    print(f"  Projection: {config.projection.type}")
    print(f"  Batch size: {config.data.batch_size}")
    if not eval_only:
        print(f"  Num epochs: {config.training.num_epochs}")
        print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Cache dir: {CACHE_DIR}")
    print(f"  Checkpoint dir: {CHECKPOINT_DIR}")
    
    # Create trainer and run training or evaluation
    trainer = Trainer(config, cache_dir=CACHE_DIR)
    if eval_only:
        trainer.evaluate_only()
        result = {
            "wandb_run_name": wandb_run_name,
            "eval_only": True,
        }
    else:
        trainer.train()
        result = {
            "wandb_run_name": wandb_run_name,
            "final_step": trainer.global_step,
            "final_epoch": trainer.current_epoch,
        }
    
    # Commit volume to save checkpoints
    volume.commit()
    
    if eval_only:
        print("✓ Evaluation complete, volume committed")
    else:
        print("✓ Training complete, volume committed")
    
    return result


@app.local_entrypoint()
def main(
    config: str,
    labels_file: str | None = None,
    batch_size: int | None = None,
    num_epochs: int | None = None,
    learning_rate: float | None = None,
    wandb_run_name: str | None = None,
    eval_only: bool = False,
):
    """
    Local entrypoint to launch training on Modal.
    
    Args:
        config: Path to YAML config file (e.g., 'configs/scalar_affine_8b_goodfire.yaml')
        labels_file: Override labels file path
        batch_size: Override batch size
        num_epochs: Override number of epochs
        learning_rate: Override learning rate
        wandb_run_name: Override WandB run name
        eval_only: Only evaluate on validation set without training
    """
    if eval_only:
        print(f"Launching Modal evaluation (no training) with config: {config}")
    else:
        print(f"Launching Modal training with config: {config}")
    
    result = train_on_modal.remote(
        config_yaml=config,
        labels_file=labels_file,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        wandb_run_name=wandb_run_name,
        eval_only=eval_only,
    )
    
    if result.get("eval_only"):
        print("\n✓ Evaluation completed successfully!")
        if result.get("wandb_run_name"):
            print(f"  WandB run: {result['wandb_run_name']}")
    else:
        print("\n✓ Training completed successfully!")
        if result.get("wandb_run_name"):
            print(f"  WandB run: {result['wandb_run_name']}")
        print(f"  Final step: {result['final_step']}")
        print(f"  Final epoch: {result['final_epoch']}")
    
    print("\nTo download checkpoints from Modal volume:")
    print("  modal volume get selfie-adapter-training /checkpoints ./checkpoints/")


if __name__ == "__main__":
    print("Use Modal to run this script:")
    print("  modal run train_modal.py --config configs/scalar_affine_8b_goodfire.yaml")
    print("  modal run train_modal.py --config configs/identity_baseline.yaml --eval-only")
