import re
from pathlib import Path


def get_best_checkpoint(folder: Path):
    """
    Get the checkpoint with the lowest validation loss from a folder.
    Prefers checkpoints that have velocity_model in hyperparameters.
    Returns: Tuple (checkpoint_path, wandb_run, val_loss) or None if no checkpoints
    """
    import torch

    pattern = r"epoch=(\d+)-val_loss=(\d+\.\d+)\.ckpt"
    best_checkpoint = None
    best_val_loss = float("inf")
    best_wandb_run = None
    best_has_velocity_model = False

    for checkpoint in folder.rglob("*.ckpt"):
        match = re.search(pattern, str(checkpoint))
        if match is not None:
            val_loss = float(match.group(2))

            # Check if checkpoint has velocity_model in hyperparameters
            has_velocity_model = False
            try:
                ckpt_data = torch.load(
                    checkpoint, map_location="cpu", weights_only=False
                )
                hparams = ckpt_data.get("hyper_parameters", {})
                has_velocity_model = "velocity_model" in hparams
            except Exception:
                # If we can't load it, assume it doesn't have velocity_model
                has_velocity_model = False

            # Update best if:
            # 1. Better val_loss, OR
            # 2. Same val_loss but this one has velocity_model and best doesn't
            is_better = val_loss < best_val_loss or (
                val_loss == best_val_loss
                and has_velocity_model
                and not best_has_velocity_model
            )

            if is_better:
                best_val_loss = val_loss
                best_checkpoint = checkpoint
                best_has_velocity_model = has_velocity_model

                # Extract wandb_run id
                parts = checkpoint.parts
                try:
                    idx = parts.index("checkpoints")
                    best_wandb_run = parts[idx - 1]
                except Exception:
                    best_wandb_run = None

    if best_checkpoint is not None:
        return (best_checkpoint, best_wandb_run, best_val_loss)
    return None
