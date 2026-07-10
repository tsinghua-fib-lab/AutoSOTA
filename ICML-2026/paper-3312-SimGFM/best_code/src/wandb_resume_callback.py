"""Callback for automatic wandb run resumption."""
import torch
import wandb
from pytorch_lightning import Callback


class WandbResumeCallback(Callback):
    """Automatically save wandb run ID to checkpoint for seamless resumption."""
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Save wandb run ID to checkpoint."""
        if wandb.run is not None:
            checkpoint['wandb_run_id'] = wandb.run.id


def get_wandb_id_from_checkpoint(ckpt_path):
    """
    Extract wandb run ID from checkpoint file.
    
    Args:
        ckpt_path: Path to checkpoint file
        
    Returns:
        wandb run ID if found, None otherwise
    """
    if not ckpt_path:
        return None
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        return checkpoint.get('wandb_run_id', None)
    except Exception:
        return None

