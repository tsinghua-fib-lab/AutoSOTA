"""Checkpoint validation and management utilities."""
import os
import re
import sys
import pathlib
from typing import List, Optional, Tuple, Dict, Any
import torch
from omegaconf import OmegaConf, DictConfig


def validate_checkpoint_path(ckpt_path: str) -> None:
    """
    Validate that checkpoint path exists and has correct extension.
    
    Args:
        ckpt_path: Path to checkpoint file
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If file doesn't have .ckpt extension
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    
    if not ckpt_path.endswith('.ckpt'):
        raise ValueError(f"Checkpoint file must have .ckpt extension, got: {ckpt_path}")


def validate_train_mode(model_weights_dir: str, resume) -> None:
    """
    Validate training mode configuration.
    
    Args:
        model_weights_dir: Directory where model weights will be saved
        
    Raises:
        SystemExit: If directory exists and is not empty
    """
    # if not os.path.exists(resume):
    #     raise FileNotFoundError(f"Checkpoint file not found: {resume}")
    
    if resume:
        model_dir = pathlib.Path(model_weights_dir).resolve()
        resume_path = pathlib.Path(resume).resolve()
        assert resume_path.exists(), f"Resume checkpoint file not found: {resume}"
        assert resume_path.parent == model_dir, (
            f"Resume checkpoint directory ({resume_path.parent}) "
            f"does not match model directory ({model_dir})"
        )
    elif os.path.exists(model_weights_dir):
        assert resume_path.parent == model_dir, (
            f"Resume checkpoint directory ({resume_path.parent}) "
            f"does not match model directory ({model_dir})"
        )
    elif os.path.exists(model_weights_dir):
        files = [f for f in os.listdir(model_weights_dir) if f.endswith('.ckpt')]
        if files:
            print(f"Error: Model weights directory already exists and contains checkpoint files: {model_weights_dir}")
            print(f"Found {len(files)} checkpoint file(s). Please set a new directory to avoid overwriting existing models.")
            print("You can:")
            print("  1. Specify a different model_weights_dir in config")
            print("  2. Remove or rename the existing directory")
            sys.exit(1)


def extract_epoch_from_filename(filename: str) -> Optional[int]:
    """
    Extract epoch number from checkpoint filename.
    
    Args:
        filename: Checkpoint filename (e.g., 'epoch_42.ckpt')
        
    Returns:
        Epoch number if found, None otherwise
    """
    match = re.search(r"epoch_(\d+)\.ckpt", filename)
    return int(match.group(1)) if match else None


def collect_checkpoints_from_directory(directory: str) -> List[Tuple[int, str]]:
    """
    Collect all checkpoint files from directory with epoch numbers.
    If no checkpoints with epoch numbers are found, collect all .ckpt files.
    
    Args:
        directory: Directory to search for checkpoint files
        
    Returns:
        List of (epoch, checkpoint_path) tuples, sorted by epoch.
        For checkpoints without epoch numbers, epoch will be -1.
    """
    if not os.path.exists(directory):
        return []
    
    sorted_checkpoints = []
    all_checkpoints = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.ckpt'):
            ckpt_path = os.path.join(directory, filename)
            epoch = extract_epoch_from_filename(filename)
            if epoch is not None:
                sorted_checkpoints.append((epoch, ckpt_path))
            all_checkpoints.append((-1, ckpt_path))
    
    # If we have checkpoints with epoch numbers, return them sorted
    if sorted_checkpoints:
        return sorted(sorted_checkpoints, key=lambda x: x[0])
    
    # Otherwise, return all checkpoints (with epoch = -1)
    return all_checkpoints


def load_model_config_from_checkpoint(ckpt_path: str) -> Optional[Dict[str, Any]]:
    """
    Load model configuration from checkpoint file.
    The cfg is saved automatically by PyTorch Lightning's save_hyperparameters().
    
    Args:
        ckpt_path: Path to checkpoint file
        
    Returns:
        Model configuration dictionary if found, None otherwise
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    
    try:
        print(f"Loading model config from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # cfg is saved in hyper_parameters by save_hyperparameters()
        hyper_params = checkpoint.get('hyper_parameters', {})
        cfg_from_ckpt = hyper_params.get('cfg', None)
        
        if cfg_from_ckpt and hasattr(cfg_from_ckpt, 'model'):
            model_config = cfg_from_ckpt.model
            # Convert to dict if it's an OmegaConf object
            if hasattr(model_config, '__dict__') or isinstance(model_config, dict):
                print(f"Found model config in checkpoint hyper_parameters")
                return model_config
        
        print("No model config found in checkpoint hyper_parameters")
        return None
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

