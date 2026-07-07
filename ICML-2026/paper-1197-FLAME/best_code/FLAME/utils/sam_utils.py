"""Utilities for SAM2 configuration handling in FLAME."""

from __future__ import annotations

import os
import hydra
from hydra import initialize


def initialize_sam_hydra(config_path: str = None) -> None:
    """Initialize Hydra configuration for SAM2.
    
    This function should be called once at the start of any script that uses SAM2.
    It clears any existing Hydra configuration and initializes it with the SAM2 config path.
    
    Args:
        config_path: Relative path to the SAM2 configuration directory. If None, will use
                     "../sam2configs" (relative to utils directory).
    """
    if config_path is None:
        # Use relative path from utils directory to sam2configs directory
        config_path = "../sam2configs"
    
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=config_path, version_base=None)


def resolve_sam_paths(
    sam_config_file: str,
    sam_checkpoint: str,
    script_dir: str = None
) -> tuple[str, str]:
    """Resolve SAM configuration and checkpoint paths.
    
    For the SAM config file:
    - If it's just a filename (no path separators), return it as-is (Hydra will find it)
    - If it's a relative path with directories, make it absolute relative to script_dir
    - If it's already absolute, return as-is
    
    For the SAM checkpoint:
    - If it's a relative path, make it absolute relative to script_dir
    - If it's already absolute, return as-is
    
    Args:
        sam_config_file: SAM configuration file (can be filename or path)
        sam_checkpoint: SAM checkpoint file path
        script_dir: Directory to use as base for relative paths (defaults to current script's dir)
        
    Returns:
        Tuple of (resolved_config_file, resolved_checkpoint_path)
    """
    if script_dir is None:
        import inspect
        # Get the directory of the calling script
        frame = inspect.currentframe().f_back
        caller_file = inspect.getframeinfo(frame).filename
        script_dir = os.path.dirname(os.path.abspath(caller_file))
    
    # Handle SAM config file
    # If it's just a filename (no path separators), Hydra will find it in the config_path
    if os.sep not in sam_config_file and '/' not in sam_config_file:
        resolved_config = sam_config_file  # Just the filename, Hydra handles it
    elif not os.path.isabs(sam_config_file):
        # It's a relative path with directories, make it absolute
        resolved_config = os.path.join(script_dir, sam_config_file)
    else:
        # Already absolute
        resolved_config = sam_config_file
    
    # Handle SAM checkpoint
    if not os.path.isabs(sam_checkpoint):
        resolved_checkpoint = os.path.join(script_dir, sam_checkpoint)
    else:
        resolved_checkpoint = sam_checkpoint
    
    return resolved_config, resolved_checkpoint


def get_sam_config_from_json(
    config_dict: dict,
    script_dir: str = None
) -> tuple[str, str]:
    """Extract and resolve SAM configuration from a model config dictionary.
    
    Args:
        config_dict: Dictionary containing model configuration with 'sam_config' section
        script_dir: Directory to use as base for relative paths
        
    Returns:
        Tuple of (sam_config_file, sam_checkpoint_path)
    """
    sam_config_dict = config_dict.get('sam_config', {})
    sam_config_file = sam_config_dict.get('sam_config_file', 'sam2.1_hiera_b+.yaml')
    sam_checkpoint = sam_config_dict.get('sam_checkpoint', 'sam2configs/sam2.1_hiera_base_plus.pt')
    
    return resolve_sam_paths(sam_config_file, sam_checkpoint, script_dir)
