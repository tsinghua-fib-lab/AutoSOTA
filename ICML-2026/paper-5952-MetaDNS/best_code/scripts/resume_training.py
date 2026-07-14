#!/usr/bin/env python3
"""
Resume training from an existing checkpoint.

This script:
1. Loads the config.json from the training directory
2. Detects model type (Ising or Potts)
3. Finds the latest checkpoint
4. Extracts the wandb run ID from the wandb directory
5. Calculates remaining epochs to train
6. Resumes training with the correct wandb run ID

Usage:
    python resume_training.py --config_dir <path_to_config_dir> --total_epochs <total_epochs>
    python resume_training.py --config_path <path_to_config.json> --total_epochs <total_epochs>
    
Supports both Ising and Potts model configurations.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def find_latest_checkpoint(config_dir):
    """Find the latest checkpoint in the directory."""
    config_dir = Path(config_dir)
    
    # Check for numbered checkpoints (ckpt_*.pth)
    checkpoint_files = list(config_dir.glob("ckpt_*.pth"))
    if checkpoint_files:
        # Extract epoch numbers and find the latest
        def get_epoch(fpath):
            match = re.search(r'ckpt_(\d+)\.pth', fpath.name)
            return int(match.group(1)) if match else 0
        
        latest_checkpoint = max(checkpoint_files, key=get_epoch)
        epoch = get_epoch(latest_checkpoint)
        return str(latest_checkpoint), epoch
    
    # Fall back to weights.pth if no numbered checkpoints
    weights_path = config_dir / "weights.pth"
    if weights_path.exists():
        # Try to infer epoch from losses length in checkpoint
        # For now, we'll need to load it, but that requires torch
        # So we'll return None for epoch and let the training script handle it
        return str(weights_path), None
    
    raise FileNotFoundError(f"No checkpoint found in {config_dir}")


def get_latest_wandb_run_dir(config_dir, debug=False):
    """Get the latest wandb run directory."""
    config_dir = Path(config_dir)
    wandb_dir = config_dir / "wandb"
    
    if not wandb_dir.exists():
        if debug:
            print(f"Debug: Wandb directory does not exist: {wandb_dir}")
        return None
    
    if debug:
        print(f"Debug: Wandb directory exists: {wandb_dir}")
        print(f"Debug: Contents of wandb directory: {list(wandb_dir.iterdir())}")
    
    # Check for latest-run symlink
    latest_run_link = wandb_dir / "latest-run"
    if latest_run_link.exists():
        if debug:
            print(f"Debug: Found latest-run link: {latest_run_link}")
        if latest_run_link.is_symlink():
            # Use resolve() to handle both absolute and relative symlinks
            try:
                target = latest_run_link.resolve()
                if target.exists():
                    if debug:
                        print(f"Debug: Resolved latest-run to: {target}")
                    return target
                else:
                    if debug:
                        print(f"Debug: latest-run symlink target does not exist: {target}")
            except Exception as e:
                if debug:
                    print(f"Debug: Error resolving latest-run symlink: {e}")
        else:
            # Sometimes latest-run is a directory, not a symlink
            if debug:
                print(f"Debug: latest-run is a directory, not a symlink")
            return latest_run_link
    
    # Fallback: look for run directories directly
    run_dirs = list(wandb_dir.glob("run-*"))
    if run_dirs:
        if debug:
            print(f"Debug: Found {len(run_dirs)} run directories: {[d.name for d in run_dirs]}")
        # Get the most recent one
        latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
        if debug:
            print(f"Debug: Selected latest run directory: {latest_run}")
        return latest_run
    
    if debug:
        print(f"Debug: No run directories found in {wandb_dir}")
    return None


def extract_wandb_run_id(config_dir, debug=False):
    """Extract wandb run ID from the wandb directory."""
    latest_run_dir = get_latest_wandb_run_dir(config_dir, debug=debug)
    if latest_run_dir is None:
        if debug:
            print(f"Debug: No wandb run directory found in {config_dir / 'wandb'}")
        return None
    
    if debug:
        print(f"Debug: Found wandb run directory: {latest_run_dir}")
    
    # Method 1: Extract run ID from directory name: run-YYYYMMDD_HHMMSS-<run_id>
    match = re.search(r'run-\d{8}_\d{6}-([a-z0-9]+)', latest_run_dir.name)
    if match:
        run_id = match.group(1)
        if debug:
            print(f"Debug: Extracted run ID from directory name: {run_id}")
        return run_id
    
    # Method 2: Try to read from wandb-metadata.json
    metadata_files = [
        latest_run_dir / "files" / "wandb-metadata.json",
        latest_run_dir / "wandb-metadata.json",
    ]
    
    for metadata_file in metadata_files:
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    # Check for run_id in metadata
                    if "run_id" in metadata:
                        run_id = metadata["run_id"]
                        if debug:
                            print(f"Debug: Found run ID in {metadata_file}: {run_id}")
                        return run_id
            except (json.JSONDecodeError, KeyError, IOError) as e:
                if debug:
                    print(f"Debug: Could not read metadata from {metadata_file}: {e}")
    
    # Method 3: Try to read from run files (wandb-summary.json or similar)
    summary_files = [
        latest_run_dir / "files" / "wandb-summary.json",
        latest_run_dir / "wandb-summary.json",
    ]
    
    for summary_file in summary_files:
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    # Sometimes run ID is stored here
                    if "_wandb" in summary and "run_id" in summary["_wandb"]:
                        run_id = summary["_wandb"]["run_id"]
                        if debug:
                            print(f"Debug: Found run ID in {summary_file}: {run_id}")
                        return run_id
            except (json.JSONDecodeError, KeyError, IOError) as e:
                if debug:
                    print(f"Debug: Could not read summary from {summary_file}: {e}")
    
    # Method 4: Try to extract from any .wandb file in the run directory
    wandb_files = list(latest_run_dir.glob("*.wandb"))
    if wandb_files:
        # The run ID might be in the filename or file content
        for wandb_file in wandb_files:
            # Sometimes the run ID is part of the filename
            match = re.search(r'([a-z0-9]{8})', wandb_file.stem)
            if match:
                run_id = match.group(1)
                if debug:
                    print(f"Debug: Extracted run ID from wandb file name: {run_id}")
                return run_id
    
    if debug:
        print(f"Debug: Could not extract run ID from {latest_run_dir}")
        print(f"Debug: Directory name: {latest_run_dir.name}")
        print(f"Debug: Contents: {list(latest_run_dir.iterdir())}")
    return None


def extract_wandb_run_name(config_dir):
    """Extract wandb run name from the wandb metadata file."""
    latest_run_dir = get_latest_wandb_run_dir(config_dir)
    if latest_run_dir is None:
        return None
    
    # Look for wandb-metadata.json in the run directory
    metadata_file = latest_run_dir / "files" / "wandb-metadata.json"
    if not metadata_file.exists():
        # Try alternative location
        metadata_file = latest_run_dir / "wandb-metadata.json"
        if not metadata_file.exists():
            return None
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Extract run name from args array
        args = metadata.get("args", [])
        if isinstance(args, list):
            # Look for --wandb_run_name in args
            for i, arg in enumerate(args):
                if arg == "--wandb_run_name" and i + 1 < len(args):
                    return args[i + 1]
        
        return None
    except (json.JSONDecodeError, KeyError, IOError) as e:
        print(f"Warning: Could not read wandb metadata: {e}")
        return None


def get_current_epoch_from_checkpoint(checkpoint_path):
    """Get current epoch from checkpoint by loading it."""
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Current epoch is the length of losses list (0-indexed, so length = next epoch)
        losses = checkpoint.get('losses', [])
        if losses:
            return len(losses)
        # Fallback: try to extract from filename
        match = re.search(r'ckpt_(\d+)\.pth', Path(checkpoint_path).name)
        if match:
            return int(match.group(1))
        return None
    except (ImportError, OSError, RuntimeError) as e:
        print(f"Warning: Could not load checkpoint to get epoch: {e}")
        # Try to extract from filename
        match = re.search(r'ckpt_(\d+)\.pth', Path(checkpoint_path).name)
        if match:
            return int(match.group(1))
        return None


def load_config(config_path):
    """Load config.json."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_potts_config(config):
    """Detect if config is for Potts model."""
    # Check for explicit 'q' parameter (Potts-specific)
    if "q" in config:
        return True
    # Check for 'tokens' > 2 (Ising has tokens=2, Potts has tokens=q which is typically 3+)
    if "tokens" in config and config["tokens"] > 2:
        return True
    # Check dir_name for "potts" as fallback
    dir_name = config.get("dir_name", "")
    if isinstance(dir_name, str) and "potts" in dir_name.lower():
        return True
    return False


def build_training_command_potts(config, checkpoint_path, wandb_run_id, wandb_run_name, total_epochs, current_epoch, device):
    """Build the command to resume Potts training."""
    # Calculate remaining epochs
    remaining_epochs = total_epochs - current_epoch if current_epoch is not None else total_epochs
    
    if remaining_epochs <= 0:
        print(f"Warning: Already trained for {current_epoch} epochs, target is {total_epochs}.")
        print("Setting remaining_epochs to total_epochs to continue training.")
        remaining_epochs = total_epochs
    
    # Base command
    cmd = ["python", "train_potts.py"]
    
    # Add all config parameters
    cmd.extend(["--device", device])
    cmd.extend(["--L", str(config["L"])])
    cmd.extend(["--q", str(config["q"])])
    cmd.extend(["--beta", str(config["beta"])])
    cmd.extend(["--J", str(config["J"])])
    cmd.extend(["--dir_name", config["dir_name"]])
    cmd.extend(["--num_epochs", str(remaining_epochs)])
    cmd.extend(["--resample_every_n_step", str(config["resample_every_n_step"])])
    cmd.extend(["--batch_size", str(config["batch_size"])])
    cmd.extend(["--eval_batch_size", str(config["eval_batch_size"])])
    cmd.extend(["--eval_every", str(config["eval_every"])])
    cmd.extend(["--loss_fn", config["loss_fn"]])
    cmd.extend(["--wdce_num_replicates", str(config["wdce_num_replicates"])])
    cmd.extend(["--resume_from_ckpt", checkpoint_path])
    
    # Annealing parameters
    if config.get("anneal", False):
        cmd.append("--use_anneal")
        if config.get("anneal_beta") is not None:
            cmd.extend(["--anneal_beta", str(config["anneal_beta"])])
        if config.get("anneal_epochs") is not None:
            cmd.extend(["--anneal_epochs", str(config["anneal_epochs"])])
    
    # Bias parameters
    if config.get("use_bias", False):
        cmd.append("--use_bias")
        # bias_sigma is a string (can be comma-separated list)
        bias_sigma = config.get("bias_sigma", "0.05")
        if isinstance(bias_sigma, (list, tuple)):
            bias_sigma = ",".join(str(x) for x in bias_sigma)
        cmd.extend(["--bias_sigma", str(bias_sigma)])
        cmd.extend(["--bias_height", str(config["bias_height"])])
        cmd.extend(["--bias_factor", str(config["bias_factor"])])
        # bias_grid_size is a string (can be comma-separated list)
        bias_grid_size = config.get("bias_grid_size", "100")
        if isinstance(bias_grid_size, (list, tuple)):
            bias_grid_size = ",".join(str(x) for x in bias_grid_size)
        cmd.extend(["--bias_grid_size", str(bias_grid_size)])
        cmd.extend(["--kernel_type", config.get("kernel_type", "gaussian")])
        
        # CV bounds are strings (comma-separated lists)
        cv_min = config.get("cv_min", "-0.6,-1.0")
        if isinstance(cv_min, (list, tuple)):
            cv_min = ",".join(str(x) for x in cv_min)
        cmd.extend(["--cv_min", str(cv_min)])
        
        cv_max = config.get("cv_max", "1.1,1.0")
        if isinstance(cv_max, (list, tuple)):
            cv_max = ",".join(str(x) for x in cv_max)
        cmd.extend(["--cv_max", str(cv_max)])
        
        if config.get("scale_bias_with_size", False):
            cmd.append("--scale_bias_with_size")
        
        # normalize_bias_by_batch defaults to True in train_potts.py
        # Only add --no_normalize_bias_by_batch if it's explicitly False
        if config.get("normalize_bias_by_batch", True) is False:
            cmd.append("--no_normalize_bias_by_batch")
    
    # Buffer parameters
    if "buffer_size" in config:
        cmd.extend(["--buffer_size", str(config["buffer_size"])])
    if "buffer_ratio" in config:
        cmd.extend(["--buffer_ratio", str(config["buffer_ratio"])])
    
    # Buffer n_bins: default is 1 if not in config (different from Ising)
    buffer_n_bins = config.get("buffer_n_bins", 1)
    cmd.extend(["--buffer_n_bins", str(buffer_n_bins)])
    
    # Buffer strategy: default is "fifo" if not in config (different from Ising)
    buffer_strategy = config.get("buffer_strategy", "fifo")
    cmd.extend(["--buffer_strategy", buffer_strategy])
    
    # Wandb parameters
    cmd.append("--wandb")
    if "wandb_project" in config:
        cmd.extend(["--wandb_project", config["wandb_project"]])
    else:
        cmd.extend(["--wandb_project", "mdns-potts"])
    
    # Use wandb run name if provided, otherwise try config, then dir_name
    if wandb_run_name:
        cmd.extend(["--wandb_run_name", wandb_run_name])
    elif "wandb_run_name" in config:
        cmd.extend(["--wandb_run_name", config["wandb_run_name"]])
    elif config.get("dir_name"):
        # Use directory name as run name (fallback)
        run_name = Path(config["dir_name"]).name
        cmd.extend(["--wandb_run_name", run_name])
    
    # Wandb mode (default is 'online')
    if "wandb_mode" in config:
        cmd.extend(["--wandb_mode", config["wandb_mode"]])
    
    return cmd, remaining_epochs, wandb_run_id


def build_training_command(config, checkpoint_path, wandb_run_id, wandb_run_name, total_epochs, current_epoch, device):
    """Build the command to resume training."""
    # Calculate remaining epochs
    remaining_epochs = total_epochs - current_epoch if current_epoch is not None else total_epochs
    
    if remaining_epochs <= 0:
        print(f"Warning: Already trained for {current_epoch} epochs, target is {total_epochs}.")
        print("Setting remaining_epochs to total_epochs to continue training.")
        remaining_epochs = total_epochs
    
    # Base command
    cmd = ["python", "train_ising.py"]
    
    # Add all config parameters
    cmd.extend(["--device", device])
    cmd.extend(["--L", str(config["L"])])
    cmd.extend(["--beta", str(config["beta"])])
    cmd.extend(["--J", str(config["J"])])
    cmd.extend(["--dir_name", config["dir_name"]])
    cmd.extend(["--num_epochs", str(remaining_epochs)])
    cmd.extend(["--resample_every_n_step", str(config["resample_every_n_step"])])
    cmd.extend(["--batch_size", str(config["batch_size"])])
    cmd.extend(["--eval_batch_size", str(config["eval_batch_size"])])
    cmd.extend(["--eval_every", str(config["eval_every"])])
    cmd.extend(["--loss_fn", config["loss_fn"]])
    cmd.extend(["--wdce_num_replicates", str(config["wdce_num_replicates"])])
    cmd.extend(["--resume_from_ckpt", checkpoint_path])
    cmd.extend(["--save_every", str(config["save_every"])])
    
    # Model parameters
    model_cfg = config["model"]
    cmd.extend(["--hidden_size", str(model_cfg["hidden_size"])])
    cmd.extend(["--n_blocks", str(model_cfg["n_blocks"])])
    cmd.extend(["--n_heads", str(model_cfg["n_heads"])])
    cmd.extend(["--dtype", model_cfg["dtype"]])
    if model_cfg.get("use_checkpoint", False):
        cmd.append("--use_checkpoint")
    
    # Temperature/field parameters
    if "temps" in config and config["temps"]:
        cmd.extend(["--temps"] + [str(t) for t in config["temps"]])
    if "fields" in config and config["fields"]:
        cmd.extend(["--fields"] + [str(f) for f in config["fields"]])
    if config.get("sample_delta_temp", False):
        cmd.append("--sample_delta_temp")
    if config.get("sample_delta_field", False):
        cmd.append("--sample_delta_field")
    if "delta_temp" in config:
        cmd.extend(["--delta_temp", str(config["delta_temp"])])
    if "delta_field" in config:
        cmd.extend(["--delta_field", str(config["delta_field"])])
    if "min_temp" in config:
        cmd.extend(["--min_temp", str(config["min_temp"])])
    if "max_temp" in config:
        cmd.extend(["--max_temp", str(config["max_temp"])])
    
    # Bias parameters
    if config.get("use_bias", False):
        cmd.append("--use_bias")
        cmd.extend(["--bias_sigma", str(config["bias_sigma"])])
        cmd.extend(["--bias_height", str(config["bias_height"])])
        cmd.extend(["--bias_factor", str(config["bias_factor"])])
        cmd.extend(["--bias_grid_size", str(config["bias_grid_size"])])
        cmd.extend(["--kernel_type", config["kernel_type"]])
        
        # Set CV defaults based on model type if not in config
        # Ising: CV (magnetization) ranges from -1 to 1
        # CuAu: CV (concentration) ranges from 0 to 1
        if "cv_min" in config and "cv_max" in config:
            cv_min = config["cv_min"]
            cv_max = config["cv_max"]
        else:
            # Determine model type: Ising has "L" and "J", CuAu has "size" or other indicators
            is_ising = "L" in config and "J" in config
            is_cuau = "size" in config or (config.get("dir_name", "").lower().find("cuau") != -1 or 
                                          config.get("dir_name", "").lower().find("au") != -1)
            
            if is_cuau:
                # CuAu: concentration CV ranges from 0 to 1
                cv_min = 0.0
                cv_max = 1.0
                print(f"Using default CV range for CuAu: [{cv_min}, {cv_max}]")
            else:
                # Ising: magnetization CV ranges from -1 to 1 (default)
                cv_min = -1.0
                cv_max = 1.0
                print(f"Using default CV range for Ising: [{cv_min}, {cv_max}]")
        
        cmd.extend(["--cv_min", str(cv_min)])
        cmd.extend(["--cv_max", str(cv_max)])
        
        if config.get("scale_bias_with_size", False):
            cmd.append("--scale_bias_with_size")
    
    # Buffer parameters
    if "buffer_size" in config:
        cmd.extend(["--buffer_size", str(config["buffer_size"])])
    if "buffer_ratio" in config:
        cmd.extend(["--buffer_ratio", str(config["buffer_ratio"])])
    
    # Buffer n_bins: default is 8 if not in config
    buffer_n_bins = config.get("buffer_n_bins", 8)
    cmd.extend(["--buffer_n_bins", str(buffer_n_bins)])
    
    # Buffer strategy: default is "balanced" if not in config
    buffer_strategy = config.get("buffer_strategy", "balanced")
    cmd.extend(["--buffer_strategy", buffer_strategy])
    
    # Wandb parameters
    cmd.append("--wandb")
    if "wandb_project" in config:
        cmd.extend(["--wandb_project", config["wandb_project"]])
    else:
        cmd.extend(["--wandb_project", "mdns-ising"])
    
    # Use wandb run name if provided, otherwise try config, then dir_name
    if wandb_run_name:
        cmd.extend(["--wandb_run_name", wandb_run_name])
    elif "wandb_run_name" in config:
        cmd.extend(["--wandb_run_name", config["wandb_run_name"]])
    elif config.get("dir_name"):
        # Use directory name as run name (fallback)
        run_name = Path(config["dir_name"]).name
        cmd.extend(["--wandb_run_name", run_name])
    
    # Add wandb resume ID if available
    # Note: train_ising.py doesn't currently support wandb resume, so we'll need to modify it
    # For now, we'll pass it as an environment variable or modify the script
    # Actually, wandb.init() with id= and resume="allow" should work
    
    return cmd, remaining_epochs, wandb_run_id


def main():
    parser = argparse.ArgumentParser(
        description="Resume training from an existing checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Resume training to 30000 total epochs
    python resume_training.py --config_dir /path/to/training/dir --total_epochs 30000
    
    # Resume training with specific config file
    python resume_training.py --config_path /path/to/config.json --total_epochs 30000
    
    # Resume training with custom device
    python resume_training.py --config_dir /path/to/training/dir --total_epochs 30000 --device cuda:1
        """
    )
    
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="Path to directory containing config.json"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config.json file"
    )
    parser.add_argument(
        "--total_epochs",
        type=int,
        required=True,
        help="Total number of epochs to train (will calculate remaining epochs from current checkpoint)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for training (default: cuda:0)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the command without executing it"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for wandb run ID detection"
    )
    
    args = parser.parse_args()
    
    # Determine config path
    if args.config_path:
        config_path = Path(args.config_path)
        config_dir = config_path.parent
    elif args.config_dir:
        config_dir = Path(args.config_dir)
        config_path = config_dir / "config.json"
    else:
        parser.error("Either --config_dir or --config_path must be provided")
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load config
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Detect model type
    is_potts = is_potts_config(config)
    model_type = "Potts" if is_potts else "Ising"
    print(f"Detected model type: {model_type}")
    
    # Find latest checkpoint
    print(f"Looking for checkpoints in: {config_dir}")
    try:
        checkpoint_path, epoch_from_filename = find_latest_checkpoint(config_dir)
        print(f"Found checkpoint: {checkpoint_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Get current epoch
    current_epoch = get_current_epoch_from_checkpoint(checkpoint_path)
    if current_epoch is None and epoch_from_filename is not None:
        current_epoch = epoch_from_filename
    elif current_epoch is None:
        print("Warning: Could not determine current epoch. Will train for total_epochs.")
        current_epoch = 0
    
    print(f"Current epoch: {current_epoch}")
    
    # Extract wandb run ID and run name
    # First, try to get from config.json if it was saved there
    wandb_run_id = config.get("wandb_run_id")
    if wandb_run_id:
        print(f"Found wandb run ID in config.json: {wandb_run_id}")
    else:
        # Try to extract from wandb directory
        wandb_run_id = extract_wandb_run_id(config_dir, debug=args.debug)
        if wandb_run_id:
            print(f"Found wandb run ID from wandb directory: {wandb_run_id}")
        else:
            print("Warning: Could not find wandb run ID. Will create new wandb run.")
            print("  This is normal if this is the first time resuming, or if wandb logging was disabled.")
            if args.debug:
                print(f"  Use --debug flag for more information about wandb directory structure.")
    
    wandb_run_name = extract_wandb_run_name(config_dir)
    if wandb_run_name:
        print(f"Found wandb run name: {wandb_run_name}")
    
    # Build command based on model type
    if is_potts:
        cmd, remaining_epochs, wandb_run_id = build_training_command_potts(
            config, checkpoint_path, wandb_run_id, wandb_run_name, args.total_epochs, current_epoch, args.device
        )
    else:
        cmd, remaining_epochs, wandb_run_id = build_training_command(
            config, checkpoint_path, wandb_run_id, wandb_run_name, args.total_epochs, current_epoch, args.device
        )
    
    print(f"\nTarget total epochs: {args.total_epochs}")
    print(f"Current epoch: {current_epoch}")
    print(f"Remaining epochs to train: {remaining_epochs}")
    print(f"\nCommand to execute:")
    print(" ".join(cmd))
    
    if wandb_run_id:
        print("\nNote: Wandb run ID found. The script will automatically resume wandb logging.")
        print(f"      Run ID: {wandb_run_id}")
    
    if args.dry_run:
        print("\nDry run mode - not executing command")
        return
    
    # Change to MDNS directory (parent of scripts directory)
    mdns_dir = Path(__file__).parent.parent
    os.chdir(mdns_dir)
    print(f"Changed to MDNS directory: {mdns_dir}")
    
    # Set wandb environment variables if run ID found
    env = os.environ.copy()
    if wandb_run_id:
        env["WANDB_RESUME"] = "allow"
        env["WANDB_RUN_ID"] = wandb_run_id
        print("\nSetting environment variables:")
        print("  WANDB_RESUME=allow")
        print(f"  WANDB_RUN_ID={wandb_run_id}")
    
    # Execute command
    print("\nExecuting training command...")
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
