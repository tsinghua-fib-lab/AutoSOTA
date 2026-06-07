"""Configuration helpers for MASS training and evaluation scripts.

This module loads YAML configs, applies command-line overrides, prepares output
directories, and exposes small helpers shared by entry points and trainers.
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import torch.distributed as dist


def load_yaml(path: Path) -> dict:
    """Read a YAML file and return a dict."""
    with path.open() as f:
        return yaml.safe_load(f)


def merge_configs(base_cfg: dict, override_cfg: dict) -> dict:
    """Recursively merge override into base config (creates a new dict)."""
    merged = base_cfg.copy()
    for k, v in override_cfg.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


def apply_cli_overrides(cfg: dict, overrides: List[str]) -> dict:
    """Apply CLI overrides of the form key1.key2=val."""
    if not overrides:
        return cfg
    
    result = cfg.copy()
    for override in overrides:
        key_path, raw_val = override.split('=', 1)
        keys = key_path.split('.')
        
        d = result
        for subkey in keys[:-1]:
            if subkey not in d:
                d[subkey] = {}
            elif not isinstance(d[subkey], dict):
                d[subkey] = {}
            d = d[subkey]
        
        d[keys[-1]] = yaml.safe_load(raw_val)
    
    return result


def get_timestamp() -> str:
    """Generate a timestamp string for unique run identification."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_config(config: dict, save_dir: str, filename: str = "config.yaml") -> None:
    """Save configuration to a YAML file."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return save_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for configuration."""
    parser = argparse.ArgumentParser(description="Iris: In-context Reference Image guided Segmentation")
    
    # Core configuration arguments
    parser.add_argument('--config', type=str, default='config/pretrain/mask_guided_self_supervised.yaml',
                        help='Path to base configuration file')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Name of experiment configuration to override base config')
    parser.add_argument('--override', nargs='*', default=None,
                        help='Override specific config values: key1.key2=value')
    
    # DDP arguments
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID(s) to use, comma-separated for multiple GPUs')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='Number of nodes for distributed training')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='Ranking within the nodes for distributed training')
    
    # Run specifications
    parser.add_argument('--name', type=str, default=None,
                        help='Run name (default: auto-generated from timestamp and config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def prepare_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Prepare configuration by loading base config, applying experiment
    overrides, and CLI overrides.
    
    Supports auto-loading config from checkpoint if --resume is provided without --config.
    """
    if args.resume and not args.config:
        from utils.checkpoint import load_checkpoint
        
        print(f"Auto-loading configuration from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume, config_only=True)
        config = checkpoint['config']
        
        config['auto_resumed'] = True

        if 'output_dir' not in config:
            config['output_dir'] = 'runs'
    else:
        if not args.config:
            raise ValueError("Either --config or --resume must be provided")

        config_path = Path(args.config)
        config = load_yaml(config_path)

        if args.experiment:
            exp_path = config_path.parent / "experiments" / f"{args.experiment}.yaml"
            if exp_path.exists():
                exp_config = load_yaml(exp_path)
                config = merge_configs(config, exp_config)
            else:
                raise FileNotFoundError(f"Experiment config not found: {exp_path}")
    
    if args.override:
        # CLI overrides are applied last so launch scripts can change a single
        # nested value without copying the whole YAML.
        config = apply_cli_overrides(config, args.override)

    timestamp = get_timestamp()
    config['run'] = config.get('run', {})
    if 'seed' not in config and 'seed' in config['run']:
        config['seed'] = config['run'].pop('seed')

    if args.resume:
        config['run']['resume'] = args.resume
        checkpoint_path = Path(args.resume)

        if checkpoint_path.parent.name == 'checkpoints':
            # Standard run layout: output_dir/checkpoints/latest.pth.
            original_output_dir = str(checkpoint_path.parent.parent)
            config['run']['output_dir'] = original_output_dir
            print(f"Continuing in original output directory: {original_output_dir}")
            config['run']['name'] = Path(original_output_dir).name
        else:
            config['run']['timestamp'] = timestamp
            config['run']['name'] = f"resumed_{timestamp}"
            config['run']['output_dir'] = os.path.join(
                config.get('output_dir', 'runs'),
                config['run']['name']
            )
            print(f"Creating new output directory: {config['run']['output_dir']}")
    else:
        config['run']['timestamp'] = timestamp
        if args.name:
            config['run']['name'] = args.name
        elif 'name' not in config['run']:
            experiment_name = args.experiment or 'default'
            config['run']['name'] = f"{experiment_name}_{timestamp}"

        config['run']['output_dir'] = os.path.join(
            config['run']['output_dir'],
            config['run']['name']
        )
    
    gpus = [int(g) for g in args.gpu.split(',')]
    distributed_config = config.get('distributed', {})

    # Keep distributed metadata in the config so trainers and launch utilities
    # do not need to re-parse CLI arguments.
    distributed_config.update({
        'gpus': gpus,
        'gpus_per_node': len(gpus),
        'world_size': args.num_nodes * len(gpus),
        'node_rank': args.node_rank,
        'num_nodes': args.num_nodes,
    })

    config['distributed'] = distributed_config
    
    return config


def is_master() -> bool:
    """Check if this is the master process in distributed training."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def setup_output_dir(config: Dict[str, Any]) -> str:
    """Set up output directory for checkpoints, logs, etc."""
    if 'run' not in config or 'output_dir' not in config['run']:
        raise ValueError(
            "Configuration is missing 'run.output_dir'. "
            "This can happen when loading a checkpoint without its training config. "
            "Please specify --config with your configuration file."
        )
    
    output_dir = config['run']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    logs_dir = os.path.join(output_dir, 'logs')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    if is_master():
        save_config(config, output_dir)
        
    return output_dir
