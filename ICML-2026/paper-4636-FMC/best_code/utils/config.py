"""Configuration utilities for the refactored config system.

This module provides helpers for:
1. Resolving method configs from the new modular structure
2. Backward compatibility with old task-embedded configs
3. Experiment naming for hyperparameter sweeps

The new config structure separates:
- Task configs: simulator, data settings (configs/task/)
- Method configs: training params, method settings (configs/method/)
- Architecture configs: neural network architecture (configs/architecture/)

Usage:
    from utils.config import get_method_config, resolve_experiment_name

    # Get fully resolved method config
    method_cfg = get_method_config(cfg, "fm_post_transform")

    # Get experiment name (auto-generated for sweeps)
    exp_name = resolve_experiment_name(cfg)
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf


def get_method_config(
    cfg: DictConfig,
    method_name: str,
    fallback_to_task: bool = True,
) -> Dict[str, Any]:
    """Get fully resolved configuration for a method.

    Supports both old (task-embedded) and new (modular) config structures.

    Args:
        cfg: Full Hydra configuration
        method_name: Name of the method (e.g., "fm_post_transform")
        fallback_to_task: If True, fall back to task-embedded config

    Returns:
        Dictionary with method configuration including:
        - config: Method-specific config (architecture, params)
        - training_params: Training hyperparameters

    Raises:
        KeyError: If method config not found anywhere
    """
    # Try new structure first: method config at root level
    if method_name in cfg and not _is_task_embedded(cfg, method_name):
        return OmegaConf.to_container(cfg[method_name], resolve=True)

    # Fall back to task-embedded config (old structure)
    if fallback_to_task and "task" in cfg and method_name in cfg.task:
        return OmegaConf.to_container(cfg.task[method_name], resolve=True)

    # Check if method config was loaded via defaults
    if hasattr(cfg, "method_name") and cfg.method_name == method_name:
        # Build config from composed defaults
        return _build_method_config_from_defaults(cfg, method_name)

    raise KeyError(
        f"No configuration found for method '{method_name}'. "
        f"Ensure it's defined in configs/task/{cfg.task.name}.yaml or "
        f"configs/method/{method_name}/base.yaml"
    )


def _is_task_embedded(cfg: DictConfig, method_name: str) -> bool:
    """Check if method config is from task (old style) vs root (new style)."""
    if "task" not in cfg:
        return False
    # If method exists in both root and task, prefer root (new style)
    return method_name in cfg.task and method_name not in cfg


def _build_method_config_from_defaults(
    cfg: DictConfig,
    method_name: str,
) -> Dict[str, Any]:
    """Build method config from Hydra defaults composition."""
    result = {
        "config": {},
        "training_params": {},
    }

    # Get architecture if available
    if "architecture" in cfg:
        if method_name in ("fm_post_transform", "fm_post_transform_only_ut"):
            result["config"]["flow_theta"] = OmegaConf.to_container(
                cfg.architecture.flow_theta, resolve=True
            )
            result["config"]["flow_x"] = OmegaConf.to_container(
                cfg.architecture.flow_x, resolve=True
            )
        elif method_name in ("fmpe", "fmcpe_fm"):
            result["config"] = OmegaConf.to_container(
                cfg.architecture.flow_theta, resolve=True
            )

    # Get training params
    if "training_params" in cfg:
        result["training_params"] = OmegaConf.to_container(
            cfg.training_params, resolve=True
        )

    # Get NPE type for calibration methods
    if "npe_type" in cfg:
        result["config"]["npe"] = cfg.npe_type

    return result


def resolve_experiment_name(
    cfg: DictConfig,
    include_task: bool = True,
    include_method: bool = False,
    max_length: int = 50,
) -> str:
    """Generate experiment name, auto-generating for sweeps.

    For regular runs, uses cfg.experiment_name.
    For sweeps, appends config hash to ensure uniqueness.

    Args:
        cfg: Full Hydra configuration
        include_task: Include task name in generated name
        include_method: Include method name in generated name
        max_length: Maximum length for auto-generated names

    Returns:
        Experiment name string
    """
    base_name = cfg.get("experiment_name", "experiment")

    # Check if this is a sweep (Hydra job has overrides)
    is_sweep = _is_hydra_sweep(cfg)

    if not is_sweep:
        return base_name

    # For sweeps, generate unique name from config
    parts = [base_name]

    if include_task and "task" in cfg:
        parts.append(cfg.task.name)

    if include_method and "method_name" in cfg:
        parts.append(cfg.method_name)

    # Add hash of relevant config values for uniqueness
    config_hash = _hash_config(cfg)
    parts.append(config_hash[:8])

    name = "_".join(parts)
    return name[:max_length]


def _is_hydra_sweep(cfg: DictConfig) -> bool:
    """Check if running as part of a Hydra sweep."""
    # Check for multirun markers
    if hasattr(cfg, "_metadata"):
        return getattr(cfg._metadata, "multirun", False)

    # Check for sweep directory pattern
    if "hydra" in cfg:
        sweep_dir = cfg.hydra.get("sweep", {}).get("dir", "")
        return "sweeps" in sweep_dir

    return False


def _hash_config(cfg: DictConfig, keys: Optional[List[str]] = None) -> str:
    """Generate short hash of config for uniqueness."""
    if keys is None:
        # Hash key training parameters
        keys = ["training_params", "architecture", "sweep"]

    hash_input = []
    for key in keys:
        if key in cfg:
            value = OmegaConf.to_yaml(cfg[key])
            hash_input.append(f"{key}:{value}")

    hash_str = "|".join(hash_input)
    return hashlib.md5(hash_str.encode()).hexdigest()


def get_sweep_config(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    """Extract sweep configuration if present.

    Returns:
        Sweep config dict or None if not a sweep
    """
    if "sweep" not in cfg:
        return None
    return OmegaConf.to_container(cfg.sweep, resolve=True)


def list_method_variants(method_name: str, config_dir: Path = None) -> List[str]:
    """List available variants for a method.

    Args:
        method_name: Base method name (e.g., "fm_post_transform")
        config_dir: Path to configs directory

    Returns:
        List of variant names (e.g., ["base", "large", "deep"])
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / "configs"

    method_dir = config_dir / "method" / method_name
    if not method_dir.exists():
        return []

    variants = []
    for yaml_file in method_dir.glob("*.yaml"):
        variants.append(yaml_file.stem)

    return sorted(variants)


def list_architectures(config_dir: Path = None) -> List[str]:
    """List available architecture configs.

    Args:
        config_dir: Path to configs directory

    Returns:
        List of architecture names
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / "configs"

    arch_dir = config_dir / "architecture"
    if not arch_dir.exists():
        return []

    architectures = []
    for yaml_file in arch_dir.glob("*.yaml"):
        architectures.append(yaml_file.stem)

    return sorted(architectures)


def print_config_summary(cfg: DictConfig, methods: List[str] = None):
    """Print a summary of the configuration.

    Args:
        cfg: Full Hydra configuration
        methods: Optional list of methods to summarize
    """
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)

    # Experiment info
    print(f"Experiment: {cfg.get('experiment_name', 'N/A')}")
    print(f"Results dir: {cfg.get('results_dir', 'N/A')}")

    # Task info
    if "task" in cfg:
        print(f"Task: {cfg.task.name}")
        if "simulator" in cfg.task:
            print(f"Simulator: {cfg.task.simulator.name}")

    # Architecture info
    if "architecture" in cfg:
        arch_name = cfg.architecture.get("name", "unknown")
        print(f"Architecture: {arch_name}")

    # Methods
    if methods:
        print(f"\nMethods: {methods}")
        for method in methods:
            try:
                method_cfg = get_method_config(cfg, method)
                tp = method_cfg.get("training_params", {})
                print(f"  {method}:")
                print(f"    lr={tp.get('lr', 'N/A')}, "
                      f"epochs={tp.get('epochs', 'N/A')}, "
                      f"batch_size={tp.get('batch_size', 'N/A')}")
            except KeyError:
                print(f"  {method}: (config not found)")

    print("=" * 60)


# Register custom OmegaConf resolvers
def register_custom_resolvers():
    """Register custom resolvers for config interpolation."""
    from omegaconf import OmegaConf

    # Resolver to repeat a value n times: ${repeat:64,5} -> [64,64,64,64,64]
    if not OmegaConf.has_resolver("repeat"):
        OmegaConf.register_new_resolver(
            "repeat",
            lambda value, n: [int(value)] * int(n)
        )

    # Resolver to create range: ${range:32,256,2} -> [32,64,128,256]
    if not OmegaConf.has_resolver("range_power2"):
        OmegaConf.register_new_resolver(
            "range_power2",
            lambda start, end, step=2: [
                int(start) * (int(step) ** i)
                for i in range(int((end / start).bit_length()))
                if int(start) * (int(step) ** i) <= int(end)
            ]
        )


# Auto-register resolvers on import
register_custom_resolvers()


# =============================================================================
# Variant Resolution Helpers
# =============================================================================

def get_variant_config_path(
    variant_name: str,
    task: Optional[str] = None,
    config_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Get path to variant config file.

    Args:
        variant_name: Name of the variant (e.g., "fm_pt_large")
        task: Optional task name for task-specific variant lookup
        config_dir: Path to configs directory (defaults to ./configs)

    Returns:
        Path to variant config file, or None if not found
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / "configs"

    # Check task-specific first
    if task is not None:
        task_path = config_dir / "variant" / "task" / task / f"{variant_name}.yaml"
        if task_path.exists():
            return task_path

    # Fall back to global
    global_path = config_dir / "variant" / f"{variant_name}.yaml"
    if global_path.exists():
        return global_path

    return None


def list_variants(
    include_task_specific: bool = True,
    config_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """List all available variants, grouped by scope.

    Args:
        include_task_specific: Include task-specific variants
        config_dir: Path to configs directory

    Returns:
        Dict with 'global' key and optionally task names as keys,
        each containing list of variant names
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / "configs"

    result = {"global": []}

    # List global variants
    variant_dir = config_dir / "variant"
    if variant_dir.exists():
        result["global"] = sorted([
            f.stem for f in variant_dir.glob("*.yaml")
            if f.is_file()
        ])

    # List task-specific variants
    if include_task_specific:
        task_dir = variant_dir / "task"
        if task_dir.exists():
            for task_path in task_dir.iterdir():
                if task_path.is_dir():
                    task_variants = sorted([
                        f.stem for f in task_path.glob("*.yaml")
                    ])
                    if task_variants:
                        result[task_path.name] = task_variants

    return result
