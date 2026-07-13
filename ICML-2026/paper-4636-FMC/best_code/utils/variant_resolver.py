"""Variant resolution with per-task override support.

This module provides task-aware variant resolution:
1. First checks `configs/variant/task/{task}/{variant_name}.yaml`
2. Falls back to `configs/variant/{variant_name}.yaml`
3. Falls back to task-embedded config (legacy)

The `_base` key in task-specific variants enables inheritance from global variants.

Usage:
    from utils.variant_resolver import resolve_variant_config

    # Get variant config for a specific task
    config = resolve_variant_config("high_dim_gaussian", "fm_pt_large")

    # This might return the task-specific config if it exists,
    # or the global fm_pt_large config otherwise
"""

from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf


# Default configs directory (relative to project root)
_CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def resolve_variant_config(
    task: str,
    variant_name: str,
    configs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Resolve variant configuration with task-specific override support.

    Resolution order:
    1. configs/variant/task/{task}/{variant_name}.yaml (task-specific)
    2. configs/variant/{variant_name}.yaml (global fallback)
    3. Empty dict (if nothing found - legacy behavior handles this)

    Args:
        task: Task name (e.g., "high_dim_gaussian")
        variant_name: Variant name (e.g., "fm_pt_large")
        configs_dir: Optional path to configs directory

    Returns:
        Dictionary with resolved variant configuration
    """
    configs_dir = configs_dir or _CONFIGS_DIR

    # Path 1: Task-specific variant
    task_variant_path = configs_dir / "variant" / "task" / task / f"{variant_name}.yaml"

    # Path 2: Global variant
    global_variant_path = configs_dir / "variant" / f"{variant_name}.yaml"

    resolved_config = {}

    # Check task-specific first
    if task_variant_path.exists():
        task_cfg = OmegaConf.load(task_variant_path)

        # Check for _base inheritance
        if "_base" in task_cfg:
            base_name = task_cfg._base
            base_config = _load_global_variant(base_name, configs_dir)
            # Deep merge: base first, then task-specific overrides
            resolved_config = _deep_merge(base_config, OmegaConf.to_container(task_cfg, resolve=True))
            # Remove the _base key from final config
            resolved_config.pop("_base", None)
        else:
            resolved_config = OmegaConf.to_container(task_cfg, resolve=True)

    # Fall back to global variant
    elif global_variant_path.exists():
        resolved_config = _load_global_variant(variant_name, configs_dir)

    return resolved_config


def _load_global_variant(variant_name: str, configs_dir: Path) -> Dict[str, Any]:
    """Load a global variant config.

    Note: We use resolve=False to avoid interpolation errors when loading
    standalone variant configs that may reference parent contexts (e.g., ${variant.base_dist}).
    Interpolations will be resolved later when the config is merged with the full context.
    """
    variant_path = configs_dir / "variant" / f"{variant_name}.yaml"
    if not variant_path.exists():
        return {}
    cfg = OmegaConf.load(variant_path)
    # Use resolve=False to avoid errors with interpolations that reference parent contexts
    return OmegaConf.to_container(cfg, resolve=False)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary
        override: Override dictionary (values take precedence)

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            # Recursively merge nested dicts
            result[key] = _deep_merge(result[key], value)
        else:
            # Override value
            result[key] = value

    return result


def get_variant_paths(
    task: str,
    variant_name: str,
    configs_dir: Optional[Path] = None,
) -> Dict[str, Optional[Path]]:
    """Get paths to variant configs (for debugging/logging).

    Args:
        task: Task name
        variant_name: Variant name
        configs_dir: Optional path to configs directory

    Returns:
        Dict with 'task_specific' and 'global' paths (None if doesn't exist)
    """
    configs_dir = configs_dir or _CONFIGS_DIR

    task_path = configs_dir / "variant" / "task" / task / f"{variant_name}.yaml"
    global_path = configs_dir / "variant" / f"{variant_name}.yaml"

    return {
        "task_specific": task_path if task_path.exists() else None,
        "global": global_path if global_path.exists() else None,
    }


def list_task_variants(task: str, configs_dir: Optional[Path] = None) -> list[str]:
    """List available task-specific variants for a task.

    Args:
        task: Task name
        configs_dir: Optional path to configs directory

    Returns:
        List of variant names with task-specific configs
    """
    configs_dir = configs_dir or _CONFIGS_DIR
    task_dir = configs_dir / "variant" / "task" / task

    if not task_dir.exists():
        return []

    return sorted([f.stem for f in task_dir.glob("*.yaml")])


def list_global_variants(configs_dir: Optional[Path] = None) -> list[str]:
    """List available global variants.

    Args:
        configs_dir: Optional path to configs directory

    Returns:
        List of global variant names
    """
    configs_dir = configs_dir or _CONFIGS_DIR
    variant_dir = configs_dir / "variant"

    if not variant_dir.exists():
        return []

    # Only list yaml files directly in variant/, not in subdirectories
    return sorted([f.stem for f in variant_dir.glob("*.yaml")])


def resolve_variant_model_config(
    task: str,
    variant_name: str,
    task_config: Dict[str, Any],
    configs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Resolve complete model config by merging variant config with task config.

    This is the main function used by VariantTrainer to get the final config.

    Resolution priority (highest to lowest):
    1. Task-specific variant config (configs/variant/task/{task}/{variant}.yaml)
    2. Global variant config (configs/variant/{variant}.yaml)
    3. Task config's method-specific settings (task_config[method])

    Args:
        task: Task name
        variant_name: Variant name
        task_config: Task configuration dictionary
        configs_dir: Optional path to configs directory

    Returns:
        Merged configuration with all overrides applied
    """
    # Get variant config (already resolved with _base inheritance)
    variant_config = resolve_variant_config(task, variant_name, configs_dir)

    # If variant has a 'config' section, it defines the model architecture
    # This overrides task config's method-specific architecture
    model_config = variant_config.get("config", {})

    # Get method name from variant
    method = variant_config.get("method", "fm_post_transform")

    # Get task's method config as base
    task_method_config = task_config.get(method, {})
    task_model_config = task_method_config.get("config", {})

    # Deep merge: task config provides base, variant provides overrides
    if model_config:
        final_model_config = _deep_merge(task_model_config, model_config)
    else:
        final_model_config = task_model_config

    return {
        "variant_name": variant_name,
        "method": method,
        "base_dist": variant_config.get("base_dist", "fmpe"),
        "training_params": variant_config.get("training_params", {}),
        "config": final_model_config,
        "hpo": variant_config.get("hpo", {}),
    }
