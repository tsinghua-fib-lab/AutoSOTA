"""Shared utilities for benchmark modules."""

from pathlib import Path
from typing import Any, Dict


def apply_overrides(config: Dict, overrides: Dict) -> Dict:
    """Recursively apply overrides to a config dict."""
    result = config.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = apply_overrides(result[key], value)
        else:
            result[key] = value
    return result


def get_task_config(
    task: str,
    task_configs: Dict[str, Dict],
    task_overrides: Dict[str, Any],
) -> Dict:
    """Get task configuration with overrides applied.

    Args:
        task: Task name
        task_configs: Dict of pre-loaded task configs
        task_overrides: Overrides to apply to the config
    """
    task_cfg = None

    if task in task_configs:
        task_cfg = task_configs[task]
        if task_overrides:
            task_cfg = apply_overrides(task_cfg, task_overrides)
    else:
        try:
            from omegaconf import OmegaConf
            config_path = Path("configs/task") / f"{task}.yaml"
            if config_path.exists():
                loaded_cfg = OmegaConf.load(config_path)

                # Apply overrides BEFORE resolving interpolations
                # This ensures ${task.rescale} picks up the overridden value
                if task_overrides:
                    override_cfg = OmegaConf.create(task_overrides)
                    loaded_cfg = OmegaConf.merge(loaded_cfg, override_cfg)

                # Wrap in 'task:' to allow self-references like ${task.rescale}
                wrapped = OmegaConf.create({"task": loaded_cfg})
                resolved = OmegaConf.to_container(wrapped, resolve=True)
                task_cfg = resolved["task"]
        except (ImportError, Exception) as e:
            print(f"  Warning: Could not load task config for {task}: {e}")

    # Use minimal config if nothing found
    if task_cfg is None:
        task_cfg = {"name": task, "generation": "independent"}
        if task_overrides:
            task_cfg = apply_overrides(task_cfg, task_overrides)

    return task_cfg
