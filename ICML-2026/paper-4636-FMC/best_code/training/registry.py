"""Trainer registry for discoverable method registration.

This module provides a registry pattern for trainers, allowing:
- Automatic discovery of available training methods
- Configuration-driven trainer instantiation
- Separation of trainer implementation from orchestration

Usage:
    from training.registry import register_trainer, get_trainer

    # Register a new trainer
    @register_trainer("my_method", category="calibration")
    class MyMethodTrainer(BaseTrainer):
        ...

    # Get a trainer instance
    trainer = get_trainer("my_method", config, model_path, task_config)

    # Get trainer with full config resolution (new modular config support)
    trainer = get_trainer_from_config(cfg, "fm_post_transform", model_path)
"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Type, Union

from training.base import BaseTrainer, TrainerConfig


class TrainerRegistry:
    """Registry for trainer classes.

    Provides a central location for registering and looking up trainers
    by name. Supports categorization (simulation, calibration, baseline).
    """

    _trainers: Dict[str, Type[BaseTrainer]] = {}
    _categories: Dict[str, List[str]] = {
        "simulation": [],
        "calibration": [],
        "baseline": [],
    }

    @classmethod
    def register(
        cls,
        name: str,
        trainer_class: Type[BaseTrainer],
        category: str = "calibration",
    ):
        """Register a trainer class.

        Args:
            name: Trainer name (used for lookup)
            trainer_class: Trainer class to register
            category: Category ("simulation", "calibration", or "baseline")
        """
        if name in cls._trainers:
            raise ValueError(f"Trainer '{name}' is already registered")

        cls._trainers[name] = trainer_class

        if category not in cls._categories:
            cls._categories[category] = []
        cls._categories[category].append(name)

        # Set class attributes
        trainer_class.name = name
        trainer_class.category = category

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseTrainer]]:
        """Get a trainer class by name."""
        return cls._trainers.get(name)

    @classmethod
    def list_trainers(cls, category: Optional[str] = None) -> List[str]:
        """List all registered trainer names.

        Args:
            category: Optional filter by category

        Returns:
            List of trainer names
        """
        if category is not None:
            return list(cls._categories.get(category, []))
        return list(cls._trainers.keys())

    @classmethod
    def list_categories(cls) -> List[str]:
        """List all categories."""
        return list(cls._categories.keys())

    @classmethod
    def get_category(cls, name: str) -> Optional[str]:
        """Get the category of a trainer."""
        for category, names in cls._categories.items():
            if name in names:
                return category
        return None

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a trainer is registered."""
        return name in cls._trainers


def register_trainer(name: str, category: str = "calibration"):
    """Decorator to register a trainer class.

    Usage:
        @register_trainer("my_method", category="calibration")
        class MyMethodTrainer(BaseTrainer):
            ...
    """
    def decorator(cls: Type[BaseTrainer]) -> Type[BaseTrainer]:
        TrainerRegistry.register(name, cls, category)
        return cls
    return decorator


def get_trainer(
    name: str,
    config: TrainerConfig,
    model_path: Path,
    task_config: Mapping[str, Any],
    logger: Optional[Any] = None,
) -> BaseTrainer:
    """Get a trainer instance by name.

    Args:
        name: Trainer name
        config: Training configuration
        model_path: Path for saving/loading models
        task_config: Task-specific configuration
        logger: Optional logger

    Returns:
        Trainer instance

    Raises:
        ValueError: If trainer not found
    """
    trainer_class = TrainerRegistry.get(name)
    if trainer_class is None:
        available = TrainerRegistry.list_trainers()
        raise ValueError(
            f"Trainer '{name}' not found. Available trainers: {available}"
        )

    return trainer_class(config, model_path, task_config, logger)


def get_trainer_from_task_config(
    name: str,
    task_config: Mapping[str, Any],
    model_path: Path,
    logger: Optional[Any] = None,
) -> BaseTrainer:
    """Get a trainer using task configuration.

    Extracts training parameters from task config and creates trainer.

    Args:
        name: Trainer name
        task_config: Full task configuration (contains method-specific config)
        model_path: Path for saving/loading models
        logger: Optional logger

    Returns:
        Trainer instance
    """
    # Extract method-specific config
    if name not in task_config:
        raise ValueError(f"No configuration found for '{name}' in task config")

    method_config = task_config[name]

    # Build TrainerConfig from method config
    training_params = method_config.get("training_params", method_config.get("training", {}))
    config = TrainerConfig.from_dict(training_params)
    config.model_config = method_config.get("config", method_config.get("params", {}))

    return get_trainer(name, config, model_path, task_config, logger)


def get_trainer_from_config(
    cfg: "DictConfig",
    name: str,
    model_path: Path,
    logger: Optional[Any] = None,
) -> BaseTrainer:
    """Get a trainer using full Hydra config with modular config support.

    Supports both old (task-embedded) and new (modular) config structures.
    For hyperparameter sweeps, use this function to get properly resolved configs.

    Args:
        cfg: Full Hydra DictConfig
        name: Trainer name
        model_path: Path for saving/loading models
        logger: Optional logger

    Returns:
        Trainer instance

    Example:
        # With new modular config:
        # python train_calibration.py method/fm_post_transform=large
        trainer = get_trainer_from_config(cfg, "fm_post_transform", model_path)
    """
    from utils.config import get_method_config

    # Get resolved method config (handles both old and new formats)
    method_config = get_method_config(cfg, name, fallback_to_task=True)

    # Build TrainerConfig
    training_params = method_config.get("training_params", method_config.get("training", {}))
    config = TrainerConfig.from_dict(training_params)
    config.model_config = method_config.get("config", method_config.get("params", {}))

    # Pass full task config for backward compatibility
    task_config = cfg.task if "task" in cfg else cfg

    return get_trainer(name, config, model_path, task_config, logger)
