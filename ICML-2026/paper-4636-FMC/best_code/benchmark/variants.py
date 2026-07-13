"""Variant training module for benchmark experiments (Phase 2).

This module handles training fm_post_transform variants and other calibration
methods that build on the shared foundation models.

Usage:
    trainer = VariantTrainer.from_config(cfg, variant_name="fm_pt_fmpe_lr1e3")
    trainer.run(tasks=["pendulum", "gaussian"])

    # With HPO
    trainer = VariantTrainer.from_config(cfg, variant_name="fm_pt_hpo")
    trainer.run_hpo(tasks=["pendulum"], n_trials=50)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from core.checkpointing import CheckpointManager, ResultsPath, TrainingLog
from utils.data_split import BenchmarkSplitter, DEFAULT_CALIBRATION_SIZES
from utils.locking import TrainingLock, atomic_write
from utils.variant_resolver import resolve_variant_config, resolve_variant_model_config


@dataclass
class VariantConfig:
    """Configuration for variant training."""

    # Variant identification
    variant_name: str
    method: str = "fm_post_transform"

    # Base distribution (for fm_post_transform)
    base_dist: str = "fmpe"  # "npe" or "fmpe"

    # Training parameters (None means use task config)
    lr: Optional[float] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    max_patience: Optional[int] = None  # None in config means use task config; null in yaml means no early stopping
    train_size: Optional[float] = None
    rescale: Optional[str] = None  # None means use task config

    # Data configuration
    calibration_sizes: List[int] = field(default_factory=lambda: DEFAULT_CALIBRATION_SIZES.copy())
    seeds: List[int] = field(default_factory=lambda: [33, 43, 53])

    # Model configuration (method-specific)
    model_config: Dict[str, Any] = field(default_factory=dict)

    # Paths
    results_dir: Path = Path("results")
    experiment_name: str = "comparison_benchmark"

    # Parallel execution
    use_locks: bool = True
    lock_timeout: float = 300.0  # 5 minutes

    # HPO configuration
    hpo_enabled: bool = False
    hpo_n_trials: int = 50
    hpo_search_space: Dict[str, Any] = field(default_factory=dict)

    # Task overrides - applied to all task configs
    task_overrides: Dict[str, Any] = field(default_factory=dict)

    # Method overrides - applied to specific methods (e.g., fm_post_transform: {rescale: none})
    method_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg, variant_name: str) -> "VariantConfig":
        """Create from Hydra config."""
        # Get variant-specific config if it exists
        variant_cfg = cfg.get("variant", {})
        training_params = variant_cfg.get("training_params", {})

        # Extract task_overrides from config
        task_overrides = cfg.get("task_overrides", {})
        if hasattr(task_overrides, "items"):
            task_overrides = dict(task_overrides)
        else:
            task_overrides = {}

        # Extract method_overrides from config
        method_overrides = cfg.get("method_overrides", {})
        if hasattr(method_overrides, "items"):
            # Convert OmegaConf to dict recursively
            method_overrides = {k: dict(v) if hasattr(v, "items") else v for k, v in method_overrides.items()}
        else:
            method_overrides = {}

        return cls(
            variant_name=variant_name,
            method=variant_cfg.get("method", "fm_post_transform"),
            base_dist=variant_cfg.get("base_dist", "fmpe"),
            lr=training_params.get("lr"),  # None if not specified
            epochs=training_params.get("epochs"),
            batch_size=training_params.get("batch_size"),
            max_patience=training_params.get("max_patience"),  # Can be None (use task) or explicit null (no early stop)
            train_size=training_params.get("train_size"),
            rescale=training_params.get("rescale"),  # None means use task config
            calibration_sizes=list(cfg.get("data", {}).get(
                "calibration_sizes", DEFAULT_CALIBRATION_SIZES
            )),
            # Seeds from experiment config (no fallback to config.yaml)
            seeds=list(cfg.get("seeds", [33, 43, 53])),
            model_config=dict(variant_cfg.get("config", {})),
            results_dir=Path(cfg.get("results_dir", "results")),
            experiment_name=cfg.get("experiment", {}).get("name", "comparison_benchmark"),
            use_locks=cfg.get("parallel", {}).get("use_locks", True),
            lock_timeout=cfg.get("parallel", {}).get("lock_timeout", 300.0),
            hpo_enabled=variant_cfg.get("hpo", {}).get("enabled", False),
            hpo_n_trials=variant_cfg.get("hpo", {}).get("n_trials", 50),
            hpo_search_space=dict(variant_cfg.get("hpo", {}).get("search_space", {})),
            task_overrides=task_overrides,
            method_overrides=method_overrides,
        )


class VariantTrainer:
    """Handles Phase 2: Variant training using shared foundation models."""

    def __init__(
        self,
        config: VariantConfig,
        task_configs: Optional[Dict[str, Dict]] = None,
    ):
        """Initialize variant trainer.

        Args:
            config: Variant configuration
            task_configs: Optional task-specific configs
        """
        self.config = config
        self.task_configs = task_configs or {}

        # Setup paths and checkpoint manager
        self.paths = ResultsPath(config.results_dir, config.experiment_name)
        self.checkpoint_manager = CheckpointManager(self.paths)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Save variant config
        self._save_variant_config()

    @classmethod
    def from_config(cls, cfg, variant_name: str) -> "VariantTrainer":
        """Create trainer from Hydra config."""
        variant_config = VariantConfig.from_config(cfg, variant_name)

        # Extract task configs
        task_configs = {}
        if "task" in cfg:
            task_name = cfg.task.get("name", cfg.get("task_name", "unknown"))
            task_configs[task_name] = dict(cfg.task)
        elif "tasks" in cfg:
            for task_name in cfg.tasks:
                if hasattr(cfg, task_name):
                    task_configs[task_name] = dict(cfg[task_name])

        return cls(variant_config, task_configs)

    def _save_variant_config(self):
        """Save variant configuration."""
        config_dict = {
            "variant_name": self.config.variant_name,
            "method": self.config.method,
            "base_dist": self.config.base_dist,
            "training_params": {
                "lr": self.config.lr,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "max_patience": self.config.max_patience,
                "train_size": self.config.train_size,
            },
            "model_config": self.config.model_config,
            "calibration_sizes": self.config.calibration_sizes,
            "seeds": self.config.seeds,
            "created_at": datetime.now().isoformat(),
        }

        self.checkpoint_manager.save_variant_config(
            self.config.variant_name, config_dict
        )

    def run(
        self,
        tasks: List[str],
        seeds: Optional[List[int]] = None,
        ncals: Optional[List[int]] = None,
        skip_existing: bool = True,
        parallel: bool = False,
    ) -> Dict[str, Any]:
        """Run variant training for all tasks.

        Args:
            tasks: List of task names to train
            seeds: Optional list of seeds (defaults to config)
            ncals: Optional list of ncal values (defaults to config)
            skip_existing: Skip training if model already exists
            parallel: Enable parallel execution with locks

        Returns:
            Summary of training results
        """
        seeds = seeds or self.config.seeds
        ncals = ncals or self.config.calibration_sizes

        results = {}

        for task in tasks:
            print(f"\n{'='*60}")
            print(f"Variant training: {self.config.variant_name} on {task}")
            print(f"{'='*60}")
            print(f"Seeds: {seeds}, Calibration sizes: {ncals}")

            task_results = {}
            for seed in seeds:
                task_results[seed] = {}
                for ncal in ncals:
                    result = self._train_variant(
                        task, seed, ncal, skip_existing, parallel
                    )
                    task_results[seed][ncal] = result

            results[task] = task_results

        return results

    def run_single(
        self,
        task: str,
        seed: int,
        ncal: int,
        skip_existing: bool = True,
        parallel: bool = False,
    ) -> Dict[str, Any]:
        """Run variant training for a single task/seed/ncal combination.

        Useful for cluster array jobs.
        """
        return self._train_variant(task, seed, ncal, skip_existing, parallel)

    def run_hpo(
        self,
        tasks: List[str],
        n_trials: Optional[int] = None,
        seed: int = 0,
        ncal: int = 200,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization for the variant.

        Args:
            tasks: List of tasks to optimize on
            n_trials: Number of HPO trials
            seed: Seed to use for HPO
            ncal: Calibration size to use for HPO
            study_name: Optional Optuna study name
            storage: Optional Optuna storage URL

        Returns:
            HPO results
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for HPO. Install with: pip install optuna")

        n_trials = n_trials or self.config.hpo_n_trials
        results = {}

        for task in tasks:
            print(f"\n{'='*60}")
            print(f"HPO for {self.config.variant_name} on {task}")
            print(f"{'='*60}")

            # Create or load study
            task_study_name = study_name or f"{self.config.variant_name}_{task}"
            study = optuna.create_study(
                study_name=task_study_name,
                storage=storage,
                direction="minimize",
                load_if_exists=True,
            )

            # Create objective function
            objective = self._create_hpo_objective(task, seed, ncal)

            # Run optimization
            study.optimize(objective, n_trials=n_trials)

            # Save results
            results[task] = {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "n_trials": len(study.trials),
            }

            # Save HPO results
            hpo_results_path = (
                self.paths.variant_path(self.config.variant_name)
                / "hpo"
                / task
                / "hpo_results.json"
            )
            hpo_results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(hpo_results_path, "w") as f:
                json.dump(results[task], f, indent=2)

            print(f"Best params: {study.best_params}")
            print(f"Best value: {study.best_value:.4f}")

        return results

    def _create_hpo_objective(
        self,
        task: str,
        seed: int,
        ncal: int,
    ) -> Callable:
        """Create Optuna objective function."""

        def objective(trial):
            # Sample hyperparameters
            params = self._sample_hpo_params(trial)

            # Create temporary config with sampled params
            temp_config = VariantConfig(
                variant_name=f"{self.config.variant_name}_trial{trial.number}",
                method=self.config.method,
                base_dist=self.config.base_dist,
                **params,
                model_config=self.config.model_config,
                results_dir=self.config.results_dir,
                experiment_name=self.config.experiment_name,
                calibration_sizes=self.config.calibration_sizes,
                seeds=[seed],
            )

            # Train with trial for pruning support
            result = self._do_train_variant(
                task, seed, ncal, temp_config, trial=trial
            )

            return result.get("val_loss", float("inf"))

        return objective

    def _sample_hpo_params(self, trial) -> Dict[str, Any]:
        """Sample hyperparameters from search space."""
        params = {}
        search_space = self.config.hpo_search_space

        for param_name, param_spec in search_space.items():
            param_type = param_spec.get("type", "float")

            if param_type == "loguniform":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_spec["low"],
                    param_spec["high"],
                    log=True,
                )
            elif param_type == "uniform":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_spec["low"],
                    param_spec["high"],
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_spec["low"],
                    param_spec["high"],
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_spec["choices"],
                )

        # Fill in defaults for non-HPO params
        defaults = {
            "lr": self.config.lr,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "max_patience": self.config.max_patience,
            "train_size": self.config.train_size,
        }
        for key, value in defaults.items():
            if key not in params:
                params[key] = value

        return params

    def _train_variant(
        self,
        task: str,
        seed: int,
        ncal: int,
        skip_existing: bool,
        parallel: bool,
    ) -> Dict[str, Any]:
        """Train variant for a single task/seed/ncal."""
        variant_name = self.config.variant_name

        # Check if model already exists
        if skip_existing and self.checkpoint_manager.variant_calibration_model_exists(
            variant_name, task, seed, self.config.method, ncal
        ):
            print(f"  {self.config.method} (seed={seed}, ncal={ncal}) already exists, skipping...")
            return {"status": "skipped"}

        # Lock if parallel mode
        locks_dir = self.paths.locks_path
        check_exists = lambda: self.checkpoint_manager.variant_calibration_model_exists(
            variant_name, task, seed, self.config.method, ncal
        )

        if parallel and self.config.use_locks:
            with TrainingLock(
                locks_dir, task, seed, ncal,
                variant_name=variant_name,
                check_exists=check_exists,
            ) as lock:
                if lock.already_exists:
                    return {"status": "skipped"}
                return self._do_train_variant(task, seed, ncal, self.config)
        else:
            return self._do_train_variant(task, seed, ncal, self.config)

    def _do_train_variant(
        self,
        task: str,
        seed: int,
        ncal: int,
        config: VariantConfig,
        trial: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Actually train variant model."""
        from training.registry import get_trainer
        from training.base import TrainerConfig

        # Load base distribution model
        base_model = self.checkpoint_manager.load_shared_simulation_model(
            task, config.base_dist, self.device
        )
        if base_model is None:
            raise ValueError(
                f"Base model ({config.base_dist}) not found for {task}. "
                "Run foundation training first."
            )

        # Load calibration data
        cal_data = self.checkpoint_manager.load_shared_calibration_pool(task)
        if cal_data is None:
            raise ValueError(f"No calibration data found for {task}")

        # Get calibration subset
        data_path = self.paths.shared_data_path(task)
        splitter = BenchmarkSplitter(
            n_pool=2000,  # Default
            n_test=5000,  # Default
            calibration_sizes=config.calibration_sizes,
        )
        split = splitter.load_split(data_path)
        if split is None:
            raise ValueError(f"No data split found for {task}")

        # Get calibration indices for this seed
        cal_indices = split.get_calibration_indices(ncal, subsample_seed=seed)

        theta_cal = cal_data["theta"][cal_indices]
        x_cal = cal_data["x"][cal_indices]
        y_cal = cal_data["y"][cal_indices]

        # Get task config
        task_config = self._get_task_config(task)

        # Resolve variant model config (handles task-specific variants and inheritance)
        resolved_variant = resolve_variant_model_config(
            task=task,
            variant_name=config.variant_name,
            task_config=task_config,
        )

        # Get training params from resolved variant (already merged with task config)
        variant_training_params = resolved_variant.get("training_params", {})

        # Get training params from task config's method-specific settings (fallback)
        method_config = task_config.get(config.method, {})
        task_training_params = method_config.get("training_params", method_config.get("training", {}))

        # Method-specific overrides from experiment config (e.g., fm_post_transform: {rescale: none})
        method_overrides = config.method_overrides.get(config.method, {})

        # Helper: variant config > method overrides > task config > global defaults
        # Resolution priority: resolved_variant > explicit variant config > method_overrides > task config
        def resolve(variant_val, key, default):
            # Check resolved variant first (includes task-specific overrides)
            if key in variant_training_params:
                return variant_training_params[key]
            # Then check explicit variant config value
            if variant_val is not None:
                return variant_val
            # Then method overrides
            if key in method_overrides:
                return method_overrides[key]
            # Finally task config
            return task_training_params.get(key, default)

        lr = resolve(config.lr, "lr", 1e-3)
        epochs = resolve(config.epochs, "epochs", 200)
        batch_size = resolve(config.batch_size, "batch_size", 256)
        train_size = resolve(config.train_size, "train_size", 0.8)
        rescale = resolve(config.rescale, "rescale", task_config.get("rescale", "z_score"))
        # For max_patience: None in variant config means "use task config",
        # but task config can also have null meaning "no early stopping"
        max_patience = resolve(config.max_patience, "max_patience", 20)

        # Use resolved model config (architecture from task-specific variant if exists)
        final_model_config = resolved_variant.get("config", {}) or config.model_config

        # Build trainer config
        trainer_config = TrainerConfig(
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            max_patience=max_patience,
            train_size=train_size,
            rescale=rescale,
            model_config=final_model_config,  # Use resolved model config (task-specific if exists)
        )

        # Get model path
        model_path = self.paths.variant_calibration_model_path(
            config.variant_name, task, seed, config.method, ncal
        )
        model_path.mkdir(parents=True, exist_ok=True)

        # Get trainer
        trainer = get_trainer(
            config.method,
            trainer_config,
            model_path,
            task_config,
        )

        # Set seed for reproducibility
        torch.manual_seed(seed)

        # Train model
        print(f"  Training {config.method} (seed={seed}, ncal={ncal})...")
        posterior = trainer.train(
            data=(theta_cal, x_cal, y_cal),
            device=self.device,
            logname=f"{config.method}_seed{seed}_ncal{ncal}",
            trial=trial,  # Pass trial for HPO pruning
            base_posterior=base_model,  # Pass base model
        )

        # Create training log
        training_log = TrainingLog(
            model_type=config.method,
            epochs_completed=trainer.state.epoch,
            best_val_loss=trainer.state.best_val_loss,
            final_train_loss=trainer.state.best_train_loss,
            config=trainer.config.to_dict(),
            extra={
                "seed": seed,
                "ncal": ncal,
                "variant_name": config.variant_name,
                "base_dist": config.base_dist,
            },
            loss_history=trainer.state.loss_history,
        )
        training_log.mark_completed(
            best_val_loss=trainer.state.best_val_loss,
            final_train_loss=trainer.state.best_train_loss,
        )

        # Save model
        self.checkpoint_manager.save_variant_calibration_model(
            model=posterior,
            variant_name=config.variant_name,
            task=task,
            seed=seed,
            method=config.method,
            ncal=ncal,
            config=trainer.config.to_dict(),
            training_log=training_log,
        )

        print(f"  Training completed: val_loss={trainer.state.best_val_loss:.4f}")

        return {
            "status": "trained",
            "val_loss": trainer.state.best_val_loss,
            "epochs": trainer.state.epoch,
        }

    def _get_task_config(self, task: str) -> Dict:
        """Get task configuration with overrides applied."""
        from benchmark.common import get_task_config
        return get_task_config(task, self.task_configs, self.config.task_overrides)


# Convenience functions


def train_variant(
    cfg,
    variant_name: str,
    tasks: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    ncals: Optional[List[int]] = None,
    skip_existing: bool = True,
    parallel: bool = False,
) -> Dict[str, Any]:
    """Train a variant for benchmark.

    Args:
        cfg: Hydra configuration
        variant_name: Name of the variant
        tasks: List of tasks (defaults to cfg.tasks)
        seeds: List of seeds (defaults to config)
        ncals: List of ncal values (defaults to config)
        skip_existing: Skip if model already exists
        parallel: Enable parallel execution

    Returns:
        Training results
    """
    trainer = VariantTrainer.from_config(cfg, variant_name)
    tasks = tasks or list(cfg.get("tasks", []))
    return trainer.run(tasks, seeds, ncals, skip_existing, parallel)


def train_variant_single(
    cfg,
    variant_name: str,
    task: str,
    seed: int,
    ncal: int,
    skip_existing: bool = True,
    parallel: bool = False,
) -> Dict[str, Any]:
    """Train a single variant configuration.

    Useful for cluster array jobs.

    Args:
        cfg: Hydra configuration
        variant_name: Name of the variant
        task: Task name
        seed: Random seed
        ncal: Number of calibration samples
        skip_existing: Skip if model already exists
        parallel: Enable parallel execution

    Returns:
        Training result
    """
    trainer = VariantTrainer.from_config(cfg, variant_name)
    return trainer.run_single(task, seed, ncal, skip_existing, parallel)


def run_variant_hpo(
    cfg,
    variant_name: str,
    tasks: Optional[List[str]] = None,
    n_trials: int = 50,
    seed: int = 0,
    ncal: int = 200,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
) -> Dict[str, Any]:
    """Run HPO for a variant.

    Args:
        cfg: Hydra configuration
        variant_name: Name of the variant
        tasks: List of tasks (defaults to cfg.tasks)
        n_trials: Number of HPO trials
        seed: Seed to use for HPO
        ncal: Calibration size for HPO
        study_name: Optional Optuna study name
        storage: Optional Optuna storage URL

    Returns:
        HPO results
    """
    trainer = VariantTrainer.from_config(cfg, variant_name)
    tasks = tasks or list(cfg.get("tasks", []))
    return trainer.run_hpo(tasks, n_trials, seed, ncal, study_name, storage)
