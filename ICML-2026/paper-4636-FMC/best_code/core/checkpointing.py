"""Checkpoint management for training and evaluation.

This module provides utilities for saving and loading model checkpoints
with standardized path conventions.
"""

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn


def _to_serializable(obj):
    """Convert OmegaConf containers and other non-serializable types to plain Python."""
    try:
        from omegaconf import DictConfig, ListConfig
        if isinstance(obj, (DictConfig, ListConfig)):
            from omegaconf import OmegaConf
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


@dataclass
class ResultsPath:
    """Manages paths within the results directory structure.

    Directory structure:
        results/{experiment}/
            shared/                              # Phase 1: Train once
                data/{task}/
                    data_split.json
                    calibration_pool.pt          # (theta, x, y) - n_pool samples
                    test_data.pt                 # (theta, x, y) - n_test samples
                    simulations.pt               # (theta_sim, x_sim) for NPE/FMPE

                simulation/{task}/
                    npe/model.pkl, config.json, training_log.json
                    fmpe/model.pkl, config.json, training_log.json

                baselines/{task}/seed_{seed}/
                    dpe/ncal_{ncal}/model.pkl, metrics.json
                    mf_npe/ncal_{ncal}/model.pkl, metrics.json

            variants/{variant_name}/             # Phase 2: Iterate methods
                config.yaml                      # Full variant config
                calibration/{task}/seed_{seed}/
                    {method}/ncal_{ncal}/
                        model.pkl
                        training_log.json
                evaluation/{task}/seed_{seed}/
                    {method}/ncal_{ncal}/
                        samples.pt
                        metrics.json

            aggregated/                          # Phase 3: Unified results
                {task}/
                    all_methods_metrics.json
                    comparison_table.csv
    """

    results_dir: Path
    experiment_name: str

    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
        self.base_path = self.results_dir / self.experiment_name

    @property
    def experiment_path(self) -> Path:
        return self.base_path

    # =========================================================================
    # Benchmark paths (shared/variants structure)
    # =========================================================================

    # --- Shared paths (Phase 1: foundation training) ---
    @property
    def shared_path(self) -> Path:
        """Base path for shared foundation models and data."""
        return self.base_path / "shared"

    def shared_data_path(self, task: str) -> Path:
        """Path for shared data (splits, calibration pool, test data)."""
        return self.shared_path / "data" / task

    def shared_simulation_path(self, task: str) -> Path:
        """Path for shared simulation models directory."""
        return self.shared_path / "simulation" / task

    def shared_simulation_model_path(self, task: str, model_type: str) -> Path:
        """Path for shared simulation model (npe, fmpe)."""
        return self.shared_simulation_path(task) / model_type

    def shared_baselines_path(self, task: str, seed: int) -> Path:
        """Path for shared baselines directory."""
        return self.shared_path / "baselines" / task / f"seed_{seed}"

    def shared_baseline_model_path(
        self, task: str, seed: int, baseline: str, ncal: int
    ) -> Path:
        """Path for shared baseline model."""
        return self.shared_baselines_path(task, seed) / baseline / f"ncal_{ncal}"

    # --- Variant paths (Phase 2: variant training) ---
    def variant_path(self, variant_name: str) -> Path:
        """Base path for a specific variant."""
        return self.base_path / "variants" / variant_name

    def variant_config_path(self, variant_name: str) -> Path:
        """Path for variant configuration file."""
        return self.variant_path(variant_name) / "config.yaml"

    def variant_calibration_path(
        self, variant_name: str, task: str, seed: int
    ) -> Path:
        """Path for variant calibration models directory."""
        return self.variant_path(variant_name) / "calibration" / task / f"seed_{seed}"

    def variant_calibration_model_path(
        self, variant_name: str, task: str, seed: int, method: str, ncal: int
    ) -> Path:
        """Path for variant calibration model."""
        return (
            self.variant_calibration_path(variant_name, task, seed)
            / method
            / f"ncal_{ncal}"
        )

    def variant_evaluation_path(
        self, variant_name: str, task: str, seed: int
    ) -> Path:
        """Path for variant evaluation results."""
        return self.variant_path(variant_name) / "evaluation" / task / f"seed_{seed}"

    def variant_samples_path(
        self, variant_name: str, task: str, seed: int, method: str, ncal: int
    ) -> Path:
        """Path for variant samples."""
        return (
            self.variant_evaluation_path(variant_name, task, seed)
            / method
            / f"ncal_{ncal}"
        )

    def variant_metrics_path(
        self, variant_name: str, task: str, seed: int, method: str, ncal: int
    ) -> Path:
        """Path for variant metrics."""
        return self.variant_samples_path(variant_name, task, seed, method, ncal)

    # --- Aggregated paths (Phase 3: unified results) ---
    def benchmark_aggregated_path(self) -> Path:
        """Base path for aggregated benchmark results."""
        return self.base_path / "aggregated"

    def benchmark_aggregated_task_path(self, task: str) -> Path:
        """Path for task-specific aggregated results."""
        return self.benchmark_aggregated_path() / task

    # --- Locks directory ---
    @property
    def locks_path(self) -> Path:
        """Path for lock files used in parallel execution."""
        return self.base_path / ".locks"


@dataclass
class TrainingLog:
    """Training log for a model."""

    model_type: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    epochs_completed: int = 0
    best_val_loss: Optional[float] = None
    final_train_loss: Optional[float] = None
    training_time_seconds: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    loss_history: Dict[str, List[float]] = field(default_factory=dict)

    def mark_completed(self, best_val_loss: float, final_train_loss: float):
        self.completed_at = datetime.now().isoformat()
        self.best_val_loss = best_val_loss
        self.final_train_loss = final_train_loss

    def to_dict(self) -> Dict[str, Any]:
        # Ensure config is serializable
        config = self.config
        extra = self.extra
        try:
            from omegaconf import OmegaConf, DictConfig
            if isinstance(config, DictConfig):
                config = OmegaConf.to_container(config, resolve=True)
            if isinstance(extra, DictConfig):
                extra = OmegaConf.to_container(extra, resolve=True)
        except ImportError:
            pass
        return {
            "model_type": self.model_type,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "epochs_completed": self.epochs_completed,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.final_train_loss,
            "training_time_seconds": self.training_time_seconds,
            "config": config,
            "extra": extra,
            "loss_history": self.loss_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingLog":
        return cls(**data)


class CheckpointManager:
    """Manages saving and loading of model checkpoints.

    Provides a unified interface for checkpoint operations with
    standardized file naming and metadata tracking.
    """

    def __init__(self, paths: ResultsPath):
        self.paths = paths

    def _ensure_serializable(self, config) -> Dict:
        """Ensure config is JSON serializable (convert DictConfig to dict recursively)."""
        if config is None:
            return {}
        try:
            from omegaconf import OmegaConf, DictConfig, ListConfig
            if isinstance(config, (DictConfig, ListConfig)):
                return OmegaConf.to_container(config, resolve=True)
        except ImportError:
            pass
        # Recursively convert nested dicts
        if isinstance(config, dict):
            return {k: self._ensure_serializable(v) for k, v in config.items()}
        if isinstance(config, list):
            return [self._ensure_serializable(v) for v in config]
        return config

    # --- Private helpers for save/load patterns ---

    def _save_pickle_model(
        self,
        model_dir: Path,
        model: Any,
        config: Dict[str, Any],
        training_log: Optional[TrainingLog] = None,
        save_state_dict: bool = False,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a model as pickle with config and optional training log/metrics."""
        model_dir.mkdir(parents=True, exist_ok=True)

        if save_state_dict:
            torch.save(model.state_dict(), model_dir / "model.pth")

        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        config = self._ensure_serializable(config)
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        if training_log:
            log_data = self._ensure_serializable(training_log.to_dict())
            with open(model_dir / "training_log.json", "w") as f:
                json.dump(log_data, f, indent=2, default=str)

        if metrics:
            with open(model_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        return model_path

    def _load_pickle_model(
        self, model_dir: Path, device: torch.device = torch.device("cpu")
    ) -> Optional[Any]:
        """Load a pickle model, moving to device if supported."""
        pkl_path = model_dir / "model.pkl"
        if not pkl_path.exists():
            return None

        with open(pkl_path, "rb") as f:
            model = pickle.load(f)

        if hasattr(model, "to"):
            model.to(device)
        return model

    def _pickle_model_exists(self, model_dir: Path) -> bool:
        """Check if a pickle model exists in the given directory."""
        return (model_dir / "model.pkl").exists()

    def _save_json(self, path: Path, data: Dict[str, Any]) -> Path:
        """Save data as JSON, creating parent directories."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load JSON data from path, returning None if not found."""
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    # --- Experiment-level operations ---
    def save_experiment_config(
        self,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save experiment configuration at the experiment root.

        Args:
            config: Full experiment configuration dict
            metadata: Optional metadata (git commit, cli args, etc.)

        Returns:
            Path to saved config file
        """
        exp_path = self.paths.experiment_path
        exp_path.mkdir(parents=True, exist_ok=True)

        # Merge metadata into config
        if metadata:
            config = {**config, "_reproduction": metadata}

        config_path = exp_path / "experiment.yaml"

        # Use OmegaConf for YAML serialization if available
        try:
            from omegaconf import OmegaConf
            OmegaConf.save(OmegaConf.create(config), config_path)
        except ImportError:
            import yaml
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

        return config_path

    def load_experiment_config(self) -> Optional[Dict[str, Any]]:
        """Load experiment configuration from the experiment root.

        Returns:
            Experiment config dict or None if not found
        """
        config_path = self.paths.experiment_path / "experiment.yaml"

        if not config_path.exists():
            return None

        try:
            from omegaconf import OmegaConf
            return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        except ImportError:
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)

    def experiment_config_exists(self) -> bool:
        """Check if experiment config exists."""
        return (self.paths.experiment_path / "experiment.yaml").exists()

    def save_hpo_results(
        self,
        task: str,
        model: str,
        results: Dict[str, Any],
    ) -> Path:
        """Save HPO results for a task/model combination.

        Args:
            task: Task name
            model: Model name
            results: HPO results dict

        Returns:
            Path to saved results
        """
        hpo_path = self.paths.experiment_path / task / "hpo" / model
        hpo_path.mkdir(parents=True, exist_ok=True)

        results_path = hpo_path / "hpo_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        return results_path

    def load_hpo_results(self, task: str, model: str) -> Optional[Dict[str, Any]]:
        """Load HPO results for a task/model combination.

        Returns:
            HPO results dict or None if not found
        """
        results_path = self.paths.experiment_path / task / "hpo" / model / "hpo_results.json"

        if not results_path.exists():
            return None

        with open(results_path) as f:
            return json.load(f)

    # =========================================================================
    # Benchmark checkpoint operations (shared/variants structure)
    # =========================================================================

    # --- Shared data operations ---
    def save_shared_data(
        self,
        task: str,
        calibration_pool: Dict[str, torch.Tensor],
        test_data: Dict[str, torch.Tensor],
        simulations: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Path:
        """Save shared data for benchmark experiments.

        Args:
            task: Task name
            calibration_pool: Dict with theta, x, y tensors (n_pool samples)
            test_data: Dict with theta, x, y tensors (n_test samples)
            simulations: Optional dict with theta_sim, x_sim (for NPE/FMPE training)

        Returns:
            Path to data directory
        """
        data_dir = self.paths.shared_data_path(task)
        data_dir.mkdir(parents=True, exist_ok=True)

        torch.save(calibration_pool, data_dir / "calibration_pool.pt")
        torch.save(test_data, data_dir / "test_data.pt")

        if simulations is not None:
            torch.save(simulations, data_dir / "simulations.pt")

        return data_dir

    def load_shared_calibration_pool(
        self, task: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Load shared calibration pool data."""
        data_path = self.paths.shared_data_path(task) / "calibration_pool.pt"
        if not data_path.exists():
            return None
        return torch.load(data_path, weights_only=False)

    def load_shared_test_data(
        self, task: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Load shared test data."""
        data_path = self.paths.shared_data_path(task) / "test_data.pt"
        if not data_path.exists():
            return None
        return torch.load(data_path, weights_only=False)

    def load_shared_simulations(
        self, task: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Load shared simulation data for NPE/FMPE training."""
        data_path = self.paths.shared_data_path(task) / "simulations.pt"
        if not data_path.exists():
            return None
        return torch.load(data_path, weights_only=False)

    def shared_data_exists(self, task: str) -> bool:
        """Check if shared data exists for a task."""
        data_dir = self.paths.shared_data_path(task)
        return (
            (data_dir / "calibration_pool.pt").exists()
            and (data_dir / "test_data.pt").exists()
        )

    # --- Shared simulation model operations ---
    def save_shared_simulation_model(
        self,
        model: nn.Module,
        task: str,
        model_type: str,
        config: Dict[str, Any],
        training_log: Optional[TrainingLog] = None,
    ) -> Path:
        """Save a shared simulation model (NPE or FMPE)."""
        model_dir = self.paths.shared_simulation_model_path(task, model_type)
        return self._save_pickle_model(
            model_dir, model, config, training_log, save_state_dict=True
        )

    def load_shared_simulation_model(
        self,
        task: str,
        model_type: str,
        device: torch.device = torch.device("cpu"),
    ) -> Optional[nn.Module]:
        """Load a shared simulation model."""
        model_dir = self.paths.shared_simulation_model_path(task, model_type)
        return self._load_pickle_model(model_dir, device)

    def shared_simulation_model_exists(self, task: str, model_type: str) -> bool:
        """Check if a shared simulation model checkpoint exists."""
        model_dir = self.paths.shared_simulation_model_path(task, model_type)
        return self._pickle_model_exists(model_dir)

    # --- Shared baseline operations ---
    def save_shared_baseline(
        self,
        model: Any,
        task: str,
        seed: int,
        baseline: str,
        ncal: int,
        config: Dict[str, Any],
        training_log: Optional[TrainingLog] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a shared baseline checkpoint with optional metrics."""
        baseline_dir = self.paths.shared_baseline_model_path(task, seed, baseline, ncal)
        return self._save_pickle_model(
            baseline_dir, model, config, training_log, metrics=metrics
        )

    def load_shared_baseline(
        self,
        task: str,
        seed: int,
        baseline: str,
        ncal: int,
        device: torch.device = torch.device("cpu"),
    ) -> Optional[Any]:
        """Load a shared baseline checkpoint."""
        baseline_dir = self.paths.shared_baseline_model_path(task, seed, baseline, ncal)
        return self._load_pickle_model(baseline_dir, device)

    def shared_baseline_exists(
        self, task: str, seed: int, baseline: str, ncal: int
    ) -> bool:
        """Check if a shared baseline checkpoint exists."""
        baseline_dir = self.paths.shared_baseline_model_path(task, seed, baseline, ncal)
        return self._pickle_model_exists(baseline_dir)

    def load_shared_baseline_metrics(
        self, task: str, seed: int, baseline: str, ncal: int
    ) -> Optional[Dict[str, Any]]:
        """Load shared baseline metrics."""
        metrics_path = (
            self.paths.shared_baseline_model_path(task, seed, baseline, ncal)
            / "metrics.json"
        )
        return self._load_json(metrics_path)

    # --- Variant config operations ---
    def save_variant_config(
        self, variant_name: str, config: Dict[str, Any]
    ) -> Path:
        """Save variant configuration.

        Args:
            variant_name: Variant name
            config: Full variant configuration

        Returns:
            Path to saved config
        """
        variant_dir = self.paths.variant_path(variant_name)
        variant_dir.mkdir(parents=True, exist_ok=True)

        config_path = self.paths.variant_config_path(variant_name)

        try:
            from omegaconf import OmegaConf
            OmegaConf.save(OmegaConf.create(config), config_path)
        except ImportError:
            import yaml
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

        return config_path

    def load_variant_config(
        self, variant_name: str
    ) -> Optional[Dict[str, Any]]:
        """Load variant configuration."""
        config_path = self.paths.variant_config_path(variant_name)

        if not config_path.exists():
            return None

        try:
            from omegaconf import OmegaConf
            return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        except ImportError:
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)

    def variant_exists(self, variant_name: str) -> bool:
        """Check if a variant exists."""
        return self.paths.variant_config_path(variant_name).exists()

    # --- Variant calibration model operations ---
    def save_variant_calibration_model(
        self,
        model: Any,
        variant_name: str,
        task: str,
        seed: int,
        method: str,
        ncal: int,
        config: Dict[str, Any],
        training_log: Optional[TrainingLog] = None,
    ) -> Path:
        """Save a variant calibration model."""
        model_dir = self.paths.variant_calibration_model_path(
            variant_name, task, seed, method, ncal
        )
        return self._save_pickle_model(model_dir, model, config, training_log)

    def load_variant_calibration_model(
        self,
        variant_name: str,
        task: str,
        seed: int,
        method: str,
        ncal: int,
        device: torch.device = torch.device("cpu"),
    ) -> Optional[Any]:
        """Load a variant calibration model."""
        model_dir = self.paths.variant_calibration_model_path(
            variant_name, task, seed, method, ncal
        )
        return self._load_pickle_model(model_dir, device)

    def variant_calibration_model_exists(
        self, variant_name: str, task: str, seed: int, method: str, ncal: int
    ) -> bool:
        """Check if a variant calibration model exists."""
        model_dir = self.paths.variant_calibration_model_path(
            variant_name, task, seed, method, ncal
        )
        return self._pickle_model_exists(model_dir)

    # --- Variant evaluation operations ---
    def save_variant_samples(
        self,
        samples: torch.Tensor,
        variant_name: str,
        task: str,
        seed: int,
        method: str,
        ncal: int,
    ) -> Path:
        """Save variant samples."""
        samples_dir = self.paths.variant_samples_path(
            variant_name, task, seed, method, ncal
        )
        samples_dir.mkdir(parents=True, exist_ok=True)

        samples_path = samples_dir / "samples.pt"
        torch.save(samples, samples_path)
        return samples_path

    def load_variant_samples(
        self,
        variant_name: str,
        task: str,
        seed: int,
        method: str,
        ncal: int,
    ) -> Optional[torch.Tensor]:
        """Load variant samples."""
        samples_path = (
            self.paths.variant_samples_path(variant_name, task, seed, method, ncal)
            / "samples.pt"
        )
        if not samples_path.exists():
            return None
        return torch.load(samples_path, weights_only=False)

    def save_variant_metrics(
        self,
        metrics: Dict[str, Any],
        variant_name: str,
        task: str,
        seed: int,
        method: str,
        ncal: int,
    ) -> Path:
        """Save variant metrics."""
        metrics_path = (
            self.paths.variant_metrics_path(variant_name, task, seed, method, ncal)
            / "metrics.json"
        )
        return self._save_json(metrics_path, metrics)

    def load_variant_metrics(
        self,
        variant_name: str,
        task: str,
        seed: int,
        method: str,
        ncal: int,
    ) -> Optional[Dict[str, Any]]:
        """Load variant metrics."""
        metrics_path = (
            self.paths.variant_metrics_path(variant_name, task, seed, method, ncal)
            / "metrics.json"
        )
        return self._load_json(metrics_path)

    def variant_metrics_exist(
        self, variant_name: str, task: str, seed: int, method: str, ncal: int
    ) -> bool:
        """Check if variant metrics exist."""
        metrics_path = (
            self.paths.variant_metrics_path(variant_name, task, seed, method, ncal)
            / "metrics.json"
        )
        return metrics_path.exists()

    # --- Aggregated benchmark results ---
    def save_benchmark_aggregated_metrics(
        self,
        task: str,
        metrics: Dict[str, Any],
        filename: str = "all_methods_metrics.json",
    ) -> Path:
        """Save aggregated benchmark metrics for a task."""
        metrics_path = self.paths.benchmark_aggregated_task_path(task) / filename
        return self._save_json(metrics_path, metrics)

    def load_benchmark_aggregated_metrics(
        self, task: str, filename: str = "all_methods_metrics.json"
    ) -> Optional[Dict[str, Any]]:
        """Load aggregated benchmark metrics for a task."""
        metrics_path = self.paths.benchmark_aggregated_task_path(task) / filename
        return self._load_json(metrics_path)

    def save_comparison_table(
        self, task: str, table_data: Any, filename: str = "comparison_table.csv"
    ) -> Path:
        """Save comparison table for a task.

        Args:
            task: Task name
            table_data: DataFrame or dict to save as CSV
            filename: Output filename

        Returns:
            Path to saved table
        """
        agg_dir = self.paths.benchmark_aggregated_task_path(task)
        agg_dir.mkdir(parents=True, exist_ok=True)

        table_path = agg_dir / filename

        # Support both pandas DataFrame and dict
        if hasattr(table_data, "to_csv"):
            table_data.to_csv(table_path, index=False)
        else:
            import csv
            with open(table_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=table_data[0].keys())
                writer.writeheader()
                writer.writerows(table_data)

        return table_path

    # --- Utility methods for benchmark ---
    def list_variants(self) -> List[str]:
        """List all variants in the experiment."""
        variants_dir = self.paths.base_path / "variants"
        if not variants_dir.exists():
            return []
        return [d.name for d in variants_dir.iterdir() if d.is_dir()]

    def list_shared_baselines(self, task: str, seed: int) -> List[str]:
        """List all shared baselines for a task/seed."""
        baselines_dir = self.paths.shared_baselines_path(task, seed)
        if not baselines_dir.exists():
            return []
        return [d.name for d in baselines_dir.iterdir() if d.is_dir()]

    def list_shared_simulation_models(self, task: str) -> List[str]:
        """List all shared simulation models for a task."""
        sim_dir = self.paths.shared_simulation_path(task)
        if not sim_dir.exists():
            return []
        return [d.name for d in sim_dir.iterdir() if d.is_dir()]
