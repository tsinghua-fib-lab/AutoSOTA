"""Foundation training module for benchmark experiments (Phase 1).

This module handles training simulation models (NPE, FMPE) and baselines once,
saving them to the shared/ directory for use by all variants.

Usage:
    trainer = FoundationTrainer.from_config(cfg)
    trainer.run(tasks=["pendulum", "gaussian"])
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from core.checkpointing import CheckpointManager, ResultsPath, TrainingLog
from simulator import get_simulator, generate_simulation_dataset, generate_calibration_dataset
from simulator.base import Simulator
from utils.data_split import (
    BenchmarkDataSplit,
    BenchmarkSplitter,
    DEFAULT_N_POOL,
    DEFAULT_N_TEST,
    DEFAULT_CALIBRATION_SIZES,
)
from utils.locking import (
    DataGenerationLock,
    SimulationModelLock,
    TrainingLock,
    atomic_write,
)


@dataclass
class FoundationConfig:
    """Configuration for foundation training."""

    # Data protocol
    n_pool: int = DEFAULT_N_POOL
    n_test: int = DEFAULT_N_TEST
    n_sim: Optional[int] = None  # If set, overrides task config's num_samples
    calibration_sizes: List[int] = field(default_factory=lambda: DEFAULT_CALIBRATION_SIZES.copy())
    data_seed: int = 42

    # Training
    simulation_models: List[str] = field(default_factory=lambda: ["npe", "fmpe"])
    baselines: List[str] = field(default_factory=lambda: ["dpe", "mf_npe"])
    seeds: List[int] = field(default_factory=lambda: [33, 43, 53])

    # Paths
    results_dir: Path = Path("results")
    experiment_name: str = "comparison_benchmark"
    data_dir: Optional[str] = None  # Base directory for file-based simulators

    # Parallel execution
    use_locks: bool = True
    lock_timeout: float = 600.0  # 10 minutes

    # Task overrides - applied to all task configs
    task_overrides: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg) -> "FoundationConfig":
        """Create from Hydra config."""
        data_cfg = cfg.get("data", {})
        foundation_cfg = cfg.get("foundation", {})

        # Extract task_overrides from config
        task_overrides = cfg.get("task_overrides", {})
        if hasattr(task_overrides, "items"):
            # Convert OmegaConf to dict if needed
            task_overrides = dict(task_overrides)
        else:
            task_overrides = {}

        return cls(
            n_pool=data_cfg.get("n_pool", DEFAULT_N_POOL),
            n_test=data_cfg.get("n_test", DEFAULT_N_TEST),
            n_sim=data_cfg.get("n_sim"),  # None if not specified, will use task config
            calibration_sizes=list(data_cfg.get("calibration_sizes", DEFAULT_CALIBRATION_SIZES)),
            data_seed=data_cfg.get("seed", 42),
            simulation_models=list(foundation_cfg.get("simulation_models", ["npe", "fmpe"])),
            baselines=list(foundation_cfg.get("baselines", ["dpe", "mf_npe"])),
            # Seeds from experiment config (no fallback to config.yaml)
            seeds=list(cfg.get("seeds", [33, 43, 53])),
            results_dir=Path(cfg.get("results_dir", "results")),
            experiment_name=cfg.get("experiment", {}).get("name", "comparison_benchmark"),
            data_dir=cfg.get("data_dir"),
            use_locks=cfg.get("parallel", {}).get("use_locks", True),
            lock_timeout=cfg.get("parallel", {}).get("lock_timeout", 600.0),
            task_overrides=task_overrides,
        )


class FoundationTrainer:
    """Handles Phase 1: Foundation training of simulation models and baselines."""

    def __init__(
        self,
        config: FoundationConfig,
        task_configs: Optional[Dict[str, Dict]] = None,
    ):
        """Initialize foundation trainer.

        Args:
            config: Foundation configuration
            task_configs: Optional task-specific configs (loaded from Hydra otherwise)
        """
        self.config = config
        self.task_configs = task_configs or {}

        # Setup paths and checkpoint manager
        self.paths = ResultsPath(config.results_dir, config.experiment_name)
        self.checkpoint_manager = CheckpointManager(self.paths)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_config(cls, cfg) -> "FoundationTrainer":
        """Create trainer from Hydra config."""
        foundation_config = FoundationConfig.from_config(cfg)

        # Extract task configs
        task_configs = {}
        if "task" in cfg:
            # Single task mode
            task_name = cfg.task.get("name", cfg.get("task_name", "unknown"))
            task_configs[task_name] = dict(cfg.task)
        elif "tasks" in cfg:
            # Multi-task mode - load task configs
            for task_name in cfg.tasks:
                if hasattr(cfg, task_name):
                    task_configs[task_name] = dict(cfg[task_name])

        return cls(foundation_config, task_configs)

    def run(
        self,
        tasks: List[str],
        skip_existing: bool = True,
        parallel: bool = False,
    ) -> Dict[str, Any]:
        """Run foundation training for all tasks.

        Args:
            tasks: List of task names to train
            skip_existing: Skip training if model already exists
            parallel: Enable parallel execution with locks

        Returns:
            Summary of training results
        """
        results = {}

        for task in tasks:
            print(f"\n{'=' * 60}")
            print(f"Foundation training for task: {task}")
            print(f"{'=' * 60}")

            task_results = self._train_task(task, skip_existing, parallel)
            results[task] = task_results

        return results

    def _train_task(
        self,
        task: str,
        skip_existing: bool,
        parallel: bool,
    ) -> Dict[str, Any]:
        """Train foundation for a single task."""
        results = {
            "data": None,
            "simulation_models": {},
            "baselines": {},
        }

        # Step 1: Generate and save data
        print(f"\n[1/3] Generating data for {task}...")
        data_result = self._generate_data(task, skip_existing, parallel)
        results["data"] = data_result

        # Step 2: Train simulation models
        print(f"\n[2/3] Training simulation models for {task}...")
        for model_type in self.config.simulation_models:
            model_result = self._train_simulation_model(task, model_type, skip_existing, parallel)
            results["simulation_models"][model_type] = model_result

        # Step 3: Train baselines
        print(f"\n[3/3] Training baselines for {task}...")
        print(f"Seeds: {self.config.seeds}, Calibration sizes: {self.config.calibration_sizes}")
        for seed in self.config.seeds:
            results["baselines"][seed] = {}
            for baseline in self.config.baselines:
                for ncal in self.config.calibration_sizes:
                    baseline_result = self._train_baseline(
                        task, seed, baseline, ncal, skip_existing, parallel
                    )
                    key = f"{baseline}_ncal{ncal}"
                    results["baselines"][seed][key] = baseline_result

        return results

    def _generate_data(
        self,
        task: str,
        skip_existing: bool,
        parallel: bool,
    ) -> Dict[str, Any]:
        """Generate and save data for a task."""
        data_path = self.paths.shared_data_path(task)

        # Check if data already exists
        if skip_existing and self.checkpoint_manager.shared_data_exists(task):
            print(f"  Data already exists for {task} in {str(data_path)}, skipping...")
            return {"status": "skipped", "path": str(data_path)}

        # Lock if parallel mode
        locks_dir = self.paths.locks_path
        check_exists = lambda: self.checkpoint_manager.shared_data_exists(task)

        if parallel and self.config.use_locks:
            with DataGenerationLock(locks_dir, task, check_exists=check_exists) as lock:
                if lock.already_exists:
                    return {"status": "skipped", "path": str(data_path)}
                return self._do_generate_data(task, data_path)
        else:
            return self._do_generate_data(task, data_path)

    def _do_generate_data(self, task: str, data_path: Path) -> Dict[str, Any]:
        """Actually generate data (called within or without lock)."""
        # Get simulator
        task_config = self._get_task_config(task)
        simulator_config = task_config.get("simulator", {})
        simulator_params = dict(simulator_config.get("params", {}))
        simulator_name = simulator_config.get("name", task)
        # Pass data_dir if configured (for file-based simulators)
        if self.config.data_dir is not None:
            try:
                simulator = get_simulator(
                    simulator_name, data_dir=self.config.data_dir, **simulator_params
                )
            except TypeError:
                # Simulator doesn't accept data_dir (not file-based)
                simulator = get_simulator(simulator_name, **simulator_params)
        else:
            simulator = get_simulator(simulator_name, **simulator_params)

        # For file-based tasks, check available observations
        n_pool = self.config.n_pool
        n_test = self.config.n_test
        calibration_sizes = self.config.calibration_sizes.copy()
        if not (simulator.callable_simulator and simulator.callable_dgp):
            # File-based task - check total available
            total_available = simulator.get_total_observations()
            required = n_pool + n_test
            if total_available < required:
                # Keep n_pool at config value if possible, reduce n_test
                if total_available >= n_pool:
                    n_test = total_available - n_pool
                else:
                    # Not enough even for n_pool, split proportionally
                    n_pool = total_available // 2
                    n_test = total_available - n_pool
                # Also filter calibration_sizes to not exceed n_pool
                calibration_sizes = [c for c in calibration_sizes if c <= n_pool]
                if not calibration_sizes:
                    calibration_sizes = [n_pool]
                print(f"  Note: {task} has {total_available} observations (requested {required})")
                print(
                    f"  Using n_pool={n_pool}, n_test={n_test}, calibration_sizes={calibration_sizes}"
                )

        # Create data split
        splitter = BenchmarkSplitter(
            n_pool=n_pool,
            n_test=n_test,
            calibration_sizes=calibration_sizes,
            seed=self.config.data_seed,
        )
        split = splitter.get_or_create_split(data_path)

        # Generate data based on task type
        if simulator.callable_simulator and simulator.callable_dgp:
            # Fully generative task
            data = self._generate_generative_task_data(simulator, split, task_config)
        else:
            # File-based task
            data = self._generate_file_based_task_data(simulator, split, task_config)

        # Save data
        self.checkpoint_manager.save_shared_data(
            task=task,
            calibration_pool=data["calibration_pool"],
            test_data=data["test_data"],
            simulations=data.get("simulations"),
        )

        print(f"  Generated data for {task}:")
        print(f"    Calibration pool: {data['calibration_pool']['theta'].shape[0]} samples")
        print(f"    Test pool: {data['test_data']['theta'].shape[0]} samples")
        if "simulations" in data:
            print(f"    Simulations: {data['simulations']['theta'].shape[0]} samples")

        return {"status": "generated", "path": str(data_path)}

    def _generate_generative_task_data(
        self,
        simulator: Simulator,
        split: BenchmarkDataSplit,
        task_config: Dict,
    ) -> Dict[str, Dict[str, Tensor]]:
        """Generate data for fully generative task."""
        generation = task_config.get("generation", "independent")

        # Priority: data.n_sim (if specified) > task_config.num_samples > default 50000
        if self.config.n_sim is not None:
            n_sim = self.config.n_sim
        else:
            n_sim = task_config.get("num_samples", task_config.get("n_sim", 50000))

        # Generate simulation data for NPE/FMPE training
        theta_sim, x_sim = generate_simulation_dataset(simulator, n_sim)

        # Generate calibration pool
        theta_cal, x_cal, y_cal = generate_calibration_dataset(simulator, split.n_pool, generation)

        # Generate test data
        theta_test, x_test, y_test = generate_calibration_dataset(
            simulator, split.n_test, generation
        )

        return {
            "simulations": {"theta": theta_sim, "x": x_sim},
            "calibration_pool": {"theta": theta_cal, "x": x_cal, "y": y_cal},
            "test_data": {"theta": theta_test, "x": x_test, "y": y_test},
        }

    def _generate_file_based_task_data(
        self,
        simulator: Simulator,
        split: BenchmarkDataSplit,
        task_config: Dict,
    ) -> Dict[str, Dict[str, Tensor]]:
        """Generate data for file-based task."""
        generation = task_config.get("generation", "independent")

        # Get indices for calibration and test pools
        cal_indices = split.calibration_pool_indices
        test_indices = split.test_pool_indices

        # Load calibration pool from files
        theta_cal, x_cal, y_cal = generate_calibration_dataset(
            simulator,
            n=len(cal_indices),
            generation=generation,
            indices=tuple(cal_indices),
        )

        # Load test data from files
        theta_test, x_test, y_test = generate_calibration_dataset(
            simulator,
            n=len(test_indices),
            generation=generation,
            indices=tuple(test_indices),
        )

        # For file-based tasks, we may not have separate simulation data
        # In this case, simulation training uses the calibration pool
        # or we generate simulations if the simulator is callable
        simulations = None
        if simulator.callable_simulator:
            # Priority: data.n_sim (if specified) > task_config.num_samples > default 50000
            if self.config.n_sim is not None:
                n_sim = self.config.n_sim
            else:
                n_sim = task_config.get("num_samples", task_config.get("n_sim", 50000))
            theta_sim, x_sim = generate_simulation_dataset(simulator, n_sim)
            simulations = {"theta": theta_sim, "x": x_sim}

        result = {
            "calibration_pool": {"theta": theta_cal, "x": x_cal, "y": y_cal},
            "test_data": {"theta": theta_test, "x": x_test, "y": y_test},
        }
        if simulations is not None:
            result["simulations"] = simulations

        return result

    def _train_simulation_model(
        self,
        task: str,
        model_type: str,
        skip_existing: bool,
        parallel: bool,
    ) -> Dict[str, Any]:
        """Train a simulation model (NPE or FMPE)."""
        # Check if model already exists
        if skip_existing and self.checkpoint_manager.shared_simulation_model_exists(
            task, model_type
        ):
            print(f"  {model_type.upper()} model already exists for {task}, skipping...")
            return {"status": "skipped"}

        # Lock if parallel mode
        locks_dir = self.paths.locks_path
        check_exists = lambda: self.checkpoint_manager.shared_simulation_model_exists(
            task, model_type
        )

        if parallel and self.config.use_locks:
            with SimulationModelLock(
                locks_dir, task, model_type, check_exists=check_exists
            ) as lock:
                if lock.already_exists:
                    return {"status": "skipped"}
                return self._do_train_simulation_model(task, model_type)
        else:
            return self._do_train_simulation_model(task, model_type)

    def _do_train_simulation_model(self, task: str, model_type: str) -> Dict[str, Any]:
        """Actually train simulation model."""
        from training.registry import get_trainer_from_task_config
        from training.base import TrainerConfig

        # Load training data
        sim_data = self.checkpoint_manager.load_shared_simulations(task)
        if sim_data is None:
            # Try loading calibration pool as fallback
            cal_data = self.checkpoint_manager.load_shared_calibration_pool(task)
            if cal_data is None:
                raise ValueError(f"No training data found for {task}")
            theta_sim, x_sim = cal_data["theta"], cal_data["x"]
        else:
            theta_sim, x_sim = sim_data["theta"], sim_data["x"]

        # Get task config
        task_config = self._get_task_config(task)

        # Get trainer
        model_path = self.paths.shared_simulation_model_path(task, model_type)
        trainer = get_trainer_from_task_config(model_type, task_config, model_path)

        # Train model
        print(f"  Training {model_type.upper()} for {task}...")
        posterior = trainer.train(
            data=(theta_sim, x_sim),
            device=self.device,
            logname=model_type,
        )

        # Save model
        training_log = TrainingLog(
            model_type=model_type,
            epochs_completed=trainer.state.epoch,
            best_val_loss=trainer.state.best_val_loss,
            final_train_loss=trainer.state.best_train_loss,
            config=trainer.config.to_dict(),
            loss_history=trainer.state.loss_history,
        )
        training_log.mark_completed(
            best_val_loss=trainer.state.best_val_loss,
            final_train_loss=trainer.state.best_train_loss,
        )

        # The trainer already saves the model, but we also save with benchmark structure
        self.checkpoint_manager.save_shared_simulation_model(
            model=posterior,
            task=task,
            model_type=model_type,
            config=trainer.config.to_dict(),
            training_log=training_log,
        )

        print(
            f"  {model_type.upper()} training completed: val_loss={trainer.state.best_val_loss:.4f}"
        )

        return {
            "status": "trained",
            "val_loss": trainer.state.best_val_loss,
            "epochs": trainer.state.epoch,
        }

    def _train_baseline(
        self,
        task: str,
        seed: int,
        baseline: str,
        ncal: int,
        skip_existing: bool,
        parallel: bool,
    ) -> Dict[str, Any]:
        """Train a baseline method."""
        # Check if baseline already exists
        if skip_existing and self.checkpoint_manager.shared_baseline_exists(
            task, seed, baseline, ncal
        ):
            print(f"  {baseline} (seed={seed}, ncal={ncal}) already exists, skipping...")
            return {"status": "skipped"}

        # Lock if parallel mode
        locks_dir = self.paths.locks_path
        check_exists = lambda: self.checkpoint_manager.shared_baseline_exists(
            task, seed, baseline, ncal
        )

        if parallel and self.config.use_locks:
            with TrainingLock(
                locks_dir,
                task,
                seed,
                ncal,
                method=baseline,
                check_exists=check_exists,
            ) as lock:
                if lock.already_exists:
                    return {"status": "skipped"}
                return self._do_train_baseline(task, seed, baseline, ncal)
        else:
            return self._do_train_baseline(task, seed, baseline, ncal)

    def _do_train_baseline(
        self,
        task: str,
        seed: int,
        baseline: str,
        ncal: int,
    ) -> Dict[str, Any]:
        """Actually train baseline."""
        from training.registry import get_trainer_from_task_config

        # Load calibration data
        cal_data = self.checkpoint_manager.load_shared_calibration_pool(task)
        if cal_data is None:
            raise ValueError(f"No calibration data found for {task}")

        # Get calibration subset
        data_path = self.paths.shared_data_path(task)
        splitter = BenchmarkSplitter.from_config(
            {
                "data": {
                    "n_pool": self.config.n_pool,
                    "n_test": self.config.n_test,
                    "calibration_sizes": self.config.calibration_sizes,
                }
            }
        )
        split = splitter.load_split(data_path)

        # Get calibration indices for this seed
        cal_indices = split.get_calibration_indices(ncal, subsample_seed=seed)

        theta_cal = cal_data["theta"][cal_indices]
        x_cal = cal_data["x"][cal_indices]
        y_cal = cal_data["y"][cal_indices]

        # Get task config
        task_config = self._get_task_config(task)

        # Get trainer (skip if baseline not configured for this task)
        model_path = self.paths.shared_baseline_model_path(task, seed, baseline, ncal)
        try:
            trainer = get_trainer_from_task_config(baseline, task_config, model_path)
        except ValueError as e:
            print(f"  Skipping {baseline}: {e}")
            return {"status": "skipped", "reason": str(e)}

        # Set seed for reproducibility
        torch.manual_seed(seed)

        # Train baseline
        print(f"  Training {baseline} (seed={seed}, ncal={ncal})...")

        # Baselines use (theta, x, y) tuple format
        if baseline == "dpe":
            posterior = trainer.train(
                data=(theta_cal, x_cal, y_cal),
                device=self.device,
                logname=f"{baseline}_seed{seed}_ncal{ncal}",
            )
        elif baseline in ("mf_npe", "rope"):
            # MF-NPE and ROPE need the pretrained NPE model
            npe_model = self.checkpoint_manager.load_shared_simulation_model(
                task, "npe", self.device
            )
            if npe_model is None:
                raise ValueError(f"NPE model not found for {task}. Train simulation models first.")
            posterior = trainer.train(
                data=(theta_cal, x_cal, y_cal),
                device=self.device,
                logname=f"{baseline}_seed{seed}_ncal{ncal}",
                npe=npe_model,
            )
        else:
            # Generic baseline training
            posterior = trainer.train(
                data=(theta_cal, x_cal, y_cal),
                device=self.device,
                logname=f"{baseline}_seed{seed}_ncal{ncal}",
            )

        # Create training log
        training_log = TrainingLog(
            model_type=baseline,
            epochs_completed=trainer.state.epoch,
            best_val_loss=trainer.state.best_val_loss,
            final_train_loss=trainer.state.best_train_loss,
            config=trainer.config.to_dict(),
            extra={"seed": seed, "ncal": ncal},
            loss_history=trainer.state.loss_history,
        )
        training_log.mark_completed(
            best_val_loss=trainer.state.best_val_loss,
            final_train_loss=trainer.state.best_train_loss,
        )

        # Save baseline
        self.checkpoint_manager.save_shared_baseline(
            model=posterior,
            task=task,
            seed=seed,
            baseline=baseline,
            ncal=ncal,
            config=trainer.config.to_dict(),
            training_log=training_log,
        )

        print(f"  {baseline} training completed: val_loss={trainer.state.best_val_loss:.4f}")

        return {
            "status": "trained",
            "val_loss": trainer.state.best_val_loss,
            "epochs": trainer.state.epoch,
        }

    def _get_task_config(self, task: str) -> Dict:
        """Get task configuration with overrides applied."""
        from benchmark.common import get_task_config

        return get_task_config(task, self.task_configs, self.config.task_overrides)


# Convenience functions for direct use


def train_foundation(
    cfg,
    tasks: Optional[List[str]] = None,
    skip_existing: bool = True,
    parallel: bool = False,
) -> Dict[str, Any]:
    """Train foundation models for benchmark.

    Args:
        cfg: Hydra configuration
        tasks: List of tasks to train (defaults to cfg.tasks)
        skip_existing: Skip if models already exist
        parallel: Enable parallel execution

    Returns:
        Training results summary
    """
    trainer = FoundationTrainer.from_config(cfg)
    tasks = tasks or list(cfg.get("tasks", []))
    return trainer.run(tasks, skip_existing, parallel)


def generate_and_save_data(
    cfg,
    task: str,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """Generate and save data for a task.

    Args:
        cfg: Hydra configuration
        task: Task name
        skip_existing: Skip if data already exists

    Returns:
        Data generation result
    """
    trainer = FoundationTrainer.from_config(cfg)
    return trainer._generate_data(task, skip_existing, parallel=False)


def train_simulation_model(
    cfg,
    task: str,
    model_type: str,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """Train a simulation model.

    Args:
        cfg: Hydra configuration
        task: Task name
        model_type: Model type ('npe' or 'fmpe')
        skip_existing: Skip if model already exists

    Returns:
        Training result
    """
    trainer = FoundationTrainer.from_config(cfg)
    return trainer._train_simulation_model(task, model_type, skip_existing, parallel=False)


def train_baselines(
    cfg,
    task: str,
    seeds: Optional[List[int]] = None,
    baselines: Optional[List[str]] = None,
    ncals: Optional[List[int]] = None,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """Train baselines for a task.

    Args:
        cfg: Hydra configuration
        task: Task name
        seeds: List of seeds (defaults to config)
        baselines: List of baselines (defaults to config)
        ncals: List of ncal values (defaults to config)
        skip_existing: Skip if model already exists

    Returns:
        Training results
    """
    trainer = FoundationTrainer.from_config(cfg)

    seeds = seeds or trainer.config.seeds
    baselines = baselines or trainer.config.baselines
    ncals = ncals or trainer.config.calibration_sizes

    results = {}
    for seed in seeds:
        results[seed] = {}
        for baseline in baselines:
            for ncal in ncals:
                result = trainer._train_baseline(
                    task, seed, baseline, ncal, skip_existing, parallel=False
                )
                key = f"{baseline}_ncal{ncal}"
                results[seed][key] = result

    return results
