#!/usr/bin/env python3
"""Run a named experiment across multiple tasks.

This is the main entry point for running experiments. It orchestrates:
1. Loading experiment configuration
2. Applying profile scaling
3. Running training across all tasks (sequential)
4. Handling sweeps and HPO
5. Running evaluation

Usage:
    # Run experiment with development profile (default)
    python run_experiment.py +experiment=baseline_benchmark

    # Quick sanity check
    python run_experiment.py +experiment=baseline_benchmark profile=sanity

    # Full production run
    python run_experiment.py +experiment=baseline_benchmark profile=production

    # List available experiments
    python run_experiment.py list

    # Dry run (show config without executing)
    python run_experiment.py +experiment=baseline_benchmark dry_run=true

    # Reproduce from checkpoint
    python run_experiment.py from_checkpoint=./results/baseline_benchmark_production_20260122/

    # Override tasks
    python run_experiment.py +experiment=baseline_benchmark tasks=[pure_gaussian,pendulum]
"""

import sys
from pathlib import Path

# Handle special commands before Hydra
if len(sys.argv) > 1 and sys.argv[1] == "list":
    from utils.experiment import list_experiments, list_profiles

    print("\n" + "=" * 60)
    print("AVAILABLE EXPERIMENTS")
    print("=" * 60)

    experiments = list_experiments()
    if experiments:
        for name, desc in sorted(experiments.items()):
            print(f"  {name:30} {desc}")
    else:
        print("  No experiments found in configs/experiment/")

    print("\n" + "-" * 60)
    print("AVAILABLE PROFILES")
    print("-" * 60)

    profiles = list_profiles()
    if profiles:
        for name, desc in sorted(profiles.items()):
            print(f"  {name:15} {desc}")
    else:
        print("  No profiles found in configs/experiment/_profiles/")

    print("\n" + "=" * 60)
    sys.exit(0)

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from core.checkpointing import CheckpointManager, ResultsPath
from utils.experiment import (
    ExperimentConfig,
    expand_sweep,
    load_experiment_from_checkpoint,
    make_experiment_name,
    print_experiment_summary,
    resolve_experiment,
    resolve_profile,
    apply_profile,
    save_experiment_config,
    SweepPoint,
)


def setup_device() -> torch.device:
    """Setup compute device."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def run_simulation_training(
    exp: ExperimentConfig,
    task: str,
    sweep_point: SweepPoint,
    checkpoint_manager: CheckpointManager,
    device: torch.device,
    cfg: DictConfig,
    force: bool = False,
) -> None:
    """Run simulation model training for a task.

    Args:
        exp: Experiment configuration
        task: Task name
        sweep_point: Current sweep point
        checkpoint_manager: Checkpoint manager
        device: Compute device
        cfg: Full Hydra config
        force: Force retraining
    """
    from train_simulation import train_simulation_model, load_or_generate_data

    if not exp.run.simulation_models:
        return

    print(f"\n{'='*60}")
    print(f"SIMULATION TRAINING: {task}")
    if sweep_point.tag:
        print(f"Sweep point: {sweep_point.tag}")
    print(f"{'='*60}")

    # Load task config
    task_cfg = _get_task_config(cfg, task)

    # Apply sweep overrides
    if sweep_point.overrides:
        task_cfg = _apply_overrides(task_cfg, sweep_point.overrides)

    # Apply method overrides from experiment
    for model in exp.run.simulation_models:
        if model in exp.method_overrides:
            if model in task_cfg:
                task_cfg[model] = OmegaConf.merge(
                    task_cfg[model],
                    OmegaConf.create(exp.method_overrides[model]),
                )

    # Load or generate data
    theta_sim, x_sim, theta_cal, x_cal, y_cal = load_or_generate_data(
        cfg, checkpoint_manager, task
    )

    # Train each simulation model
    for model in exp.run.simulation_models:
        print(f"\nTraining {model.upper()}...")
        train_simulation_model(
            trainer_name=model,
            task=task_cfg,
            theta_sim=theta_sim,
            x_sim=x_sim,
            device=device,
            checkpoint_manager=checkpoint_manager,
            task_name=task,
            force=force,
        )


def run_calibration_training(
    exp: ExperimentConfig,
    task: str,
    sweep_point: SweepPoint,
    checkpoint_manager: CheckpointManager,
    device: torch.device,
    cfg: DictConfig,
    force: bool = False,
) -> None:
    """Run calibration method training for a task.

    Args:
        exp: Experiment configuration
        task: Task name
        sweep_point: Current sweep point
        checkpoint_manager: Checkpoint manager
        device: Compute device
        cfg: Full Hydra config
        force: Force retraining
    """
    if not exp.run.calibration_methods:
        return

    print(f"\n{'='*60}")
    print(f"CALIBRATION TRAINING: {task}")
    if sweep_point.tag:
        print(f"Sweep point: {sweep_point.tag}")
    print(f"{'='*60}")

    # Import training functions
    from train_calibration import train_calibration_method

    task_cfg = _get_task_config(cfg, task)

    # Apply sweep and method overrides
    if sweep_point.overrides:
        task_cfg = _apply_overrides(task_cfg, sweep_point.overrides)

    for method in exp.run.calibration_methods:
        if method in exp.method_overrides:
            if method in task_cfg:
                task_cfg[method] = OmegaConf.merge(
                    task_cfg[method],
                    OmegaConf.create(exp.method_overrides[method]),
                )

    # Train for each seed and ncal
    for seed in exp.run.seeds:
        for ncal in exp.run.num_cal:
            for method in exp.run.calibration_methods:
                print(f"\nTraining {method} (seed={seed}, ncal={ncal})...")
                # Note: This calls the training function - implementation depends on
                # how train_calibration.py exposes its training logic
                # For now, we print what would be trained
                print(f"  [Would train {method} on {task} with seed={seed}, ncal={ncal}]")


def run_baseline_training(
    exp: ExperimentConfig,
    task: str,
    sweep_point: SweepPoint,
    checkpoint_manager: CheckpointManager,
    device: torch.device,
    cfg: DictConfig,
    force: bool = False,
) -> None:
    """Run baseline training for a task."""
    if not exp.run.baselines:
        return

    print(f"\n{'='*60}")
    print(f"BASELINE TRAINING: {task}")
    print(f"{'='*60}")

    for seed in exp.run.seeds:
        for ncal in exp.run.num_cal:
            for baseline in exp.run.baselines:
                print(f"\nTraining {baseline} (seed={seed}, ncal={ncal})...")
                print(f"  [Would train {baseline} on {task} with seed={seed}, ncal={ncal}]")


def run_hpo(
    exp: ExperimentConfig,
    task: str,
    checkpoint_manager: CheckpointManager,
    device: torch.device,
    cfg: DictConfig,
) -> None:
    """Run Optuna HPO for a task.

    Args:
        exp: Experiment configuration
        task: Task name
        checkpoint_manager: Checkpoint manager
        device: Compute device
        cfg: Full Hydra config
    """
    from copy import deepcopy

    from omegaconf import OmegaConf

    from training import TrainerConfig, get_trainer
    from utils.optuna_utils import (
        apply_suggested_params,
        create_study,
        get_search_space,
        print_study_summary,
        save_study_results,
        suggest_params,
    )

    print(f"\n{'='*60}")
    print(f"HPO: {task}")
    print(f"Backend: {exp.hpo.backend}")
    print(f"Trials: {exp.hpo.n_trials}")
    print(f"{'='*60}")

    task_overrides = exp.task_overrides.get(task, {})

    # Load task config - we need the full task config for this task
    task_cfg = _load_task_config(cfg, task)
    task_cfg_dict = OmegaConf.to_container(task_cfg, resolve=True)

    # Load data once for all trials
    print(f"\nLoading data for {task}...")
    theta_sim, x_sim, theta_cal, x_cal, y_cal = _load_task_data(
        cfg, task, task_cfg, checkpoint_manager
    )
    print(f"Data loaded: theta={theta_sim.shape}, x={x_sim.shape}")

    # Run HPO for each model
    models_to_tune = exp.run.simulation_models + exp.run.calibration_methods
    for model in models_to_tune:
        if model not in exp.hpo.search_spaces:
            print(f"\nSkipping {model} - no search space defined")
            continue

        print(f"\n{'='*50}")
        print(f"Tuning {model}...")
        print(f"{'='*50}")

        # Get search space
        search_space = get_search_space(exp.hpo, model, task_overrides)
        if not search_space:
            print(f"  No search space defined for {model}, skipping")
            continue

        print(f"Search space: {list(search_space.keys())}")

        # Get base method config
        base_method_config = task_cfg_dict.get(model, {})

        # Apply experiment-level method overrides
        if model in exp.method_overrides:
            base_method_config = apply_suggested_params(
                base_method_config, _flatten_dict(exp.method_overrides[model])
            )

        # Create study
        study = create_study(exp, task, model)

        # Create objective function that actually trains
        def make_objective(model_name, base_config, search_sp):
            """Factory to capture variables in closure."""

            def objective(trial):
                # Suggest parameters
                suggested = suggest_params(trial, search_sp)
                print(f"\n  Trial {trial.number}: {suggested}")

                # Apply suggested params to config
                method_config = apply_suggested_params(deepcopy(base_config), suggested)

                # Extract training params
                training_params = method_config.get(
                    "training_params", method_config.get("training", {})
                )

                # Build TrainerConfig
                config = TrainerConfig(
                    epochs=training_params.get("epochs", 1000),
                    lr=training_params.get("lr", 1e-4),
                    batch_size=training_params.get("batch_size", 256),
                    train_size=training_params.get("train_size", 0.9),
                    max_patience=training_params.get("max_patience", 20),
                    rescale=training_params.get("rescale", "z_score"),
                    load=False,  # Never load during HPO
                    save=False,  # Don't save intermediate models
                )

                # Create temporary model path for this trial
                trial_path = (
                    checkpoint_manager.paths.experiment_path
                    / task
                    / "hpo"
                    / model_name
                    / f"trial_{trial.number}"
                )
                trial_path.mkdir(parents=True, exist_ok=True)

                # Create trainer
                trainer = get_trainer(
                    model_name,
                    config,
                    trial_path,
                    task_cfg,  # Pass OmegaConf for interpolation support
                    logger=None,
                )

                # Train with Optuna trial for pruning support
                try:
                    trainer.train(
                        data=(theta_sim, x_sim),
                        device=device,
                        logname=f"trial_{trial.number}",
                        trial=trial,  # Pass trial for pruning
                    )
                except Exception as e:
                    print(f"  Trial {trial.number} failed: {e}")
                    raise

                # Return best validation loss
                val_loss = trainer.get_best_val_loss()
                print(f"  Trial {trial.number} complete: val_loss={val_loss:.6f}")
                return val_loss

            return objective

        objective = make_objective(model, base_method_config, search_space)

        # Run optimization
        study.optimize(objective, n_trials=exp.hpo.n_trials)

        # Save results
        output_path = checkpoint_manager.paths.experiment_path / task / "hpo" / model
        save_study_results(study, output_path)
        print_study_summary(study)

        # Print parameter importance if enough trials
        if len(study.trials) >= 10:
            from utils.optuna_utils import get_param_importance

            importance = get_param_importance(study)
            if importance:
                print("\nParameter importance:")
                for param, score in sorted(
                    importance.items(), key=lambda x: x[1], reverse=True
                ):
                    print(f"  {param}: {score:.4f}")


def run_evaluation(
    exp: ExperimentConfig,
    task: str,
    checkpoint_manager: CheckpointManager,
    device: torch.device,
    cfg: DictConfig,
) -> None:
    """Run evaluation for a task."""
    if not exp.evaluation.enabled:
        return

    print(f"\n{'='*60}")
    print(f"EVALUATION: {task}")
    print(f"Metrics: {exp.evaluation.metrics}")
    print(f"{'='*60}")

    # TODO: Integrate with validate_simulation.py
    print("  [Evaluation not yet implemented in experiment runner]")


def _get_task_config(cfg: DictConfig, task: str) -> DictConfig:
    """Get task configuration, loading if necessary."""
    # For now, return the current task config
    # In a full implementation, we'd load the specific task config
    return cfg.task


def _load_task_config(cfg: DictConfig, task: str) -> DictConfig:
    """Load task configuration for a specific task.

    If the current config already has this task loaded, return it.
    Otherwise, load from configs/task/{task}.yaml.
    """
    # Check if already loaded
    if cfg.task.name == task:
        return cfg.task

    # Load from file
    from pathlib import Path

    from omegaconf import OmegaConf

    task_path = Path("configs/task") / f"{task}.yaml"
    if task_path.exists():
        task_cfg = OmegaConf.load(task_path)
        # Merge with defaults to get rescale etc
        merged = OmegaConf.create({"task": task_cfg})
        # Resolve interpolations within the task config
        OmegaConf.resolve(merged)
        return merged.task
    else:
        raise ValueError(f"Task config not found: {task_path}")


def _load_task_data(
    cfg: DictConfig,
    task_name: str,
    task_cfg: DictConfig,
    checkpoint_manager: CheckpointManager,
):
    """Load or generate data for a specific task.

    This is similar to load_or_generate_data but uses a task-specific config.
    """
    from omegaconf import OmegaConf

    from simulator import generate_calibration_dataset, generate_simulation_dataset, get_simulator
    from utils.data_split import DataSplitter

    # Check if data already exists (in shared data path)
    data_dir = checkpoint_manager.paths.shared_data_path(task_name)
    sim_path = data_dir / "simulations.pt"
    cal_path = data_dir / "calibrations.pt"
    if sim_path.exists() and cal_path.exists():
        print(f"Loading existing training data for {task_name}")
        sim_data = torch.load(sim_path, weights_only=False)
        cal_data = torch.load(cal_path, weights_only=False)
        return (
            sim_data["theta"],
            sim_data["x"],
            cal_data["theta"],
            cal_data["x"],
            cal_data["y"],
        )

    # Generate new data
    print(f"Generating training data for {task_name}")

    # Get simulator params
    simulator_params = dict(OmegaConf.to_container(task_cfg.simulator.params, resolve=True))

    # Add data_dir for file-based tasks
    if task_name in ["wind_tunnel", "light_tunnel", "js"]:
        simulator_params["data_dir"] = cfg.get("data_dir", "./data")

    simulator = get_simulator(task_cfg.simulator.name, **simulator_params)

    # Simulation data
    theta_sim, x_sim = generate_simulation_dataset(simulator, task_cfg.num_samples)

    # Get num_cal from task config or use defaults
    num_cal_list = task_cfg.get("num_cal", [10, 50, 100, 500, 1000])
    if isinstance(num_cal_list, int):
        num_cal_max = num_cal_list
    else:
        num_cal_max = max(num_cal_list)

    # Create data splitter
    splitter = DataSplitter.from_config(cfg)

    # Get total available observations
    if hasattr(simulator, "get_total_observations"):
        total_available = simulator.get_total_observations()
    else:
        total_available = num_cal_max * 10

    # Create or load the split
    experiment_path = checkpoint_manager.paths.shared_data_path(task_name)
    split = splitter.get_or_create_split(experiment_path, total_available)

    # Get calibration indices
    cal_indices = split.get_calibration_indices(num_cal_max)

    # Generate calibration data
    generation = task_cfg.get("generation", "transitive")
    theta_cal, x_cal, y_cal = generate_calibration_dataset(
        simulator, num_cal_max, generation, indices=tuple(cal_indices)
    )

    # Save data to shared data path
    data_dir = checkpoint_manager.paths.shared_data_path(task_name)
    data_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"theta": theta_sim, "x": x_sim}, data_dir / "simulations.pt")
    torch.save({"theta": theta_cal, "x": x_cal, "y": y_cal}, data_dir / "calibrations.pt")

    return theta_sim, x_sim, theta_cal, x_cal, y_cal


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict into dot-notation keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _apply_overrides(cfg: DictConfig, overrides: dict) -> DictConfig:
    """Apply overrides to a config."""
    cfg = OmegaConf.to_container(cfg, resolve=True)
    for key, value in overrides.items():
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return OmegaConf.create(cfg)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main experiment runner."""
    # Get special config options
    dry_run = cfg.get("dry_run", False)
    from_checkpoint = cfg.get("from_checkpoint", None)
    force = cfg.get("force", False)

    # Handle from_checkpoint
    if from_checkpoint:
        checkpoint_path = Path(from_checkpoint)
        print(f"Reproducing from checkpoint: {checkpoint_path}")
        exp = load_experiment_from_checkpoint(checkpoint_path)
        print(f"Loaded experiment: {exp.name}")
        print(f"Original profile: {exp.profile}")
        print(f"Original CLI args: {exp.cli_args}")
    else:
        # Check if experiment is specified
        if "experiment" not in cfg:
            print("ERROR: No experiment specified.")
            print("Usage: python run_experiment.py +experiment=<name>")
            print("\nRun with --list to see available experiments.")
            return

        # Resolve experiment config
        exp = resolve_experiment(cfg)

        # Apply profile if present
        if "profile" in cfg:
            profile = resolve_profile(cfg)
            exp = apply_profile(exp, profile)

    # Override HPO storage if specified (for cluster runs)
    if cfg.get("hpo_storage"):
        exp.hpo.storage = cfg.hpo_storage

    # Handle dry_run
    if dry_run:
        print_experiment_summary(exp)
        print("\n[DRY RUN - No training will be executed]")
        return

    # Setup
    device = setup_device()
    torch.manual_seed(cfg.get("seed", 0))

    # Generate experiment name and setup paths
    exp_name = make_experiment_name(exp)
    print(f"\nExperiment: {exp_name}")

    paths = ResultsPath(cfg.get("results_dir", "./results"), exp_name)
    checkpoint_manager = CheckpointManager(paths)

    # Save experiment config
    save_experiment_config(exp, paths.experiment_path, cli_args=sys.argv[1:])
    print(f"Config saved to: {paths.experiment_path / 'experiment.yaml'}")

    # Print summary
    print_experiment_summary(exp)

    # Run experiment
    print("\n" + "=" * 60)
    print("STARTING EXPERIMENT")
    print("=" * 60)

    for task in exp.tasks:
        print(f"\n{'#'*60}")
        print(f"# TASK: {task}")
        print(f"{'#'*60}")

        # HPO mode
        if exp.hpo.enabled:
            run_hpo(exp, task, checkpoint_manager, device, cfg)
            continue

        # Standard training with optional sweep
        for sweep_point in expand_sweep(exp):
            if sweep_point.tag:
                print(f"\n--- Sweep point: {sweep_point.tag} ---")

            # Train simulation models
            run_simulation_training(
                exp, task, sweep_point, checkpoint_manager, device, cfg, force
            )

            # Train calibration methods
            run_calibration_training(
                exp, task, sweep_point, checkpoint_manager, device, cfg, force
            )

            # Train baselines
            run_baseline_training(
                exp, task, sweep_point, checkpoint_manager, device, cfg, force
            )

        # Run evaluation (after all sweep points)
        run_evaluation(exp, task, checkpoint_manager, device, cfg)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {paths.experiment_path}")


if __name__ == "__main__":
    main()
