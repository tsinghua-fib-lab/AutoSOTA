"""Optuna utilities for hyperparameter optimization.

This module provides utilities for:
1. Creating Optuna studies with experiment-specific settings
2. Building search spaces from experiment config
3. Suggesting parameters during trials
4. Saving and loading study results

Usage:
    from utils.optuna_utils import (
        create_study,
        build_objective,
        suggest_params,
        save_study_results,
    )

    # Create study for a specific task/model
    study = create_study(experiment, task="pure_gaussian", model="fmpe")

    # Run optimization
    study.optimize(objective, n_trials=50)

    # Save results
    save_study_results(study, output_path)
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from utils.experiment import ExperimentConfig, HPOConfig


def create_study(
    experiment: ExperimentConfig,
    task: str,
    model: str,
    direction: str = "minimize",
) -> optuna.Study:
    """Create an Optuna study for a task/model combination.

    Args:
        experiment: Experiment configuration
        task: Task name
        model: Model name (npe, fmpe, etc.)
        direction: Optimization direction ("minimize" or "maximize")

    Returns:
        Optuna Study object
    """
    hpo = experiment.hpo

    # Build study name
    if hpo.study_name_template:
        study_name = (
            hpo.study_name_template.replace("${experiment.name}", experiment.name)
            .replace("${task}", task)
            .replace("${model}", model)
        )
    else:
        study_name = f"{experiment.name}_{task}_{model}"

    # Configure sampler
    sampler = TPESampler(seed=42)

    # Configure pruner
    pruner = MedianPruner() if hpo.pruning else optuna.pruners.NopPruner()

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=hpo.storage,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=True,  # Resume if study exists
    )

    return study


def suggest_params(
    trial: optuna.Trial,
    search_space: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Suggest hyperparameters for a trial.

    Args:
        trial: Optuna trial
        search_space: Search space definition from HPO config

    Returns:
        Dict of suggested parameter values
    """
    suggested = {}

    for param_path, config in search_space.items():
        param_type = config.get("type", "categorical")

        if param_type == "categorical":
            choices = config["choices"]
            # Handle list choices (convert to tuple for hashing)
            if choices and isinstance(choices[0], list):
                # Create string keys for list choices
                choice_map = {str(i): c for i, c in enumerate(choices)}
                key = trial.suggest_categorical(param_path, list(choice_map.keys()))
                value = choice_map[key]
            else:
                value = trial.suggest_categorical(param_path, choices)

        elif param_type == "int":
            value = trial.suggest_int(
                param_path,
                config["low"],
                config["high"],
                step=config.get("step", 1),
            )

        elif param_type == "float":
            value = trial.suggest_float(
                param_path,
                config["low"],
                config["high"],
                step=config.get("step"),
            )

        elif param_type == "loguniform":
            value = trial.suggest_float(
                param_path,
                config["low"],
                config["high"],
                log=True,
            )

        elif param_type == "uniform":
            value = trial.suggest_float(
                param_path,
                config["low"],
                config["high"],
            )

        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

        suggested[param_path] = value

    return suggested


def apply_suggested_params(
    base_config: Dict[str, Any],
    suggested: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply suggested parameters to a base config.

    Args:
        base_config: Base configuration dict
        suggested: Suggested parameters with dot-notation paths

    Returns:
        Modified config dict
    """
    import copy

    config = copy.deepcopy(base_config)

    for param_path, value in suggested.items():
        _set_nested(config, param_path, value)

    return config


def _set_nested(d: Dict, path: str, value: Any) -> None:
    """Set a value in a nested dict using dot notation."""
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _get_nested(d: Dict, path: str, default: Any = None) -> Any:
    """Get a value from a nested dict using dot notation."""
    keys = path.split(".")
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d


def get_search_space(
    hpo_config: HPOConfig,
    model: str,
    task_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Get search space for a model, with optional task-specific overrides.

    Args:
        hpo_config: HPO configuration
        model: Model name
        task_overrides: Optional task-specific HPO overrides

    Returns:
        Search space dict
    """
    # Get base search space
    search_space = hpo_config.search_spaces.get(model, {}).copy()

    # Apply task-specific overrides
    if task_overrides and "hpo" in task_overrides:
        task_hpo = task_overrides["hpo"]
        if model in task_hpo:
            for param, config in task_hpo[model].items():
                search_space[param] = config

    return search_space


def save_study_results(
    study: optuna.Study,
    output_path: Path,
    include_trials: bool = True,
) -> Path:
    """Save study results to JSON.

    Args:
        study: Completed Optuna study
        output_path: Output directory
        include_trials: Whether to include all trial data

    Returns:
        Path to saved results file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "n_trials": len(study.trials),
        "best_trial": {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
        },
    }

    if include_trials:
        results["trials"] = [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
                "datetime_start": t.datetime_start.isoformat()
                if t.datetime_start
                else None,
                "datetime_complete": t.datetime_complete.isoformat()
                if t.datetime_complete
                else None,
            }
            for t in study.trials
        ]

    results_path = output_path / "hpo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results_path


def load_study_results(results_path: Path) -> Dict[str, Any]:
    """Load study results from JSON.

    Args:
        results_path: Path to results file

    Returns:
        Results dict
    """
    with open(results_path) as f:
        return json.load(f)


def get_best_config(
    study: optuna.Study,
    base_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], float]:
    """Get the best configuration from a completed study.

    Args:
        study: Completed Optuna study
        base_config: Base configuration to apply params to

    Returns:
        Tuple of (best_config, best_value)
    """
    best_params = study.best_trial.params
    best_config = apply_suggested_params(base_config, best_params)
    return best_config, study.best_trial.value


def create_objective_factory(
    train_fn: Callable,
    task_config: Dict[str, Any],
    model: str,
    search_space: Dict[str, Dict[str, Any]],
    base_method_config: Dict[str, Any],
) -> Callable[[optuna.Trial], float]:
    """Create an objective function for Optuna.

    Args:
        train_fn: Training function that returns validation loss
        task_config: Task configuration
        model: Model name
        search_space: Search space definition
        base_method_config: Base method configuration

    Returns:
        Objective function for Optuna
    """

    def objective(trial: optuna.Trial) -> float:
        # Suggest parameters
        suggested = suggest_params(trial, search_space)

        # Apply to config
        method_config = apply_suggested_params(base_method_config, suggested)

        # Train and get validation loss
        val_loss = train_fn(
            model_name=model,
            task_config=task_config,
            method_config=method_config,
            trial=trial,  # Pass trial for pruning
        )

        return val_loss

    return objective


def print_study_summary(study: optuna.Study) -> None:
    """Print a summary of a completed study."""
    print(f"\nStudy: {study.study_name}")
    print(f"Direction: {study.direction.name}")
    print(f"Completed trials: {len(study.trials)}")
    print(f"\nBest trial:")
    print(f"  Value: {study.best_trial.value:.6f}")
    print(f"  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


def get_param_importance(study: optuna.Study) -> Dict[str, float]:
    """Get parameter importance from a study.

    Args:
        study: Completed Optuna study

    Returns:
        Dict mapping parameter names to importance scores
    """
    try:
        return optuna.importance.get_param_importances(study)
    except Exception:
        return {}
