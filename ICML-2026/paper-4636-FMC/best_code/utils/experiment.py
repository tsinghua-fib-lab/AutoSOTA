"""Experiment configuration utilities.

This module provides utilities for:
1. Parsing experiment configs into structured dataclasses
2. Applying profile scaling (sanity/development/production)
3. Expanding grid sweeps into individual run configs
4. Saving/loading experiment configs for reproducibility

Usage:
    from utils.experiment import (
        ExperimentConfig,
        resolve_experiment,
        apply_profile,
        expand_sweep,
        list_experiments,
    )

    # Load and resolve experiment
    exp = resolve_experiment(cfg)

    # Apply profile scaling
    exp = apply_profile(exp, "sanity")

    # Expand sweep into individual configs
    for sweep_point in expand_sweep(exp):
        run_training(sweep_point)
"""

import itertools
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from omegaconf import DictConfig, OmegaConf


@dataclass
class RunConfig:
    """Configuration for what to train."""

    simulation_models: List[str]
    calibration_methods: List[str]
    baselines: List[str]
    seeds: List[int]
    num_cal: List[int]


@dataclass
class SweepConfig:
    """Configuration for grid sweeps."""

    enabled: bool = False
    type: str = "grid"
    parameter: Optional[str] = None
    values: Optional[List[Any]] = None
    axes: Optional[List[Dict[str, Any]]] = None  # For multi-axis sweeps


@dataclass
class HPOConfig:
    """Configuration for Optuna hyperparameter optimization."""

    enabled: bool = False
    backend: str = "optuna"
    per_task: bool = True
    objective: str = "val_loss"
    n_trials: int = 50
    pruning: bool = True
    storage: Optional[str] = None
    study_name_template: Optional[str] = None
    search_spaces: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    enabled: bool = True
    metrics: List[str] = field(default_factory=lambda: ["joint_mmd", "joint_c2st"])
    num_test_samples: int = 100
    compare_across: Optional[Union[str, List[str]]] = None
    custom_script: Optional[str] = None


@dataclass
class ProfileConfig:
    """Configuration for experiment profiles."""

    name: str
    description: str = ""
    tasks_override: Optional[List[str]] = None
    tasks_subset: Optional[int] = None
    seeds: List[int] = field(default_factory=lambda: [0])
    num_cal: List[int] = field(default_factory=lambda: [50])
    num_test_samples: int = 50
    training_epochs_scale: float = 1.0
    training_max_patience: int = 20
    data_num_samples_scale: float = 1.0
    hpo_n_trials_override: Optional[int] = None


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    name: str
    description: str
    tags: List[str]
    author: str

    tasks: List[str]
    run: RunConfig
    sweep: SweepConfig
    hpo: HPOConfig
    evaluation: EvalConfig

    method_overrides: Dict[str, Any] = field(default_factory=dict)
    task_overrides: Dict[str, Any] = field(default_factory=dict)

    # Profile that was applied
    profile: Optional[str] = None

    # Metadata for reproduction
    created_at: Optional[str] = None
    git_commit: Optional[str] = None
    cli_args: Optional[List[str]] = None


@dataclass
class SweepPoint:
    """A single point in a sweep."""

    tag: str  # e.g., "arch_resmlp" or "lr_1e-4_depth_5"
    overrides: Dict[str, Any]  # Config overrides for this point


def _to_container_safe(obj) -> dict:
    """Convert to dict, handling both OmegaConf and regular dicts."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    return OmegaConf.to_container(obj, resolve=True)


def resolve_experiment(cfg: DictConfig) -> ExperimentConfig:
    """Parse Hydra config into ExperimentConfig.

    Args:
        cfg: Full Hydra configuration with experiment loaded

    Returns:
        Parsed ExperimentConfig
    """
    # Handle nested structure from +experiment=name
    # The experiment YAML content is nested under cfg.experiment
    exp_root = cfg.get("experiment", {})

    # If there's nested experiment metadata, the actual config is in exp_root
    if "experiment" in exp_root and "tasks" in exp_root:
        # This is the case when using +experiment=name
        exp = exp_root.get("experiment", {})
        run = exp_root.get("run", {})
        sweep = exp_root.get("sweep", {})
        hpo = exp_root.get("hpo", {})
        evaluation = exp_root.get("evaluation", {})
        tasks = exp_root.get("tasks", [])
        method_overrides = exp_root.get("method_overrides", {})
        task_overrides = exp_root.get("task_overrides", {})
    else:
        # Legacy or flat structure
        exp = exp_root
        run = cfg.get("run", {})
        sweep = cfg.get("sweep", {})
        hpo = cfg.get("hpo", {})
        evaluation = cfg.get("evaluation", {})
        tasks = cfg.get("tasks", [])
        method_overrides = cfg.get("method_overrides", {})
        task_overrides = cfg.get("task_overrides", {})

    return ExperimentConfig(
        name=exp.get("name", "unnamed_experiment"),
        description=exp.get("description", ""),
        tags=list(exp.get("tags", [])),
        author=exp.get("author", "unknown"),
        tasks=list(tasks),
        run=RunConfig(
            simulation_models=list(run.get("simulation_models", [])),
            calibration_methods=list(run.get("calibration_methods", [])),
            baselines=list(run.get("baselines", [])),
            seeds=list(run.get("seeds", [0])),
            num_cal=list(run.get("num_cal", [50])),
        ),
        sweep=SweepConfig(
            enabled=sweep.get("enabled", False),
            type=sweep.get("type", "grid"),
            parameter=sweep.get("parameter"),
            values=list(sweep.get("values", [])) if sweep.get("values") else None,
            axes=list(sweep.get("axes", [])) if sweep.get("axes") else None,
        ),
        hpo=HPOConfig(
            enabled=hpo.get("enabled", False),
            backend=hpo.get("backend", "optuna"),
            per_task=hpo.get("per_task", True),
            objective=hpo.get("objective", "val_loss"),
            n_trials=hpo.get("n_trials", 50),
            pruning=hpo.get("pruning", True),
            storage=hpo.get("storage"),
            study_name_template=hpo.get("study_name_template"),
            search_spaces=_extract_search_spaces(hpo),
        ),
        evaluation=EvalConfig(
            enabled=evaluation.get("enabled", True),
            metrics=list(evaluation.get("metrics", ["joint_mmd", "joint_c2st"])),
            num_test_samples=evaluation.get("num_test_samples", 100),
            compare_across=evaluation.get("compare_across"),
            custom_script=evaluation.get("custom_script"),
        ),
        method_overrides=_to_container_safe(method_overrides),
        task_overrides=_to_container_safe(task_overrides),
        created_at=datetime.now().isoformat(),
        git_commit=_get_git_commit(),
    )


def _extract_search_spaces(hpo: DictConfig) -> Dict[str, Dict[str, Any]]:
    """Extract per-model search spaces from HPO config."""
    search_spaces = {}
    # Known model names to look for
    model_names = ["npe", "fmpe", "fm_post_transform", "rope", "dpe"]

    for model in model_names:
        if model in hpo:
            search_spaces[model] = OmegaConf.to_container(hpo[model], resolve=True)

    return search_spaces


def resolve_profile(cfg: DictConfig) -> ProfileConfig:
    """Parse profile config from Hydra config or load by name.

    Args:
        cfg: Hydra config that may contain either:
            - profile: dict (loaded via defaults)
            - profile: str (profile name to load)
    """
    profile = cfg.get("profile", {})

    # If profile is a string (name), load from file
    if isinstance(profile, str):
        profile = load_profile_by_name(profile)

    # If profile is empty or not present, use defaults
    if not profile:
        return ProfileConfig(name="default")

    training = profile.get("training", {})
    data = profile.get("data", {})
    hpo = profile.get("hpo", {})

    return ProfileConfig(
        name=profile.get("name", "default"),
        description=profile.get("description", ""),
        tasks_override=list(profile.get("tasks_override", []))
        if profile.get("tasks_override")
        else None,
        tasks_subset=profile.get("tasks_subset"),
        seeds=list(profile.get("seeds", [0])),
        num_cal=list(profile.get("num_cal", [50])),
        num_test_samples=profile.get("num_test_samples", 50),
        training_epochs_scale=training.get("epochs_scale", 1.0),
        training_max_patience=training.get("max_patience", 20),
        data_num_samples_scale=data.get("num_samples_scale", 1.0),
        hpo_n_trials_override=hpo.get("n_trials_override"),
    )


def load_profile_by_name(profile_name: str, config_dir: Optional[Path] = None) -> dict:
    """Load a profile by name from the profiles directory.

    Args:
        profile_name: Name of the profile (e.g., "sanity", "production")
        config_dir: Path to configs directory

    Returns:
        Profile configuration dict
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / "configs"

    profile_path = config_dir / "experiment" / "_profiles" / f"{profile_name}.yaml"

    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    return OmegaConf.to_container(OmegaConf.load(profile_path), resolve=True)


def apply_profile(exp: ExperimentConfig, profile: ProfileConfig) -> ExperimentConfig:
    """Apply profile scaling to experiment config.

    Args:
        exp: Experiment configuration
        profile: Profile configuration

    Returns:
        Modified experiment config with profile applied
    """
    # Override tasks if profile specifies
    if profile.tasks_override:
        exp.tasks = profile.tasks_override
    elif profile.tasks_subset and profile.tasks_subset < len(exp.tasks):
        exp.tasks = exp.tasks[: profile.tasks_subset]

    # Apply run config from profile
    exp.run.seeds = profile.seeds
    exp.run.num_cal = profile.num_cal

    # Apply evaluation settings
    exp.evaluation.num_test_samples = profile.num_test_samples

    # Apply HPO trial override
    if profile.hpo_n_trials_override is not None:
        exp.hpo.n_trials = profile.hpo_n_trials_override

    # Apply training scaling to method_overrides
    if profile.training_epochs_scale != 1.0:
        for method_name, overrides in exp.method_overrides.items():
            if "training_params" in overrides:
                if "epochs" in overrides["training_params"]:
                    original = overrides["training_params"]["epochs"]
                    scaled = int(original * profile.training_epochs_scale)
                    exp.method_overrides[method_name]["training_params"]["epochs"] = max(10, scaled)

    if profile.training_max_patience is not None:
        for method_name, overrides in exp.method_overrides.items():
            if "training_params" in overrides:
                exp.method_overrides[method_name]["training_params"]["max_patience"] = profile.training_max_patience

    # Store profile name
    exp.profile = profile.name

    return exp


def expand_sweep(exp: ExperimentConfig) -> Iterator[SweepPoint]:
    """Expand sweep configuration into individual run points.

    Args:
        exp: Experiment configuration with sweep

    Yields:
        SweepPoint for each combination in the sweep
    """
    if not exp.sweep.enabled:
        # No sweep - yield single empty point
        yield SweepPoint(tag="", overrides={})
        return

    if exp.sweep.type == "grid":
        if exp.sweep.axes:
            # Multi-axis sweep
            yield from _expand_multi_axis_sweep(exp.sweep.axes)
        elif exp.sweep.parameter and exp.sweep.values:
            # Single-axis sweep
            for value in exp.sweep.values:
                tag = _make_sweep_tag(exp.sweep.parameter, value)
                yield SweepPoint(
                    tag=tag, overrides={exp.sweep.parameter: value}
                )
    else:
        raise ValueError(f"Unknown sweep type: {exp.sweep.type}")


def _expand_multi_axis_sweep(axes: List[Dict[str, Any]]) -> Iterator[SweepPoint]:
    """Expand multi-axis grid sweep."""
    # Build list of (parameter, values) tuples
    axis_values = []
    for axis in axes:
        param = axis["parameter"]
        values = axis["values"]
        axis_values.append([(param, v) for v in values])

    # Cartesian product of all axes
    for combination in itertools.product(*axis_values):
        overrides = {param: value for param, value in combination}
        tag = "_".join(_make_sweep_tag(p, v) for p, v in combination)
        yield SweepPoint(tag=tag, overrides=overrides)


def _make_sweep_tag(parameter: str, value: Any) -> str:
    """Create a filesystem-safe tag for a sweep point."""
    # Extract last part of parameter path
    param_short = parameter.split(".")[-1]

    # Format value
    if isinstance(value, float):
        value_str = f"{value:.0e}".replace(".", "").replace("-", "m")
    elif isinstance(value, list):
        value_str = f"{len(value)}x{value[0]}" if value else "empty"
    else:
        value_str = str(value).replace("/", "_").replace(" ", "")

    return f"{param_short}_{value_str}"


def make_experiment_name(exp: ExperimentConfig, timestamp: bool = True) -> str:
    """Generate experiment output directory name.

    Args:
        exp: Experiment configuration
        timestamp: Whether to include timestamp

    Returns:
        Experiment name string
    """
    parts = [exp.name]

    if exp.profile:
        parts.append(exp.profile)

    if timestamp:
        parts.append(datetime.now().strftime("%Y%m%d"))

    return "_".join(parts)


def save_experiment_config(
    exp: ExperimentConfig,
    output_path: Path,
    cli_args: Optional[List[str]] = None,
) -> Path:
    """Save experiment config for reproduction.

    Args:
        exp: Experiment configuration
        output_path: Directory to save config
        cli_args: Original CLI arguments

    Returns:
        Path to saved config file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Update metadata
    exp.cli_args = cli_args
    exp.git_commit = _get_git_commit()

    # Convert to dict
    config_dict = {
        "experiment": {
            "name": exp.name,
            "description": exp.description,
            "tags": exp.tags,
            "author": exp.author,
        },
        "tasks": exp.tasks,
        "run": {
            "simulation_models": exp.run.simulation_models,
            "calibration_methods": exp.run.calibration_methods,
            "baselines": exp.run.baselines,
            "seeds": exp.run.seeds,
            "num_cal": exp.run.num_cal,
        },
        "sweep": {
            "enabled": exp.sweep.enabled,
            "type": exp.sweep.type,
            "parameter": exp.sweep.parameter,
            "values": exp.sweep.values,
            "axes": exp.sweep.axes,
        },
        "hpo": {
            "enabled": exp.hpo.enabled,
            "backend": exp.hpo.backend,
            "per_task": exp.hpo.per_task,
            "objective": exp.hpo.objective,
            "n_trials": exp.hpo.n_trials,
            "search_spaces": exp.hpo.search_spaces,
        },
        "evaluation": {
            "enabled": exp.evaluation.enabled,
            "metrics": exp.evaluation.metrics,
            "num_test_samples": exp.evaluation.num_test_samples,
            "compare_across": exp.evaluation.compare_across,
            "custom_script": exp.evaluation.custom_script,
        },
        "method_overrides": exp.method_overrides,
        "task_overrides": exp.task_overrides,
        "_reproduction": {
            "profile": exp.profile,
            "created_at": exp.created_at,
            "git_commit": exp.git_commit,
            "cli_args": exp.cli_args,
        },
    }

    config_path = output_path / "experiment.yaml"
    OmegaConf.save(OmegaConf.create(config_dict), config_path)

    return config_path


def load_experiment_from_checkpoint(checkpoint_path: Path) -> ExperimentConfig:
    """Load experiment config from a checkpoint directory.

    Args:
        checkpoint_path: Path to experiment results directory

    Returns:
        Loaded ExperimentConfig

    Raises:
        FileNotFoundError: If experiment.yaml not found
    """
    config_path = Path(checkpoint_path) / "experiment.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"No experiment.yaml found at {checkpoint_path}. "
            "This may be an old checkpoint without experiment tracking."
        )

    cfg = OmegaConf.load(config_path)
    exp = resolve_experiment(cfg)

    # Restore reproduction metadata
    repro = cfg.get("_reproduction", {})
    exp.profile = repro.get("profile")
    exp.created_at = repro.get("created_at")
    exp.git_commit = repro.get("git_commit")
    exp.cli_args = repro.get("cli_args")

    return exp


def list_experiments(config_dir: Optional[Path] = None) -> Dict[str, str]:
    """List available experiments with descriptions.

    Args:
        config_dir: Path to configs directory

    Returns:
        Dict mapping experiment name to description
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / "configs"

    exp_dir = config_dir / "experiment"
    if not exp_dir.exists():
        return {}

    experiments = {}
    for yaml_file in exp_dir.glob("*.yaml"):
        if yaml_file.name.startswith("_"):
            continue  # Skip _profiles, _base, etc.

        try:
            cfg = OmegaConf.load(yaml_file)
            exp_cfg = cfg.get("experiment", {})
            name = yaml_file.stem
            desc = exp_cfg.get("description", "")
            experiments[name] = desc
        except Exception:
            experiments[yaml_file.stem] = "(error loading config)"

    return experiments


def list_profiles(config_dir: Optional[Path] = None) -> Dict[str, str]:
    """List available profiles with descriptions.

    Args:
        config_dir: Path to configs directory

    Returns:
        Dict mapping profile name to description
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / "configs"

    profile_dir = config_dir / "experiment" / "_profiles"
    if not profile_dir.exists():
        return {}

    profiles = {}
    for yaml_file in profile_dir.glob("*.yaml"):
        try:
            cfg = OmegaConf.load(yaml_file)
            name = yaml_file.stem
            desc = cfg.get("description", "")
            profiles[name] = desc
        except Exception:
            profiles[yaml_file.stem] = "(error loading config)"

    return profiles


def _get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def print_experiment_summary(exp: ExperimentConfig) -> None:
    """Print a summary of the experiment configuration."""
    print("=" * 60)
    print(f"EXPERIMENT: {exp.name}")
    print("=" * 60)
    print(f"Description: {exp.description}")
    print(f"Profile: {exp.profile or 'none'}")
    print(f"Tasks: {exp.tasks}")
    print()
    print("Run Configuration:")
    print(f"  Simulation models: {exp.run.simulation_models}")
    print(f"  Calibration methods: {exp.run.calibration_methods}")
    print(f"  Baselines: {exp.run.baselines}")
    print(f"  Seeds: {exp.run.seeds}")
    print(f"  Calibration sizes: {exp.run.num_cal}")
    print()

    if exp.sweep.enabled:
        print("Sweep:")
        print(f"  Type: {exp.sweep.type}")
        print(f"  Parameter: {exp.sweep.parameter}")
        print(f"  Values: {exp.sweep.values}")
        print()

    if exp.hpo.enabled:
        print("HPO:")
        print(f"  Backend: {exp.hpo.backend}")
        print(f"  Trials: {exp.hpo.n_trials}")
        print(f"  Per-task: {exp.hpo.per_task}")
        print(f"  Objective: {exp.hpo.objective}")
        print(f"  Models: {list(exp.hpo.search_spaces.keys())}")
        print()

    if exp.evaluation.enabled:
        print("Evaluation:")
        print(f"  Metrics: {exp.evaluation.metrics}")
        print(f"  Test samples: {exp.evaluation.num_test_samples}")
        print()

    print("=" * 60)
