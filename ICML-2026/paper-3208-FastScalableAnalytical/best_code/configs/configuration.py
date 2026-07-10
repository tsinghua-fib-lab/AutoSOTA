"""Configuration utilities for the locality diffusion library."""

from __future__ import annotations

import logging
import os
import subprocess
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from omegaconf import OmegaConf
import shutil

LOGGER = logging.getLogger(__name__)


ARTIFACTS_DIRNAME = "artifacts"
IMAGES_DIRNAME = "images"
TENSORS_DIRNAME = "tensors"
LOGS_DIRNAME = "logs"
CONFIG_FILENAME = "config.yaml"


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentConfig:
    """Metadata describing a single experiment/run grouping."""

    name: str = "default"
    run_name: Optional[str] = None
    seed: int = 42
    tags: List[str] = field(default_factory=list)
    append_timestamp: bool = True
    device: str = field(default_factory=_default_device)
    conditional: Optional[bool] = False


@dataclass
class PathsConfig:
    """Directory layout for datasets, artifacts and cached assets."""

    root: str = ""
    datasets: Optional[str] = None
    models: Optional[str] = None
    runs: Optional[str] = None
    wandb: str = "wandb"


@dataclass
class DatasetConfig:
    """Parameters controlling dataset loading and cafching."""

    name: str = "mnist"
    split: str = "train"
    download: bool = True
    batch_size: int = 512
    num_workers: int = 4
    subset_size: Optional[int] = None
    root: Optional[str] = None
    resolution: Optional[int] = None  # Override default resolution for resizing
    num_classes: Optional[int] = 10

@dataclass
class ModelConfig:
    """Model selection and hyper-parameters."""

    name: str = "nearest_dataset"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplingConfig:
    """Parameters for the sampling procedure."""

    num_samples: int = 16
    batch_size: int = 8
    num_inference_steps: int = 10
    timestep_schedule: str = "uniform"  # uniform, quadratic, cosine, snr (ALGO-03)


@dataclass
class WandbConfig:
    """Weights & Biases integration settings."""

    enabled: bool = True
    project: str = "local-diffusion"
    entity: Optional[str] = None
    mode: str = "online"  # "online", "offline", "disabled"
    tags: Optional[List[str]] = None  # If None, inherits from experiment.tags
    job_type: str = "generation"


@dataclass
class OutputConfig:
    """Layout of artefacts produced during runs."""

    code_snapshot: bool = True
    save_metrics: bool = True
    save_final_images: bool = True
    save_image_grid: bool = True
    save_intermediate_images: bool = True


@dataclass
class MetricsConfig:
    """Configuration for evaluation, logging, and output management."""

    # Evaluation metrics
    baseline_name: Optional[str] = 'baseline_unet'
    baseline_unet_path: Optional[str] = None
    baseline_edm_path: Optional[str] = None
    
    # Output settings (what to save)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # WandB logging settings
    wandb: WandbConfig = field(default_factory=WandbConfig)



@dataclass
class RunPaths:
    run_dir: Path
    artifacts: Optional[Path] = None
    images: Optional[Path] = None
    tensors: Optional[Path] = None
    intermediate_images: Optional[Path] = None
    logs: Optional[Path] = None
    config: Optional[Path] = None



@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


def load_config(config_path: str, overrides: Optional[List[str]] = None) -> OmegaConf:
    """Load and merge configuration files using OmegaConf.
    
    Supports OmegaConf's `defaults` feature for config composition.
    Config files can include a `defaults` list to inherit from base configs:
    
    ```yaml
    defaults:
      - /base/defaults
      - /base/wiener_base
    
    experiment:
      run_name: my_run
    ```
    
    Paths in defaults are resolved relative to the configs/ directory.

    Parameters
    ----------
    config_path:
        Path to a YAML/JSON configuration file.
    overrides:
        Optional list of dot-list strings to override configuration values.

    Returns
    -------
    OmegaConf
        The fully merged, resolved configuration object.
    """

    # configs/wiener/mnist.yaml
    config_path = _resolve_config_path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config with OmegaConf's compose API which handles defaults
    # First, resolve defaults paths relative to configs/ directory
    file_conf = OmegaConf.load(config_path)
    
    # Process defaults if present
    if "defaults" in file_conf:
        defaults_list = file_conf.pop("defaults")
        resolved_defaults = []
        for default_path in defaults_list:
            default_path_str = str(default_path)
            # Handle absolute-style paths (starting with /) - treat as relative to configs/
            if default_path_str.startswith("/"):
                default_path_str = default_path_str[1:]  # Remove leading /
            resolved_default = _resolve_config_path(default_path_str)
            if not resolved_default.exists():
                LOGGER.warning("Default config not found: %s (resolved to %s)", default_path, resolved_default)
                continue
            resolved_defaults.append(str(resolved_default))
        
        # Compose configs using OmegaConf's compose API
        if resolved_defaults:
            # Load all default configs
            default_configs = [OmegaConf.load(Path(d)) for d in resolved_defaults]
            # Merge defaults in order, then merge the main config
            merged = OmegaConf.merge(*default_configs, file_conf)
        else:
            merged = file_conf
    else:
        merged = file_conf


    # Merge with structured defaults
    base_conf = OmegaConf.structured(Config)
    merged = OmegaConf.merge(base_conf, merged)


    if overrides:
        override_conf = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, override_conf)

    _resolve_default_paths(merged)
    _resolve_metrics_defaults(merged)
    OmegaConf.set_readonly(merged, True)
    return merged


def _resolve_config_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path

    # Allow referencing configs/ directory by shorthand
    candidate = Path("configs") / path
    if candidate.exists():
        return candidate
    return path


def _resolve_default_paths(cfg: OmegaConf) -> None:
    root = Path(cfg.paths.root).resolve()
    cfg.paths.root = str(root)
    

    datasets = cfg.paths.datasets or (root / "data")
    models = cfg.paths.models or (root / "services/base_models")
    runs = cfg.paths.runs or (root / "outputs")
    wandb_dir = root / "wandb"

    cfg.paths.datasets = str(Path(datasets).resolve())
    cfg.paths.models = str(Path(models).resolve())
    cfg.paths.runs = str(Path(runs).resolve())
    cfg.paths.wandb = str(wandb_dir.resolve())

    if cfg.dataset.root:
        ds_root = Path(cfg.dataset.root).resolve()
    else:
        ds_root = Path(cfg.paths.datasets)
    cfg.dataset.root = str(ds_root)


def _resolve_metrics_defaults(cfg: OmegaConf) -> None:
    """Resolve default values for metrics config, including tag inheritance."""
    # If wandb.tags is None or empty, inherit from experiment.tags
    wandb_tags = cfg.metrics.wandb.tags

    if wandb_tags is None or (isinstance(wandb_tags, list) and len(wandb_tags) == 0):
        cfg.metrics.wandb.tags = list(cfg.experiment.tags)


def ensure_run_directory(cfg: OmegaConf) -> RunPaths:
    """Create the directory tree for the current run and return its paths."""

    # ===== base output directory =====
    base_dir = Path(f"motivation_figures_2/{cfg.model.name}")
    base_dir.mkdir(parents=True, exist_ok=True)

    # ===== gather naming fields =====
    model_name = cfg.model.name
    dataset_name = cfg.dataset.name
    resolution = cfg.dataset.resolution
    seed = cfg.experiment.seed
    batch_size = cfg.sampling.batch_size
    step= cfg.sampling.num_inference_steps


    # whether KNN screening is enabled (legacy name: k_neighbors)
    k_min = cfg.model.params.get("k_min", cfg.model.params.get("k_neighbors", None))
    subset = k_min if isinstance(k_min, int) else "False"

    # streaming softmax
    weight_method = cfg.model.params.get("weight_method", None)

    # ===== assemble run name =====
    if isinstance(subset, int):
        weight_method = "ours"
    run_name = (
    f"{model_name}_{dataset_name}_{resolution}_B{batch_size}_{weight_method}_"
    f"{f'subset{subset}_' if subset else ''}"
    f"step{step}_seed{seed}"
)
    run_dir = base_dir / run_name

    # if duplicate run dir exists: append a timestamp suffix
    if run_dir.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20260117_154233
        run_name = f"{run_name}_{ts}"
        run_dir = base_dir / run_name

    run_dir.mkdir(parents=True, exist_ok=False)

    return RunPaths(
        run_dir=run_dir,
        artifacts=None,
        images=run_dir / IMAGES_DIRNAME,
        tensors=None,
        intermediate_images=None,
        logs=run_dir / LOGS_DIRNAME,
        config=run_dir / CONFIG_FILENAME,
    )


def save_config(cfg: OmegaConf, destination: Path) -> None:
    """Persist the resolved configuration to disk."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=destination)
    LOGGER.info("Saved config to %s", destination)


def get_git_tracked_paths(project_root: Optional[Path] = None) -> Optional[List[Path]]:
    """Return git-tracked file paths relative to project_root, or None if unavailable."""

    if project_root is None:
        project_root = Path(os.getcwd()).resolve()

    if not (project_root / ".git").exists():
        LOGGER.warning("Project root %s is not a git repository. Skipping git file listing.", project_root)
        return None

    try:
        tracked_files_bytes = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=str(project_root)
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        LOGGER.error("Failed to list git-tracked files: %s", exc)
        return None

    tracked_paths = [Path(p) for p in tracked_files_bytes.decode("utf-8").split("\0") if p]
    if not tracked_paths:
        LOGGER.warning("No tracked files found for %s", project_root)
        return []

    return tracked_paths


def snapshot_codebase(destination: Path, project_root: Optional[Path] = None) -> Optional[Path]:
    """Create a compressed snapshot of the codebase for reproducibility."""

    if project_root is None:
        project_root = Path(os.getcwd()).resolve()

    if not destination.parent.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)

    tracked_paths = get_git_tracked_paths(project_root)
    if not tracked_paths:
        return None

    archive_path = destination.with_suffix(".tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        for relative_path in tracked_paths:
            tar.add(project_root / relative_path, arcname=str(relative_path))

    LOGGER.info("Created code snapshot at %s", archive_path)
    return archive_path


def config_to_dict(cfg: OmegaConf) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
