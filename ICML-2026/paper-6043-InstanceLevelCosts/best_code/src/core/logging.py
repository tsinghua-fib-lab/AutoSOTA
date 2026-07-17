"""
Thin wandb wrapper for experiment logging.

Provides a simplified interface for wandb logging that gracefully handles
disabled mode (WANDB_MODE=disabled) by becoming a no-op.

Usage:
    from core.logging import init_run, log_metrics, log_artifact, finish_run

    init_run(project='my-project', name='experiment-1', config={'lr': 0.01})
    log_metrics({'accuracy': 0.95, 'loss': 0.1}, step=100)
    log_artifact('model.pkl', name='trained-model', type='model')
    finish_run()

Environment Variables:
    WANDB_MODE: Set to 'disabled' to turn all logging into no-ops
    WANDB_PROJECT: Default project name if not specified in init_run
"""

import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


# Track whether wandb is available and enabled
_wandb_available = False
_wandb_enabled = False
_current_run = None

try:
    import wandb
    _wandb_available = True
except ImportError:
    wandb = None


def _is_enabled() -> bool:
    """Check if wandb logging is enabled."""
    if not _wandb_available:
        return False
    mode = os.environ.get('WANDB_MODE', '').lower()
    return mode != 'disabled'


def init_run(
    project: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    group: Optional[str] = None,
    reinit: bool = True,
) -> Optional[Any]:
    """
    Initialize a wandb run.

    Args:
        project: Project name (defaults to WANDB_PROJECT env var)
        name: Run name (auto-generated if not provided)
        config: Configuration dictionary to log
        tags: List of tags for the run
        notes: Notes/description for the run
        group: Group name for organizing runs
        reinit: Allow reinitializing if a run already exists

    Returns:
        wandb.Run object if enabled, None otherwise
    """
    global _wandb_enabled, _current_run

    if not _is_enabled():
        _wandb_enabled = False
        _current_run = None
        return None

    _wandb_enabled = True

    # Use environment variable as default project
    if project is None:
        project = os.environ.get('WANDB_PROJECT', 'cost-sensitive-learning')

    _current_run = wandb.init(
        project=project,
        name=name,
        config=config,
        tags=tags,
        notes=notes,
        group=group,
        reinit=reinit,
    )

    return _current_run


def log_metrics(
    metrics: Dict[str, Union[float, int]],
    step: Optional[int] = None,
    commit: bool = True,
) -> None:
    """
    Log metrics to wandb.

    Args:
        metrics: Dictionary of metric names to values
        step: Optional step number (uses wandb's internal counter if not provided)
        commit: Whether to commit the log (advance step counter)
    """
    if not _wandb_enabled or _current_run is None:
        return

    wandb.log(metrics, step=step, commit=commit)


def log_summary(
    metrics: Dict[str, Union[float, int]],
) -> None:
    """
    Log summary metrics (final values) to wandb.

    These appear in the run summary table and are useful for final results.

    Args:
        metrics: Dictionary of metric names to values
    """
    if not _wandb_enabled or _current_run is None:
        return

    for key, value in metrics.items():
        wandb.run.summary[key] = value


def log_artifact(
    filepath: Union[str, Path],
    name: str,
    type: str = 'file',
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """
    Log a file or directory as a wandb artifact.

    Args:
        filepath: Path to file or directory to log
        name: Name for the artifact
        type: Type of artifact (e.g., 'model', 'dataset', 'results')
        description: Optional description
        metadata: Optional metadata dictionary

    Returns:
        wandb.Artifact object if enabled, None otherwise
    """
    if not _wandb_enabled or _current_run is None:
        return None

    filepath = Path(filepath)

    artifact = wandb.Artifact(
        name=name,
        type=type,
        description=description,
        metadata=metadata,
    )

    if filepath.is_dir():
        artifact.add_dir(str(filepath))
    else:
        artifact.add_file(str(filepath))

    wandb.log_artifact(artifact)
    return artifact


def log_table(
    key: str,
    columns: List[str],
    data: List[List[Any]],
) -> None:
    """
    Log a table to wandb.

    Args:
        key: Name/key for the table
        columns: List of column names
        data: List of rows, where each row is a list of values
    """
    if not _wandb_enabled or _current_run is None:
        return

    table = wandb.Table(columns=columns, data=data)
    wandb.log({key: table})


def finish_run(
    exit_code: Optional[int] = None,
    quiet: bool = False,
) -> None:
    """
    Finish the current wandb run.

    Args:
        exit_code: Optional exit code (0 for success)
        quiet: Suppress output messages
    """
    global _current_run

    if not _wandb_enabled or _current_run is None:
        _current_run = None
        return

    wandb.finish(exit_code=exit_code, quiet=quiet)
    _current_run = None


def get_run() -> Optional[Any]:
    """
    Get the current wandb run object.

    Returns:
        Current wandb.Run object if active, None otherwise
    """
    if not _wandb_enabled:
        return None
    return _current_run


def is_enabled() -> bool:
    """
    Check if wandb logging is currently enabled and active.

    Returns:
        True if wandb is enabled and a run is active
    """
    return _wandb_enabled and _current_run is not None


def update_config(config: Dict[str, Any]) -> None:
    """
    Update the config of the current run.

    Args:
        config: Dictionary of config values to update
    """
    if not _wandb_enabled or _current_run is None:
        return

    wandb.config.update(config)
