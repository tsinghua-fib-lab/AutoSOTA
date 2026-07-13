"""Central configuration values for the BO experiment pipeline.

This module intentionally stays small: it only defines constants and tiny
directory helpers so that `bo_cli.py`, `bo_runner.py` and plotting code share
the same defaults.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Core experiment sizes
# -----------------------------------------------------------------------------

# Number of BO iterations after the initial presample block.
EVAL_BUDGET = 50

# Number of initial objective evaluations used to seed the optimizer.
INITIAL_SAMPLE_SIZE = 10

# Size of the fixed candidate pool used for ranking metrics and analysis.
RANKING_POOL_SIZE = 200

# Top-k size used by the distance analysis helpers.
TOP_K_PROGRAMS = 10

# Default seed used when a run does not explicitly provide one.
DEFAULT_RANDOM_SEED = 42


# -----------------------------------------------------------------------------
# Distance-analysis configuration
# -----------------------------------------------------------------------------

# Structural kernels supported by the distance analysis utilities.
DISTANCE_KERNEL_OPTIONS = ("wl", "tree")


# Default structural distance kernel used in trace visualizations.
DEFAULT_DISTANCE_KERNEL: str = "wl"


def _module_results_root() -> Path:
    """Return the repository-local root folder for generated results."""
    return Path(__file__).resolve().parent / "results"


def get_default_results_dir() -> Path:
    """Return and create the default directory for per-run BO outputs."""
    path = _module_results_root() / "bo_runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_default_plot_dir() -> Path:
    """Return and create the default directory for aggregated BO plots."""
    path = _module_results_root() / "bo_plots"
    path.mkdir(parents=True, exist_ok=True)
    return path
