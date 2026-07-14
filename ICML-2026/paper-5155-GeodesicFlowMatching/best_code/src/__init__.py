"""Research pipeline modules. Run the full stack with ``python -m src.main``."""

from src.data_gen import ensure_training_data
from src.evaluate import run_evaluation
from src.train import build_trainer_config, run_training
from src.utils import get_project_root, load_config, resolve_config_paths, validate_config

__all__ = [
    "build_trainer_config",
    "ensure_training_data",
    "get_project_root",
    "load_config",
    "resolve_config_paths",
    "run_evaluation",
    "run_training",
    "validate_config",
]
