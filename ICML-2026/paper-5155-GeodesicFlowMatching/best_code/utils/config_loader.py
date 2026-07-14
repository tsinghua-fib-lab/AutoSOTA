import yaml


def load_experiments(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["experiments"]


def load_pipeline_config(path=None, project_root=None):
    """Load single-run ``configs/config.yaml`` (see :mod:`src.utils`)."""
    from src.utils import load_config

    return load_config(path, project_root=project_root)
