import os
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

mm_raw: Path | None = None
mm_preprocessed: Path | None = None
mm_results: Path | None = None

P = ParamSpec("P")
R = TypeVar("R")
ENV_VAR_NAMES = {
    "mm_raw": ("MM_RAW", "mm_raw"),
    "mm_preprocessed": ("MM_PREPROCESSED", "mm_preprocessed"),
    "mm_results": ("MM_RESULTS", "mm_results"),
}


def _resolve_path(name: str, current_value: Path | None) -> Path | None:
    if current_value is not None:
        return current_value

    for env_name in ENV_VAR_NAMES[name]:
        env_value = os.environ.get(env_name)
        if env_value:
            return Path(env_value).expanduser()

    return None


def verify_required_global_paths_set(
    required_paths: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    paths = {
        "mm_raw": _resolve_path("mm_raw", mm_raw),
        "mm_preprocessed": _resolve_path("mm_preprocessed", mm_preprocessed),
        "mm_results": _resolve_path("mm_results", mm_results),
    }
    required_names = tuple(paths) if required_paths is None else required_paths

    missing = [name for name in required_names if paths[name] is None]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            "Required path variables are not set. Define these environment "
            f"variables or Python globals first: {missing_str}"
        )

    non_existing = [
        name
        for name in required_names
        for value in [paths[name]]
        if value is not None and not value.exists()
    ]
    if non_existing:
        missing_paths = ", ".join(f"{name}={paths[name]}" for name in non_existing)
        raise FileNotFoundError(f"Required paths do not exist: {missing_paths}")

    return {name: value for name, value in paths.items() if value is not None}


def find_dataset_dir(mm_raw_path: Path, dataset_id: str) -> Path:
    pattern = f"Dataset_{dataset_id}_*"
    matches = sorted(path for path in mm_raw_path.glob(pattern) if path.is_dir())

    if not matches:
        raise FileNotFoundError(
            f"No dataset folder matching '{pattern}' found in {mm_raw_path}"
        )

    if len(matches) > 1:
        match_list = ", ".join(str(path.name) for path in matches)
        raise ValueError(
            f"Found multiple dataset folders for id {dataset_id} in {mm_raw_path}: {match_list}"
        )

    return matches[0]


def require_global_paths_set(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        verify_required_global_paths_set()
        return func(*args, **kwargs)

    return wrapper
