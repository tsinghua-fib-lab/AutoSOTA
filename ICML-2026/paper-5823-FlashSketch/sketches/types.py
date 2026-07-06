from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from typing import Protocol


class SketchConfig(Protocol):
    """Protocol for sketch method configs used by the registry."""

    method: str
    k: int
    seed: int
    dtype: str
