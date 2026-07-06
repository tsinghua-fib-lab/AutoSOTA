from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraReadyConfig:
    """Config for camera-ready end-to-end benchmarks."""

    seed: int = 0
    b_noise: float = 0.0
    require_clean_repo: bool = True
    send_slack: bool = False
    progress_every: int = 25
    datasets: list = field(default_factory=list)
    methods: list = field(default_factory=list)
    tasks: list = field(default_factory=list)
