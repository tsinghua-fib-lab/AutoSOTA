from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AblationConfig:
    """Config for ablation end-to-end benchmarks."""

    seed: int = 0
    enable_cuda_timers: bool = False
    b_noise: float = 0.0
    require_clean_repo: bool = True
    send_slack: bool = False
    progress_every: int = 25
    datasets: list = field(default_factory=list)
    methods: list = field(default_factory=list)
    tasks: list = field(default_factory=list)
