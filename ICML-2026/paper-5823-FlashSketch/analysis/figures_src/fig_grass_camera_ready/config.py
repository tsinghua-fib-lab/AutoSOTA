from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from dataclasses import dataclass, field

import globals as g
from bench.grass_external.config import MLP_MNIST_CONFIG, MLP_MNIST_FEATURE_DIM


@dataclass(frozen=True)
class GrassCameraReadyConfig:
    """Configuration for camera-ready GraSS plots."""

    use_latest_run: bool = True
    run_paths: list[str] = field(default_factory=list)
    dataset: str = g.DATASET_MNIST
    dataset_name: str = g.DATASET_MNIST
    dataset_group: str | None = None
    model: str = g.MODEL_MLP
    task: str = g.TASK_GRASS_LDS
    activation: str = "relu"
    dropout_rate: float = 0.1
    proj_time_col: str = "proj_only_time_mean_ms"
    proj_time_std_col: str = "proj_only_time_std_ms"
    metric_col: str = "lds_mean"
    metric_std_col: str = "lds_std"
    output_prefix: str = "fig_grass_camera_ready_mlp_mnist"
    input_d: int = MLP_MNIST_FEATURE_DIM
    input_n: int = MLP_MNIST_CONFIG.batch_size
    title: str = f"GraSS on MNIST (d=4k, n={MLP_MNIST_CONFIG.batch_size})"
    pareto_width_in: float = 4.6
    pareto_height_in: float = 2.6
    bar_width_in: float = 6.8
    bar_height_in: float = 2.8
    min_font_size: int = 10
    ellipse_alpha: float = 0.25
    std_multiplier: float = 1.0
    baseline_method: str = g.METHOD_GRASS
    methods: tuple[str, ...] = (
        g.METHOD_GRASS,
        g.METHOD_FLASH_SKETCH,
        g.METHOD_SJLT_CUSPARSE,
        g.METHOD_GAUSSIAN_DENSE_CUBLAS,
        g.METHOD_SRHT_FWHT,
    )


CONFIG = GrassCameraReadyConfig()
