from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from dataclasses import dataclass


MLP_MNIST_FEATURE_DIM = 109386


@dataclass(frozen=True)
class ProjectionBenchConfig:
    """Configuration for FlashSketch vs SJLT projection microbenchmarks."""

    proj_types: tuple[str, ...] = ("flashsketch",)
    seed_values: tuple[int, ...] = (42,)
    d_values: tuple[int, ...] = (MLP_MNIST_FEATURE_DIM,)
    k_values: tuple[int, ...] = (1024, 2048, 4096)
    batch_size: int = 512
    input_nonzero_fraction: float = 0.001
    skip_zeros_values: tuple[bool, ...] = (False, True)
    warmup_iters: int = 20
    timed_iters: int = 100
    device: str = "cuda"
    require_clean_repo: bool = True
    send_slack: bool = False
    progress_every: int = 10
    kappa: int = 1
    s: int = 4
    block_rows: int = 128


@dataclass(frozen=True)
class MlpMnistConfig:
    """Configuration for the external GraSS MLP+MNIST experiment."""

    proj_dim: int = 1024
    proj_dims: tuple[int, ...] = (1024, 2048, 4096)
    batch_size: int = 512
    proj_max_batch_size: int = 512
    sjlt_c: int = 4
    seed: int = 42
    repeat_seeds: tuple[int, ...] = (42,)
    val_ratio: float = 0.1
    device: str = "cuda"
    flashsketch_kappa: int = 1
    flashsketch_s: int = 4
    flashsketch_block_rows: int = 128
    flashsketch_seed: int | None = None
    flashsketch_skip_zeros: bool = True
    proj_types: tuple[str, ...] = (
        "grass",
        "flashsketch_grass",
        "sjlt_cusparse_grass",
        "gaussian_dense_cublas_grass",
        "srht_fwht_grass",
    )
    require_clean_repo: bool = True
    send_slack: bool = False
    sparsity_log_batches: int = 3
    mlp_activation: str = "relu"
    mlp_dropout_rate: float = 0.1


@dataclass(frozen=True)
class ResNetCifarConfig:
    """Configuration for the external GraSS ResNet+CIFAR experiment."""

    proj_dim: int = 4096
    proj_dims: tuple[int, ...] = (4096,)
    seed: int = 42
    val_ratio: float = 0.1
    device: str = "cuda"
    sjlt_c: int = 4
    flashsketch_kappa: int = 1
    flashsketch_s: int = 4
    flashsketch_block_rows: int = 128
    flashsketch_seed: int | None = None
    flashsketch_skip_zeros: bool = True
    proj_types: tuple[str, ...] = (
        "grass",
        "flashsketch_grass",
        "sjlt_cusparse_grass",
        "gaussian_dense_cublas_grass",
        "srht_fwht_grass",
    )
    require_clean_repo: bool = True
    send_slack: bool = False
    status_every_s: int = 60


PROJECTION_BENCH_CONFIG = ProjectionBenchConfig()
MLP_MNIST_CONFIG = MlpMnistConfig()
RESNET_CIFAR_CONFIG = ResNetCifarConfig()
