from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass

import globals as g
from data.synthetic import SyntheticDatasetConfig
from data.suitesparse.load import SuiteSparseDatasetConfig


@dataclass(frozen=True)
class FlashBlockRowTuneConfig:
    """Config for tuning the FlashBlockRow CUDA kernel."""

    seeds: tuple = (0, 1, 2)
    k_values: tuple = (512, 1024, 2048, 4096)
    block_size_values: tuple = (128,)
    kappa_values: tuple = (2,)
    s_values: tuple = (2,)
    tc_values: tuple = (32,)
    warmup: int = 2
    repeats: int = 2
    error_tolerance: float = 0.1
    require_clean_repo: bool = False
    send_slack: bool = False
    datasets: tuple = (
        SyntheticDatasetConfig(
            name="synthetic_ls_tall_16k",
            d=16384,
            n=1024,
            distribution=g.DIST_GAUSSIAN,
            seed=0,
        ),
        SuiteSparseDatasetConfig(
            group="HB",
            name="1138_bus",
            submatrix_rows=1138,
            submatrix_cols=1138,
            seed=1,
        ),
    )


CONFIG = FlashBlockRowTuneConfig()
