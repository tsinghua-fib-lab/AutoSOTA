from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import globals as g
from sketches.flashsketch import FlashSketchConfig
from sketches.flashblockrow import FlashBlockRowConfig
from sketches.gaussian_dense_cublas import GaussianDenseCublasConfig
from sketches.sjlt_grass_kernel import SjltGrassKernelConfig
from sketches.sjlt_cusparse import SjltCusparseConfig
from sketches.srht_fwht import SrhtFwhtConfig


ABLATION_K_VALUES = [64, 256, 512, 1024, 2048, 4096]
ABLATION_SEEDS = list(range(10))
ABLATION_FLASH_COMBOS = (
    (1, 1),
    (2, 1),
    (4, 1),
    (1, 2),
    (2, 2),
    (1, 4),
)
ABLATION_SJLT_S = (1, 2, 4)


def ablation_methods(seeds=None, k_values=None):
    """Return the ablation sweep method configs."""
    seeds = ABLATION_SEEDS if seeds is None else seeds
    k_values = ABLATION_K_VALUES if k_values is None else k_values

    methods = []
    for k in k_values:
        for seed in seeds:
            for kappa, s in ABLATION_FLASH_COMBOS:
                methods.append(
                    FlashSketchConfig(
                        k=k,
                        kappa=kappa,
                        s=s,
                        seed=seed,
                        dtype=g.DTYPE_FP32,
                    )
                )
                methods.append(
                    FlashBlockRowConfig(
                        k=k,
                        kappa=kappa,
                        s=s,
                        seed=seed,
                        dtype=g.DTYPE_FP32,
                    )
                )

            for s in ABLATION_SJLT_S:
                methods.append(
                    SjltGrassKernelConfig(
                        k=k,
                        s=s,
                        seed=seed,
                        dtype=g.DTYPE_FP32,
                    )
                )
                methods.append(
                    SjltCusparseConfig(
                        k=k,
                        s=s,
                        seed=seed,
                        dtype=g.DTYPE_FP32,
                    )
                )

            methods.append(GaussianDenseCublasConfig(k=k, seed=seed, dtype=g.DTYPE_FP32))
            methods.append(SrhtFwhtConfig(k=k, seed=seed, dtype=g.DTYPE_FP32))

    return methods
