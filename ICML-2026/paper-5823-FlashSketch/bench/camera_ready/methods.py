from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import globals as g
from sketches.flashsketch import FlashSketchConfig
from sketches.gaussian_dense_cublas import GaussianDenseCublasConfig
from sketches.sjlt_grass_kernel import SjltGrassKernelConfig
from sketches.sjlt_cusparse import SjltCusparseConfig
from sketches.srht_fwht import SrhtFwhtConfig


CAMERA_READY_K_VALUES = [64, 256, 512, 1024, 2048, 4096]
CAMERA_READY_SEEDS = list(range(0, 10))


def camera_ready_methods(seeds=None, k_values=None):
    """Return the camera-ready method set."""
    seeds = CAMERA_READY_SEEDS if seeds is None else seeds
    k_values = CAMERA_READY_K_VALUES if k_values is None else k_values
    methods = []
    for k in k_values:
        for seed in seeds:
            methods.append(
                FlashSketchConfig(
                    k=k,
                    kappa=1,
                    s=4,
                    seed=seed,
                    dtype=g.DTYPE_FP32,
                )
            )
            methods.append(GaussianDenseCublasConfig(k=k, seed=seed, dtype=g.DTYPE_FP32))
            methods.append(SrhtFwhtConfig(k=k, seed=seed, dtype=g.DTYPE_FP32))
            methods.append(SjltGrassKernelConfig(k=k, s=4, seed=seed, dtype=g.DTYPE_FP32))
            methods.append(SjltCusparseConfig(k=k, s=4, seed=seed, dtype=g.DTYPE_FP32))
    return methods
