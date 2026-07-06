from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import globals as g

from sketches.gaussian_dense_cublas import GaussianDenseCublasConfig, sketch as gaussian_dense_cublas
from sketches.flashsketch import FlashSketchConfig, sketch as flashsketch
from sketches.srht_fwht import SrhtFwhtConfig, sketch as srht_fwht
from sketches.flashblockrow import FlashBlockRowConfig, sketch as flashblockrow
from sketches.sjlt_cusparse import SjltCusparseConfig, sketch as sjlt_cusparse
from sketches.sjlt_grass_kernel import SjltGrassKernelConfig, sketch as sjlt_grass_kernel

SKETCH_REGISTRY = {
    g.METHOD_GAUSSIAN_DENSE_CUBLAS: (gaussian_dense_cublas, GaussianDenseCublasConfig),
    g.METHOD_SJLT_CUSPARSE: (sjlt_cusparse, SjltCusparseConfig),
    g.METHOD_FLASH_SKETCH: (flashsketch, FlashSketchConfig),
    g.METHOD_SRHT_FWHT: (srht_fwht, SrhtFwhtConfig),
    g.METHOD_FLASH_BLOCK_ROW: (flashblockrow, FlashBlockRowConfig),
    g.METHOD_SJLT_GRASS_KERNEL: (sjlt_grass_kernel, SjltGrassKernelConfig),
}


def get_sketch_fn(method):
    """Return a sketch function from the registry."""
    if method not in SKETCH_REGISTRY:
        raise KeyError(f"Unknown sketch method: {method}")
    return SKETCH_REGISTRY[method][0]


def get_sketch_config_cls(method):
    """Return a sketch config class from the registry."""
    if method not in SKETCH_REGISTRY:
        raise KeyError(f"Unknown sketch method: {method}")
    return SKETCH_REGISTRY[method][1]
