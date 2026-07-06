from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import replace
import statistics

import pytest
import torch

import globals as g
from bench.e2e.metrics import gram_errors
from sketches.registry import SKETCH_REGISTRY
from sketches.sketch_utils import finalize_sketch_output, prepare_sketch_input
from timing_utils import CudaEventTimer, WallClockTimer
from torch_utils import manual_seed


REGISTRY_ITEMS = list(SKETCH_REGISTRY.items())
SHAPES_SMALL = [(64, 8), (64, 33), (128, 16)]
FULL_DIMS = [16, 32, 37, 64, 127, 128, 256, 257, 512, 521, 769, 1024, 1031, 1543, 2048]
FULL_NS = [8, 16, 32, 37, 64, 127, 128, 256, 257, 512, 521, 769, 1024, 1031, 1543, 2048]
SHAPES_FULL = [(d, n) for d in FULL_DIMS for n in FULL_NS]
K_SMALL = 16
K_LARGE = 64
SEEDS = [0, 1, 2, 3, 4]
SMALL_WARMUP = 1
SMALL_REPEATS = 2
FULL_WARMUP = 1
FULL_REPEATS = 1
BLOCK_PERM_METHODS = {
    g.METHOD_FLASH_SKETCH,
}


def _make_config(cfg_cls, k, seed):
    """Return a sketch config with common overrides applied."""
    return replace(cfg_cls(), k=k, seed=seed, dtype=g.DTYPE_FP32)


def _make_matrix(shape, device, seed):
    """Return a deterministic dense matrix for testing."""
    manual_seed(seed)
    return torch.randn(shape, device=device, dtype=torch.float32)


def _distortion(A, SA):
    """Compute the relative Frobenius Gram distortion for SA."""
    metrics = gram_errors(A, SA)
    return metrics["gram_fro_rel"]


def _select_k(shape):
    """Return a sketch dimension tied to the input shape."""
    return max(K_SMALL, shape[1] // 2)


def _block_perm_compatible(cfg):
    """Return True if block-perm kernels support the requested config."""
    if cfg.kappa not in (1, 2, 4):
        return False
    if cfg.s not in (1, 2, 4, 8):
        return False
    if cfg.k < 64:
        return False
    return True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for sketch registry tests")
@pytest.mark.small
@pytest.mark.parametrize("method, spec", REGISTRY_ITEMS)
@pytest.mark.parametrize("shape", SHAPES_SMALL)
def test_registry_shapes_and_dtype_small(method, spec, shape):
    """All sketches should return (k, n) fp32 outputs for multiple shapes."""
    sketch_fn, cfg_cls = spec
    cfg = _make_config(cfg_cls, k=_select_k(shape), seed=0)
    A = _make_matrix(shape, torch.device("cuda"), seed=123)
    A_work, was_transposed = prepare_sketch_input(A, cfg)

    if method in BLOCK_PERM_METHODS and not _block_perm_compatible(cfg):
        with pytest.raises(ValueError):
            _ = sketch_fn(A, cfg)
        return

    SA = sketch_fn(A_work, cfg)
    SA = finalize_sketch_output(SA, was_transposed)
    assert SA.shape == (cfg.k, shape[1])
    assert SA.dtype == torch.float32
    assert SA.device == A.device
    for _ in range(SMALL_WARMUP):
        _ = sketch_fn(A_work, cfg)

    cuda_times = []
    for _ in range(SMALL_REPEATS):
        with CudaEventTimer(True) as cuda_timer:
            _ = sketch_fn(A_work, cfg)
        cuda_times.append(cuda_timer.elapsed_ms)

    for _ in range(SMALL_WARMUP):
        _ = sketch_fn(A_work, cfg)

    wall_times = []
    for _ in range(SMALL_REPEATS):
        with WallClockTimer(True) as wall_timer:
            _ = sketch_fn(A_work, cfg)
        wall_times.append(wall_timer.elapsed_ms)

    assert statistics.mean(wall_times) > 0.0
    assert statistics.mean(cuda_times) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for sketch registry tests")
@pytest.mark.full
@pytest.mark.parametrize("method, spec", REGISTRY_ITEMS)
@pytest.mark.parametrize("shape", SHAPES_FULL)
def test_registry_shapes_and_dtype_full(method, spec, shape):
    """All sketches should run across a large grid of shapes."""
    sketch_fn, cfg_cls = spec
    cfg = _make_config(cfg_cls, k=_select_k(shape), seed=0)
    A = _make_matrix(shape, torch.device("cuda"), seed=321)
    A_work, was_transposed = prepare_sketch_input(A, cfg)

    if method in BLOCK_PERM_METHODS and not _block_perm_compatible(cfg):
        with pytest.raises(ValueError):
            _ = sketch_fn(A, cfg)
        return

    SA = sketch_fn(A_work, cfg)
    SA = finalize_sketch_output(SA, was_transposed)
    assert SA.shape == (cfg.k, shape[1])
    assert SA.dtype == torch.float32
    assert SA.device == A.device
    for _ in range(FULL_WARMUP):
        _ = sketch_fn(A_work, cfg)

    cuda_times = []
    for _ in range(FULL_REPEATS):
        with CudaEventTimer(True) as cuda_timer:
            _ = sketch_fn(A_work, cfg)
        cuda_times.append(cuda_timer.elapsed_ms)

    for _ in range(FULL_WARMUP):
        _ = sketch_fn(A_work, cfg)

    wall_times = []
    for _ in range(FULL_REPEATS):
        with WallClockTimer(True) as wall_timer:
            _ = sketch_fn(A_work, cfg)
        wall_times.append(wall_timer.elapsed_ms)

    assert statistics.mean(wall_times) > 0.0
    assert statistics.mean(cuda_times) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for sketch registry tests")
@pytest.mark.small
@pytest.mark.parametrize("method, spec", REGISTRY_ITEMS)
def test_registry_distortion_monotonic_small(method, spec):
    """Median Gram distortion should decrease as k increases."""
    sketch_fn, cfg_cls = spec
    A = _make_matrix((128, 64), torch.device("cuda"), seed=456)

    cfg_small = _make_config(cfg_cls, k=K_SMALL, seed=0)
    cfg_large = _make_config(cfg_cls, k=K_LARGE, seed=0)
    A_work, was_transposed = prepare_sketch_input(A, cfg_small)
    if method in BLOCK_PERM_METHODS and not _block_perm_compatible(cfg_small):
        with pytest.raises(ValueError):
            _ = sketch_fn(A, cfg_small)
        return

    for _ in range(SMALL_WARMUP):
        _ = sketch_fn(A_work, cfg_small)

    cuda_times = []
    for _ in range(SMALL_REPEATS):
        with CudaEventTimer(True) as cuda_timer:
            _ = sketch_fn(A_work, cfg_small)
        cuda_times.append(cuda_timer.elapsed_ms)

    for _ in range(SMALL_WARMUP):
        _ = sketch_fn(A_work, cfg_small)

    wall_times = []
    for _ in range(SMALL_REPEATS):
        with WallClockTimer(True) as wall_timer:
            _ = sketch_fn(A_work, cfg_small)
        wall_times.append(wall_timer.elapsed_ms)

    assert statistics.mean(wall_times) > 0.0
    assert statistics.mean(cuda_times) > 0.0

    distort_small = []
    distort_large = []
    for seed in SEEDS:
        cfg_small = _make_config(cfg_cls, k=K_SMALL, seed=seed)
        SA_small = sketch_fn(A_work, cfg_small)
        SA_small = finalize_sketch_output(SA_small, was_transposed)
        distort_small.append(_distortion(A, SA_small))

        cfg_large = _make_config(cfg_cls, k=K_LARGE, seed=seed)
        SA_large = sketch_fn(A_work, cfg_large)
        SA_large = finalize_sketch_output(SA_large, was_transposed)
        distort_large.append(_distortion(A, SA_large))
    median_small = statistics.median(distort_small)
    median_large = statistics.median(distort_large)
    assert median_large <= median_small


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for sketch registry tests")
@pytest.mark.full
@pytest.mark.parametrize("method, spec", REGISTRY_ITEMS)
def test_registry_distortion_monotonic_full(method, spec):
    """Median Gram distortion should decrease as k increases on a larger shape."""
    sketch_fn, cfg_cls = spec
    A = _make_matrix((256, 128), torch.device("cuda"), seed=654)

    cfg_small = _make_config(cfg_cls, k=K_SMALL, seed=0)
    cfg_large = _make_config(cfg_cls, k=K_LARGE, seed=0)
    A_work, was_transposed = prepare_sketch_input(A, cfg_small)
    if method in BLOCK_PERM_METHODS and not _block_perm_compatible(cfg_small):
        with pytest.raises(ValueError):
            _ = sketch_fn(A, cfg_small)
        return

    for _ in range(FULL_WARMUP):
        _ = sketch_fn(A_work, cfg_small)

    cuda_times = []
    for _ in range(FULL_REPEATS):
        with CudaEventTimer(True) as cuda_timer:
            _ = sketch_fn(A_work, cfg_small)
        cuda_times.append(cuda_timer.elapsed_ms)

    for _ in range(FULL_WARMUP):
        _ = sketch_fn(A_work, cfg_small)

    wall_times = []
    for _ in range(FULL_REPEATS):
        with WallClockTimer(True) as wall_timer:
            _ = sketch_fn(A_work, cfg_small)
        wall_times.append(wall_timer.elapsed_ms)

    assert statistics.mean(wall_times) > 0.0
    assert statistics.mean(cuda_times) > 0.0

    distort_small = []
    distort_large = []
    for seed in SEEDS:
        cfg_small = _make_config(cfg_cls, k=K_SMALL, seed=seed)
        SA_small = sketch_fn(A_work, cfg_small)
        SA_small = finalize_sketch_output(SA_small, was_transposed)
        distort_small.append(_distortion(A, SA_small))

        cfg_large = _make_config(cfg_cls, k=K_LARGE, seed=seed)
        SA_large = sketch_fn(A_work, cfg_large)
        SA_large = finalize_sketch_output(SA_large, was_transposed)
        distort_large.append(_distortion(A, SA_large))
    median_small = statistics.median(distort_small)
    median_large = statistics.median(distort_large)
    assert median_large <= median_small
