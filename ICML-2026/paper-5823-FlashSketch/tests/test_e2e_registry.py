from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import replace

import pytest
import torch

import globals as g
from bench.e2e.tasks.gram_approx import GramApproxConfig, run_task as run_gram_approx
from bench.e2e.tasks.ose_error import OseErrorConfig, run_task as run_ose_error
from bench.e2e.tasks.ridge_regression import RidgeRegressionConfig, run_task as run_ridge
from bench.e2e.tasks.sketch_and_solve_ls import SketchAndSolveConfig, run_task as run_solve
from sketches.registry import SKETCH_REGISTRY
from torch_utils import manual_seed


REGISTRY_ITEMS = list(SKETCH_REGISTRY.items())
SHAPES_SMALL = [(64, 8), (128, 16), (32, 64)]
FULL_DIMS = [16, 32, 37, 64, 127, 128, 256, 257, 512, 521, 769, 1024, 1031, 1543, 2048]
FULL_NS = [8, 16, 32, 37, 64, 127, 128, 256, 257, 512, 521, 769, 1024, 1031, 1543, 2048]
SHAPES_FULL = [(d, n) for d in FULL_DIMS for n in FULL_NS]
TASKS = [
    (g.TASK_SKETCH_SOLVE_LS, run_solve, SketchAndSolveConfig),
    (g.TASK_RIDGE_REGRESSION, run_ridge, RidgeRegressionConfig),
    (g.TASK_GRAM_APPROX, run_gram_approx, GramApproxConfig),
    (g.TASK_OSE_ERROR, run_ose_error, OseErrorConfig),
]
TASKS_NO_RHS = {g.TASK_GRAM_APPROX, g.TASK_OSE_ERROR}
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


def _make_rhs(A, seed):
    """Return a consistent right-hand side vector for least squares."""
    manual_seed(seed)
    x_true = torch.randn(A.shape[1], device=A.device, dtype=A.dtype)
    return A @ x_true


def _select_k(shape):
    """Return a sketch dimension tied to the input shape."""
    return max(16, shape[1] // 2)


def _block_perm_compatible(cfg):
    """Return True if block-perm kernels support the requested config."""
    if cfg.kappa not in (1, 2, 4):
        return False
    if cfg.s not in (1, 2, 4, 8):
        return False
    if cfg.k < 64:
        return False
    return True



def _assert_nonzero_times(metrics):
    """Assert all available timing metrics are non-zero."""
    for key, value in metrics.items():
        if (key.endswith("_time_ms") or key.endswith("_time_cuda_ms")) and value is not None:
            assert value > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for e2e sketch tests")
@pytest.mark.small
@pytest.mark.parametrize("method, spec", REGISTRY_ITEMS)
@pytest.mark.parametrize("shape", SHAPES_SMALL)
def test_registry_e2e_tasks_run_small(method, spec, shape):
    """All sketches should run through each e2e task on multiple shapes."""
    sketch_fn, cfg_cls = spec
    device = torch.device("cuda")

    A = _make_matrix(shape, device, seed=11)
    b = _make_rhs(A, seed=19)
    cfg = _make_config(cfg_cls, k=_select_k(shape), seed=0)

    if method in BLOCK_PERM_METHODS and not _block_perm_compatible(cfg):
        with pytest.raises(ValueError):
            _ = sketch_fn(A, cfg)
        return

    for task_name, runner, task_cls in TASKS:
        cfg_task = cfg
        if task_name == g.TASK_SKETCH_SOLVE_LS:
            task_cfg = replace(task_cls(), compute_reference=False)
        else:
            task_cfg = task_cls()

        if task_name in TASKS_NO_RHS:
            metrics = runner(A, sketch_fn, cfg_task, task_cfg)
        else:
            metrics = runner(A, b, sketch_fn, cfg_task, task_cfg)

        assert metrics["task"] == task_name
        _assert_nonzero_times(metrics)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for e2e sketch tests")
@pytest.mark.full
@pytest.mark.parametrize("method, spec", REGISTRY_ITEMS)
@pytest.mark.parametrize("shape", SHAPES_FULL)
def test_registry_e2e_tasks_run_full(method, spec, shape):
    """All sketches should run through each e2e task on a large shape grid."""
    sketch_fn, cfg_cls = spec
    device = torch.device("cuda")

    A = _make_matrix(shape, device, seed=27)
    b = _make_rhs(A, seed=31)
    cfg = _make_config(cfg_cls, k=_select_k(shape), seed=0)

    if method in BLOCK_PERM_METHODS and not _block_perm_compatible(cfg):
        with pytest.raises(ValueError):
            _ = sketch_fn(A, cfg)
        return

    for task_name, runner, task_cls in TASKS:
        cfg_task = cfg
        if task_name == g.TASK_SKETCH_SOLVE_LS:
            task_cfg = replace(task_cls(), compute_reference=False)
        else:
            task_cfg = task_cls()

        if task_name in TASKS_NO_RHS:
            metrics = runner(A, sketch_fn, cfg_task, task_cfg)
        else:
            metrics = runner(A, b, sketch_fn, cfg_task, task_cfg)

        assert metrics["task"] == task_name
        _assert_nonzero_times(metrics)
