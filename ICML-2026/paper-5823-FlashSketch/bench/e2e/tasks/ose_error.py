from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass

import torch

import globals as g
from bench.e2e.metrics import ose_errors
from sketches.sketch_utils import finalize_sketch_output, prepare_sketch_input
from timing_utils import CudaEventTimer, WallClockTimer
from torch_utils import manual_seed


@dataclass(frozen=True)
class OseErrorConfig:
    """Config for OSE error task."""

    task: str = g.TASK_OSE_ERROR
    r: int = 256
    ose_variant: str = "colspace"
    probes: int = 0


def _resolve_r(task_cfg, d, n):
    """Return the effective subspace dimension r."""
    r = int(task_cfg.r)
    if task_cfg.probes > 0:
        r = int(task_cfg.probes)
    return max(1, min(r, d, n))


def _make_subspace(A, task_cfg, r, seed):
    """Return an orthonormal basis Q for the OSE metric."""
    variant = task_cfg.ose_variant
    if task_cfg.probes > 0:
        variant = "random"

    if variant == "colspace":
        Q, _ = torch.linalg.qr(A, mode="reduced")
        if Q.shape[1] > r:
            Q = Q[:, :r]
        return Q, variant
    if variant == "random":
        manual_seed(seed)
        Q = torch.randn((A.shape[0], r), device=A.device, dtype=A.dtype)
        Q, _ = torch.linalg.qr(Q, mode="reduced")
        if Q.shape[1] > r:
            Q = Q[:, :r]
        return Q, variant
    raise ValueError(f"Unknown ose_variant: {variant}")


def run_task(A, sketch_fn, method_cfg, task_cfg):
    """Run OSE error evaluation and return metrics."""
    use_cuda = A.is_cuda
    A_work, _ = prepare_sketch_input(A, method_cfg)
    with CudaEventTimer(use_cuda) as sketch_cuda_timer:
        _ = sketch_fn(A_work, method_cfg)

    with WallClockTimer(use_cuda) as sketch_wall_timer:
        _ = sketch_fn(A_work, method_cfg)

    sketch_ms = sketch_wall_timer.elapsed_ms
    sketch_cuda_ms = sketch_cuda_timer.elapsed_ms

    d, n = A.shape
    r = _resolve_r(task_cfg, d, n)
    seed = int(getattr(method_cfg, "seed", 0))
    Q, variant = _make_subspace(A, task_cfg, r, seed)

    Q_work, Q_transposed = prepare_sketch_input(Q, method_cfg)
    SQ = sketch_fn(Q_work, method_cfg)
    SQ = finalize_sketch_output(SQ, Q_transposed)
    metrics = ose_errors(SQ.float())

    return {
        "task": task_cfg.task,
        "sketch_time_ms": sketch_ms,
        "sketch_time_cuda_ms": sketch_cuda_ms,
        "ose_variant": variant,
        "ose_probes": int(task_cfg.probes),
        "ose_r": int(r),
        **metrics,
    }
