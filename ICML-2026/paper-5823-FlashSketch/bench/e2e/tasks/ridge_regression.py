from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass

import torch

import globals as g
from bench.e2e.metrics import relative_residual
from sketches.sketch_utils import finalize_sketch_output, prepare_sketch_input
from timing_utils import CudaEventTimer, WallClockTimer, sum_optional


@dataclass(frozen=True)
class RidgeRegressionConfig:
    """Config for ridge regression with sketching."""

    task: str = g.TASK_RIDGE_REGRESSION
    lambda_reg: float = 1e-3


def run_task(A, b, sketch_fn, method_cfg, task_cfg):
    """Run ridge regression with sketching and return metrics."""
    use_cuda = A.is_cuda
    A_work, A_transposed = prepare_sketch_input(A, method_cfg)
    b_work, b_transposed = prepare_sketch_input(b.unsqueeze(1), method_cfg)
    _ = sketch_fn(A_work, method_cfg)
    _ = sketch_fn(b_work, method_cfg)
    with CudaEventTimer(use_cuda) as sketch_cuda_timer:
        _ = sketch_fn(A_work, method_cfg)
        _ = sketch_fn(b_work, method_cfg)

    _ = sketch_fn(A_work, method_cfg)
    _ = sketch_fn(b_work, method_cfg)
    with WallClockTimer(use_cuda) as sketch_wall_timer:
        SA = sketch_fn(A_work, method_cfg)
        Sb = sketch_fn(b_work, method_cfg)

    sketch_ms = sketch_wall_timer.elapsed_ms
    sketch_cuda_ms = sketch_cuda_timer.elapsed_ms

    SA = finalize_sketch_output(SA, A_transposed)
    Sb = finalize_sketch_output(Sb, b_transposed)

    gram = SA.T @ SA
    reg = task_cfg.lambda_reg * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    rhs = SA.T @ Sb
    _ = torch.linalg.solve(gram + reg, rhs)
    with CudaEventTimer(use_cuda) as solve_cuda_timer:
        gram = SA.T @ SA
        reg = task_cfg.lambda_reg * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        rhs = SA.T @ Sb
        _ = torch.linalg.solve(gram + reg, rhs)

    gram = SA.T @ SA
    reg = task_cfg.lambda_reg * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    rhs = SA.T @ Sb
    _ = torch.linalg.solve(gram + reg, rhs)
    with WallClockTimer(use_cuda) as solve_wall_timer:
        gram = SA.T @ SA
        reg = task_cfg.lambda_reg * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        rhs = SA.T @ Sb
        x = torch.linalg.solve(gram + reg, rhs)
    x = x.squeeze(1)

    solve_ms = solve_wall_timer.elapsed_ms
    solve_cuda_ms = solve_cuda_timer.elapsed_ms

    residual = relative_residual(A, x, b)

    return {
        "task": task_cfg.task,
        "lambda_reg": task_cfg.lambda_reg,
        "sketch_time_ms": sketch_ms,
        "sketch_time_cuda_ms": sketch_cuda_ms,
        "solve_time_ms": solve_ms,
        "solve_time_cuda_ms": solve_cuda_ms,
        "total_time_ms": sketch_ms + solve_ms,
        "total_time_cuda_ms": sum_optional(sketch_cuda_ms, solve_cuda_ms),
        "residual": residual,
    }
