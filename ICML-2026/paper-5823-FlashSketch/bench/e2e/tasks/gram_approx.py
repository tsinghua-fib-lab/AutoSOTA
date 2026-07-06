from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass

import torch

import globals as g
from bench.e2e.metrics import gram_errors
from sketches.sketch_utils import finalize_sketch_output, prepare_sketch_input
from timing_utils import CudaEventTimer, WallClockTimer


@dataclass(frozen=True)
class GramApproxConfig:
    """Config for Gram approximation task."""

    task: str = g.TASK_GRAM_APPROX


def run_task(A, sketch_fn, method_cfg, task_cfg):
    """Run Gram approximation and return metrics."""
    use_cuda = A.is_cuda
    A_work, was_transposed = prepare_sketch_input(A, method_cfg)
    with CudaEventTimer(use_cuda) as sketch_cuda_timer:
        _ = sketch_fn(A_work, method_cfg)

    with WallClockTimer(use_cuda) as sketch_wall_timer:
        SA = sketch_fn(A_work, method_cfg)

    sketch_ms = sketch_wall_timer.elapsed_ms
    sketch_cuda_ms = sketch_cuda_timer.elapsed_ms
    SA = finalize_sketch_output(SA, was_transposed)
    errors = gram_errors(A, SA)

    result = {
        "task": task_cfg.task,
        "sketch_time_ms": sketch_ms,
        "sketch_time_cuda_ms": sketch_cuda_ms,
    }
    result.update(errors)
    return result
