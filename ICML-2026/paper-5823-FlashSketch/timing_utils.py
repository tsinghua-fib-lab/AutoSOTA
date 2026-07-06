from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import time

import torch

import globals as g


class WallClockTimer:
    """Context manager for wall-clock timing (ms) with optional CUDA sync."""

    def __init__(self, use_cuda, sync=True):
        self._use_cuda = bool(use_cuda)
        self._sync = bool(sync)
        self.elapsed_ms = None
        self._start = None

    def __enter__(self):
        if self._sync and self._use_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._sync and self._use_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        self.elapsed_ms = (end - self._start) * 1000.0
        return False


class CudaEventTimer:
    """Context manager for CUDA event timing (ms)."""

    def __init__(self, use_cuda, sync=True):
        self._use_cuda = bool(use_cuda)
        self._sync = bool(sync)
        self.elapsed_ms = None
        self._start = None
        self._end = None

    def __enter__(self):
        if not self._use_cuda or not g.ENABLE_CUDA_TIMERS:
            self.elapsed_ms = None
            return self
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for CudaEventTimer.")
        if self._sync:
            torch.cuda.synchronize()
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._use_cuda or not g.ENABLE_CUDA_TIMERS:
            self.elapsed_ms = None
            return False
        self._end.record()
        if self._sync:
            torch.cuda.synchronize()
        self.elapsed_ms = float(self._start.elapsed_time(self._end))
        return False


def sum_optional(*values):
    """Sum values if all are not None; otherwise return None."""
    if any(value is None for value in values):
        return None
    return sum(values)
