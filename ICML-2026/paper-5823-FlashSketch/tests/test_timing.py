from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass
import statistics

import pytest
import torch

from timing_utils import CudaEventTimer, WallClockTimer

pytestmark = pytest.mark.small


@dataclass(frozen=True)
class SleepConfig:
    """Config for GPU sleep-based timing tests."""

    cycles: int


def _sleep_sketch(A, cfg):
    """Sleep on the GPU for a fixed number of cycles."""
    torch.cuda._sleep(cfg.cycles)
    return A


def _matmul_sketch(A, cfg):
    """Run a square matmul to produce a size-dependent workload."""
    return A @ A


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for timing tests")
def test_time_sketch_nonzero():
    """Timing helper should report positive durations."""
    A = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
    cfg = SleepConfig(cycles=5_000_000)
    for _ in range(1):
        _ = _sleep_sketch(A, cfg)

    cuda_times = []
    for _ in range(3):
        with CudaEventTimer(True) as cuda_timer:
            _ = _sleep_sketch(A, cfg)
        cuda_times.append(cuda_timer.elapsed_ms)

    for _ in range(1):
        _ = _sleep_sketch(A, cfg)

    wall_times = []
    for _ in range(3):
        with WallClockTimer(True) as wall_timer:
            _ = _sleep_sketch(A, cfg)
        wall_times.append(wall_timer.elapsed_ms)

    assert min(wall_times) > 0.0
    assert min(cuda_times) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for timing tests")
def test_time_sketch_monotonic_sleep():
    """Longer GPU sleep should take longer in wall-clock time."""
    A = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
    short_cfg = SleepConfig(cycles=5_000_000)
    long_cfg = SleepConfig(cycles=20_000_000)

    short_wall = []
    short_cuda = []
    long_wall = []
    long_cuda = []

    for _ in range(3):
        for _ in range(1):
            _ = _sleep_sketch(A, short_cfg)

        with CudaEventTimer(True) as cuda_timer:
            _ = _sleep_sketch(A, short_cfg)
        short_cuda.append(cuda_timer.elapsed_ms)

        for _ in range(1):
            _ = _sleep_sketch(A, short_cfg)

        with WallClockTimer(True) as wall_timer:
            _ = _sleep_sketch(A, short_cfg)
        short_wall.append(wall_timer.elapsed_ms)

        for _ in range(1):
            _ = _sleep_sketch(A, long_cfg)

        with CudaEventTimer(True) as cuda_timer:
            _ = _sleep_sketch(A, long_cfg)
        long_cuda.append(cuda_timer.elapsed_ms)

        for _ in range(1):
            _ = _sleep_sketch(A, long_cfg)

        with WallClockTimer(True) as wall_timer:
            _ = _sleep_sketch(A, long_cfg)
        long_wall.append(wall_timer.elapsed_ms)

    assert statistics.mean(long_wall) > statistics.mean(short_wall) * 2.0
    assert statistics.mean(long_cuda) > statistics.mean(short_cuda) * 2.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for timing tests")
def test_time_sketch_monotonic_matmul():
    """Larger square matmul should take longer than a smaller one."""
    cfg = SleepConfig(cycles=0)

    A_small = torch.randn((1024, 1024), device="cuda", dtype=torch.float32)
    small_cuda = []
    for _ in range(3):
        for _ in range(1):
            _ = _matmul_sketch(A_small, cfg)

        with CudaEventTimer(True) as cuda_timer:
            _ = _matmul_sketch(A_small, cfg)
        small_cuda.append(cuda_timer.elapsed_ms)

    for _ in range(1):
        _ = _matmul_sketch(A_small, cfg)

    small_wall = []
    for _ in range(3):
        with WallClockTimer(True) as wall_timer:
            _ = _matmul_sketch(A_small, cfg)
        small_wall.append(wall_timer.elapsed_ms)

    del A_small
    torch.cuda.empty_cache()

    A_large = torch.randn((4096, 4096), device="cuda", dtype=torch.float32)
    large_cuda = []
    for _ in range(3):
        for _ in range(1):
            _ = _matmul_sketch(A_large, cfg)

        with CudaEventTimer(True) as cuda_timer:
            _ = _matmul_sketch(A_large, cfg)
        large_cuda.append(cuda_timer.elapsed_ms)

    for _ in range(1):
        _ = _matmul_sketch(A_large, cfg)

    large_wall = []
    for _ in range(3):
        with WallClockTimer(True) as wall_timer:
            _ = _matmul_sketch(A_large, cfg)
        large_wall.append(wall_timer.elapsed_ms)

    assert statistics.median(large_wall) > statistics.median(small_wall)
    assert statistics.median(large_cuda) > statistics.median(small_cuda)
