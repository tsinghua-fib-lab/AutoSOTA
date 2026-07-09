import sys
from pathlib import Path
import time
import multiprocessing

src_path = Path(__file__).parents[1].joinpath("src")
assert src_path.exists(), f"Path does not exist: {src_path}"
sys.path.append(src_path.as_posix())

import torch
import numpy as np
from scipy import linalg as spla
from transformers import set_seed
from tqdm import tqdm
from threadpoolctl import threadpool_limits

from qera.statistic_profiler.scale import sqrtm_newton_schulz


def compute_sqrtm_cuda(A, numIters=200):
    A_sqrt = sqrtm_newton_schulz(A, numIters)
    return A_sqrt


def compute_sqrtm_numpy(A):
    A_sqrt = spla.sqrtm(A)
    return A_sqrt


if __name__ == "__main__":
    set_seed(42)
    num_runs = 4
    matrix_size = (13824, 13824)  # CUDA: 76277.03 ms = 1.1min, NumPy kraken: 6min

    A_list = [np.random.randn(*matrix_size).astype(np.float32) for _ in range(5)]
    profile_cuda = False
    profile_cpu = True

    if profile_cuda:
        A = torch.randn(*matrix_size).cuda()
        print("Warmup for CUDA")
        for _ in range(40):
            _ = torch.matmul(A, A.transpose(0, 1))

        # CUDA
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

        for i in tqdm(range(num_runs), total=num_runs, desc="CUDA"):
            A = A_list[i % 5]
            A = torch.from_numpy(A).cuda().reshape(1, matrix_size[0], matrix_size[1])
            start_events[i].record()
            compute_sqrtm_cuda(A, numIters=200)
            end_events[i].record()

        torch.cuda.synchronize()
        cuda_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        print(f"CUDA: {np.mean(cuda_times):.2f} ms")

    if profile_cpu:
        # NumPy
        print("NumPy")
        start_times = []
        end_times = []
        results = []
        with threadpool_limits(limits=multiprocessing.cpu_count(), user_api=None):
            for i in tqdm(range(num_runs), total=num_runs):
                A = A_list[i % 5]
                start_times.append(time.time())
                A_sqrtm = compute_sqrtm_numpy(A)
                end_times.append(time.time())
                if len(results) < 5:
                    results.append(A_sqrtm)

        print(f"NumPy: {np.mean(np.array(end_times) - np.array(start_times)):.2f} s")
        # output
        for i, A_sqrtm in enumerate(results):
            with np.printoptions(precision=4, edgeitems=10, linewidth=50):
                print(f"Result {i}:\n{A_sqrtm}")
