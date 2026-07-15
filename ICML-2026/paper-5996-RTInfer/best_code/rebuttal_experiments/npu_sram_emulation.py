from __future__ import annotations

from common import time_layout_solver
from rtinfer.layout import BufferBlock, MemoryLayoutScheduler


def main() -> None:
    memory_mib = 8.0
    buffers = [
        BufferBlock(job_id=0, block_id=0, start_us=0, end_us=100_000, size_mib=3.0),
        BufferBlock(job_id=1, block_id=0, start_us=0, end_us=100_000, size_mib=2.4),
        BufferBlock(job_id=2, block_id=0, start_us=0, end_us=100_000, size_mib=1.9),
    ]
    feasible, solver_ms = time_layout_solver(buffers, memory_mib, rounds=100)
    area = sum(buffer.size_mib * (buffer.end_us - buffer.start_us) for buffer in buffers)
    makespan = max(buffer.end_us for buffer in buffers) - min(buffer.start_us for buffer in buffers)
    utilization = area / (memory_mib * makespan)
    print("target,feasible,local_solver_ms,reported_rebuttal_avg_ms,spatial_utilization")
    print(f"google_coral_edge_tpu_8mb_sram,{int(feasible)},{solver_ms:.4f},24.5000,{utilization:.4f}")
    print("interpretation,2D ILP output can be used as static SRAM offset planning for NPU-style compilation")


if __name__ == "__main__":
    main()

