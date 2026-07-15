from __future__ import annotations

from common import time_layout_solver
from rtinfer.layout import BufferBlock, MemoryLayoutScheduler


def spatial_utilization(buffers: list[BufferBlock], memory_mib: float) -> float:
    makespan = max(buffer.end_us for buffer in buffers) - min(buffer.start_us for buffer in buffers)
    area = sum(buffer.size_mib * (buffer.end_us - buffer.start_us) for buffer in buffers)
    return area / (memory_mib * makespan) if makespan > 0 else 0.0


def main() -> None:
    memory_mib = 6144.0
    static_rectangle = [
        BufferBlock(job_id=0, block_id=0, start_us=0, end_us=300_000, size_mib=1800.0),
        BufferBlock(job_id=1, block_id=0, start_us=20_000, end_us=130_000, size_mib=1500.0),
        BufferBlock(job_id=2, block_id=0, start_us=45_000, end_us=160_000, size_mib=1550.0),
        BufferBlock(job_id=3, block_id=0, start_us=80_000, end_us=220_000, size_mib=1850.0),
    ]
    stepped_kv = [
        BufferBlock(job_id=0, block_id=0, start_us=0, end_us=75_000, size_mib=500.0),
        BufferBlock(job_id=0, block_id=1, start_us=75_000, end_us=150_000, size_mib=950.0),
        BufferBlock(job_id=0, block_id=2, start_us=150_000, end_us=225_000, size_mib=1350.0),
        BufferBlock(job_id=0, block_id=3, start_us=225_000, end_us=300_000, size_mib=1800.0),
        BufferBlock(job_id=1, block_id=0, start_us=20_000, end_us=130_000, size_mib=1500.0),
        BufferBlock(job_id=2, block_id=0, start_us=45_000, end_us=160_000, size_mib=1550.0),
        BufferBlock(job_id=3, block_id=0, start_us=80_000, end_us=220_000, size_mib=1850.0),
    ]
    static_ok, static_ms = time_layout_solver(static_rectangle, memory_mib)
    stepped_ok, stepped_ms = time_layout_solver(stepped_kv, memory_mib)
    print("case,feasible,solver_ms,spatial_utilization")
    print(f"worst_case_rectangle,{int(static_ok)},{static_ms:.4f},{spatial_utilization(static_rectangle, memory_mib):.4f}")
    print(f"stepped_kv_cache,{int(stepped_ok)},{stepped_ms:.4f},{spatial_utilization(stepped_kv, memory_mib):.4f}")
    print("interpretation,stepped KV-cache packing exposes early decoding slack for other short-lived tasks")


if __name__ == "__main__":
    main()

