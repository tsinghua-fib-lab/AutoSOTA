from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtinfer.delta_graph import DeltaGraph
from rtinfer.layout import BufferBlock, MemoryLayoutScheduler
from rtinfer.model import BlockProfile, ExitProfile, Job, ModelProfile, TaskSpec, Variant
from rtinfer.scheduler import OnlineScheduler, PolicyName, SimulationResult


JETSON_EFFECTIVE_MEMORY_MIB = 6144.0
PRUNING_TIERS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)
OFFLINE_ACCURACY_TOLERANCE = 0.06
OFFLINE_MEMORY_BUDGET_MIB = JETSON_EFFECTIVE_MEMORY_MIB


@dataclass(frozen=True)
class RebuttalModelSpec:
    name: str
    input_shape: Tuple[int, ...]
    num_exits: int
    full_latency_ms: float
    full_memory_mib: float
    full_accuracy: float
    earliest_accuracy: float
    num_blocks: int = 8


def make_model(spec: RebuttalModelSpec) -> ModelProfile:
    num_blocks = max(spec.num_blocks, spec.num_exits)
    block_memory = spec.full_memory_mib / (num_blocks + 0.08)
    block_latency_us = int(spec.full_latency_ms * 1000 / num_blocks)
    blocks = tuple(
        BlockProfile(block_id=i, latency_us=block_latency_us, memory_mib=block_memory)
        for i in range(num_blocks)
    )
    exits: List[ExitProfile] = []
    for exit_idx in range(spec.num_exits):
        if spec.num_exits == 1:
            position = num_blocks - 1
            acc = spec.full_accuracy
        else:
            position = round(exit_idx * (num_blocks - 1) / (spec.num_exits - 1))
            ratio = exit_idx / (spec.num_exits - 1)
            acc = spec.earliest_accuracy + (spec.full_accuracy - spec.earliest_accuracy) * ratio
        exit_latency_us = max(1000, int(spec.full_latency_ms * 1000 * 0.08))
        exits.append(ExitProfile(exit_id=exit_idx, previous_block_id=position, latency_us=exit_latency_us, accuracy=acc))
    return ModelProfile(name=spec.name, dims=(1, *spec.input_shape), blocks=blocks, exits=tuple(exits))


def make_variants(
    model: ModelProfile,
    pruning_tiers: Sequence[float] = PRUNING_TIERS,
    chunk_prefix: str | None = None,
    accuracy_tolerance: float = OFFLINE_ACCURACY_TOLERANCE,
) -> List[Variant]:
    variants: List[Variant] = []
    prefix = chunk_prefix or model.name
    full_acc = model.full_accuracy
    for pruning in pruning_tiers:
        for exit_index, exit_profile in enumerate(model.exits):
            raw_latency_us = model.cumulative_latency_us(exit_index)
            raw_memory_mib = model.cumulative_memory_mib(exit_index)
            latency_scale = 1.0 - 0.52 * pruning
            memory_scale = 1.0 - 0.68 * pruning
            pruning_loss = 0.018 * (pruning / 0.25) ** 1.35 if pruning > 0 else 0.0
            pruning_only_accuracy_drop = 0.55 * pruning_loss
            if pruning_only_accuracy_drop > accuracy_tolerance:
                continue
            accuracy = max(0.0, min(full_acc, exit_profile.accuracy - 0.55 * pruning_loss))
            memory_mib = max(1.0, raw_memory_mib * memory_scale)
            bytes_to_load = int(memory_mib * 1024 * 1024)
            chunk_count = max(1, min(16, int(memory_mib // 128) + 1))
            chunk_keys = tuple(f"{prefix}:p{pruning:.2f}:e{exit_index}:c{i}" for i in range(chunk_count))
            variants.append(
                Variant(
                    model_name=model.name,
                    pruning=pruning,
                    exit_index=exit_index,
                    latency_us=max(1, int(raw_latency_us * latency_scale)),
                    memory_mib=memory_mib,
                    accuracy=accuracy,
                    bytes_to_load=bytes_to_load,
                    chunk_keys=chunk_keys,
                )
            )
    variants.sort(key=lambda item: (item.model_name, item.pruning, item.exit_index))
    return variants


def repeated_streams(model_name: str, count: int, deadline_ms: int, period_ms: int, duration_ms: int, shape: Tuple[int, ...]) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    for stream_id in range(count):
        tasks.append(
            TaskSpec(
                model_name=model_name,
                deadline_us=deadline_ms * 1000,
                period_us=period_ms * 1000,
                start_us=stream_id * 1000,
                end_us=duration_ms * 1000,
                shape=(1, *shape),
            )
        )
    return tasks


def run_policies(
    models: Dict[str, ModelProfile],
    atlas: Dict[str, List[Variant]],
    tasks: Sequence[TaskSpec],
    policies: Sequence[PolicyName] = ("pantheon", "rtinfer", "rtinfer-wo-alc", "rtinfer-wo-ms", "rtinfer-wo-dlp"),
    memory_mib: float = OFFLINE_MEMORY_BUDGET_MIB,
    duration_ms: int = 1000,
    bandwidth_gbps: float = 4.0,
) -> List[SimulationResult]:
    results: List[SimulationResult] = []
    for policy in policies:
        scheduler = OnlineScheduler(models, atlas, memory_mib, DeltaGraph(bandwidth_floor_gbps=bandwidth_gbps), policy)
        results.append(scheduler.run(tasks, duration_ms * 1000))
    return results


def completed_only_accuracy(result: SimulationResult) -> float:
    completed = [job for job in result.schedule_events if not job.missed and job.variant is not None]
    if not completed:
        return 0.0
    return sum(job.variant.accuracy for job in completed) / len(completed)


def print_result_table(title: str, results: Sequence[SimulationResult]) -> None:
    print(f"\n== {title} ==")
    print("policy,total,dmr,deadline_weighted_acc,completed_only_acc,avg_latency_ms,avg_load_ms")
    for result in results:
        print(
            f"{result.policy},{result.total_jobs},{result.deadline_miss_rate:.4f},"
            f"{result.average_accuracy:.4f},{completed_only_accuracy(result):.4f},"
            f"{result.average_latency_us / 1000.0:.3f},{result.average_load_us / 1000.0:.3f}"
        )


def time_layout_solver(buffers: Sequence[BufferBlock], memory_mib: float, rounds: int = 50) -> Tuple[bool, float]:
    scheduler = MemoryLayoutScheduler(memory_mib)
    start = perf_counter()
    solution = None
    for _ in range(rounds):
        solution = scheduler.place(buffers)
    elapsed_ms = (perf_counter() - start) * 1000.0 / rounds
    return solution is not None, elapsed_ms
