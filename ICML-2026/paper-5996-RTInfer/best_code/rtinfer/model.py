from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class ExitProfile:
    exit_id: int
    previous_block_id: int
    latency_us: int
    accuracy: float


@dataclass(frozen=True)
class BlockProfile:
    block_id: int
    latency_us: int
    memory_mib: float


@dataclass(frozen=True)
class ModelProfile:
    name: str
    dims: Tuple[int, ...]
    blocks: Tuple[BlockProfile, ...]
    exits: Tuple[ExitProfile, ...]

    @property
    def full_exit(self) -> ExitProfile:
        return self.exits[-1]

    @property
    def full_accuracy(self) -> float:
        return max(exit_profile.accuracy for exit_profile in self.exits)

    @property
    def peak_memory_mib(self) -> float:
        if not self.blocks:
            return 0.0
        return max(block.memory_mib for block in self.blocks)

    def submodel_latency_us(self, current_block_id: int, exit_index: int) -> int:
        exit_profile = self.exits[exit_index]
        latency = exit_profile.latency_us
        start = max(current_block_id, 0)
        for block in self.blocks[start : exit_profile.previous_block_id + 1]:
            latency += block.latency_us
        return latency

    def cumulative_latency_us(self, exit_index: int) -> int:
        return self.submodel_latency_us(-1, exit_index)

    def cumulative_memory_mib(self, exit_index: int) -> float:
        exit_profile = self.exits[exit_index]
        block_memory = sum(block.memory_mib for block in self.blocks[: exit_profile.previous_block_id + 1])
        branch_memory = max(1.0, self.peak_memory_mib * 0.08)
        return block_memory + branch_memory


@dataclass(frozen=True)
class Variant:
    model_name: str
    pruning: float
    exit_index: int
    latency_us: int
    memory_mib: float
    accuracy: float
    bytes_to_load: int
    chunk_keys: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> Tuple[str, float, int]:
        return (self.model_name, self.pruning, self.exit_index)

    @property
    def relative_accuracy(self) -> float:
        return self.accuracy


@dataclass(frozen=True)
class TaskSpec:
    model_name: str
    deadline_us: int
    period_us: int
    start_us: int
    end_us: int
    shape: Tuple[int, ...]
    priority: str = "RT"


@dataclass
class Job:
    job_id: int
    task: TaskSpec
    release_us: int
    absolute_deadline_us: int
    variant: Variant | None = None
    start_us: int | None = None
    finish_us: int | None = None
    missed: bool = False
    load_us: int = 0

    @property
    def slack_us(self) -> int:
        base = self.finish_us if self.finish_us is not None else self.release_us
        return self.absolute_deadline_us - base


def iter_jobs(tasks: Sequence[TaskSpec], duration_us: int) -> Iterable[Job]:
    job_id = 0
    releases: List[Tuple[int, TaskSpec]] = []
    for task in tasks:
        t = task.start_us
        while t < min(task.end_us, duration_us):
            releases.append((t, task))
            t += task.period_us
    releases.sort(key=lambda item: (item[0], item[1].deadline_us, item[1].model_name))
    for release_us, task in releases:
        yield Job(
            job_id=job_id,
            task=task,
            release_us=release_us,
            absolute_deadline_us=release_us + task.deadline_us,
        )
        job_id += 1


def group_variants_by_model(variants: Iterable[Variant]) -> Dict[str, List[Variant]]:
    grouped: Dict[str, List[Variant]] = {}
    for variant in variants:
        grouped.setdefault(variant.model_name, []).append(variant)
    for model_variants in grouped.values():
        model_variants.sort(key=lambda v: (v.pruning, v.exit_index))
    return grouped
