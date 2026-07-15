from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class BufferBlock:
    job_id: int
    block_id: int
    start_us: int
    end_us: int
    size_mib: float

    @property
    def lifetime_us(self) -> int:
        return max(0, self.end_us - self.start_us)

    @property
    def area(self) -> float:
        return self.size_mib * self.lifetime_us


@dataclass(frozen=True)
class Placement:
    buffer: BufferBlock
    address_mib: float


def overlaps_time(a: BufferBlock, b: BufferBlock) -> bool:
    return a.start_us < b.end_us and b.start_us < a.end_us


def overlaps_space(address_a: float, size_a: float, address_b: float, size_b: float) -> bool:
    return address_a < address_b + size_b and address_b < address_a + size_a


class MemoryLayoutScheduler:
    def __init__(self, memory_budget_mib: float, step_mib: float = 1.0, max_steps: int = 10000) -> None:
        self.memory_budget_mib = memory_budget_mib
        self.step_mib = step_mib
        self.max_steps = max_steps

    def place(self, buffers: Sequence[BufferBlock]) -> Optional[List[Placement]]:
        if not buffers:
            return []
        orderings = [
            sorted(buffers, key=lambda b: (-b.area, -b.size_mib, -b.lifetime_us)),
            sorted(buffers, key=lambda b: (-b.lifetime_us, -b.area, -b.size_mib)),
            sorted(buffers, key=lambda b: (-b.size_mib, -b.area, -b.lifetime_us)),
        ]
        for ordered in orderings:
            placements: List[Placement] = []
            if self._search(ordered, 0, placements, 0):
                return placements
        return None

    def _search(self, ordered: Sequence[BufferBlock], index: int, placements: List[Placement], steps: int) -> bool:
        if steps >= self.max_steps:
            return False
        if index >= len(ordered):
            return True
        buffer = ordered[index]
        candidate_addresses = self._candidate_addresses(buffer, placements)
        for address in candidate_addresses:
            if address + buffer.size_mib > self.memory_budget_mib:
                continue
            if self._fits(buffer, address, placements):
                placements.append(Placement(buffer=buffer, address_mib=address))
                if self._search(ordered, index + 1, placements, steps + 1):
                    return True
                placements.pop()
        return False

    def _candidate_addresses(self, buffer: BufferBlock, placements: Sequence[Placement]) -> List[float]:
        addresses = {0.0}
        for placement in placements:
            if overlaps_time(buffer, placement.buffer):
                addresses.add(round(placement.address_mib + placement.buffer.size_mib, 6))
        return sorted(addresses)

    def _fits(self, buffer: BufferBlock, address: float, placements: Sequence[Placement]) -> bool:
        for placement in placements:
            if not overlaps_time(buffer, placement.buffer):
                continue
            if overlaps_space(address, buffer.size_mib, placement.address_mib, placement.buffer.size_mib):
                return False
        return True


def buffers_for_job(job_id: int, start_us: int, latency_us: int, memory_mib: float, num_blocks: int = 4) -> List[BufferBlock]:
    if num_blocks <= 0:
        num_blocks = 1
    chunk_latency = max(1, latency_us // num_blocks)
    chunk_memory = max(1.0, memory_mib / num_blocks)
    buffers: List[BufferBlock] = []
    for block_id in range(num_blocks):
        begin = start_us + block_id * chunk_latency
        end = start_us + (block_id + 1) * chunk_latency
        if block_id == num_blocks - 1:
            end = start_us + latency_us
        buffers.append(BufferBlock(job_id=job_id, block_id=block_id, start_us=begin, end_us=end, size_mib=chunk_memory))
    return buffers
