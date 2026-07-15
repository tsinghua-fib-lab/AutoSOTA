from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .model import ModelProfile, Variant


@dataclass(frozen=True)
class Chunk:
    key: str
    size_bytes: int


class DeltaGraph:
    def __init__(self, page_mib: float = 2.0, bandwidth_floor_gbps: float = 4.0, startup_us: int = 50) -> None:
        self.page_bytes = max(1, int(page_mib * 1024 * 1024))
        self.bandwidth_floor_bytes_per_us = max(1.0, bandwidth_floor_gbps * 1_000_000_000 / 1_000_000)
        self.startup_us = startup_us

    def chunks_for_variant(self, model: ModelProfile, pruning: float, exit_index: int) -> Tuple[Chunk, ...]:
        exit_profile = model.exits[exit_index]
        chunks: List[Chunk] = []
        for block in model.blocks[: exit_profile.previous_block_id + 1]:
            scaled_bytes = int(block.memory_mib * 1024 * 1024 * (1.0 - 0.72 * pruning))
            num_pages = max(1, (scaled_bytes + self.page_bytes - 1) // self.page_bytes)
            for page in range(num_pages):
                payload = f"{model.name}:b{block.block_id}:p{round(pruning, 4)}:page{page}"
                key = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
                chunks.append(Chunk(key=key, size_bytes=min(self.page_bytes, max(1, scaled_bytes - page * self.page_bytes))))
        branch_bytes = int(max(1.0, model.peak_memory_mib * 0.08) * 1024 * 1024)
        payload = f"{model.name}:exit{exit_index}:p{round(pruning, 4)}"
        chunks.append(Chunk(key=hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16], size_bytes=branch_bytes))
        return tuple(chunks)

    def missing_bytes(self, chunks: Sequence[Chunk], resident_keys: Iterable[str]) -> int:
        resident = set(resident_keys)
        return sum(chunk.size_bytes for chunk in chunks if chunk.key not in resident)

    def load_time_us(self, bytes_to_load: int) -> int:
        if bytes_to_load <= 0:
            return 0
        return int(self.startup_us + bytes_to_load / self.bandwidth_floor_bytes_per_us)


class Residency:
    def __init__(self, memory_budget_bytes: int) -> None:
        self.memory_budget_bytes = memory_budget_bytes
        self._chunks: "OrderedDict[str, int]" = OrderedDict()

    @property
    def used_bytes(self) -> int:
        return sum(self._chunks.values())

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(self._chunks.keys())

    def touch(self, chunks: Sequence[Chunk]) -> int:
        missing = 0
        for chunk in chunks:
            if chunk.key in self._chunks:
                size = self._chunks.pop(chunk.key)
                self._chunks[chunk.key] = size
            else:
                missing += chunk.size_bytes
                self._chunks[chunk.key] = chunk.size_bytes
        self._evict_to_budget()
        return missing

    def _evict_to_budget(self) -> None:
        while self.used_bytes > self.memory_budget_bytes and self._chunks:
            self._chunks.popitem(last=False)
