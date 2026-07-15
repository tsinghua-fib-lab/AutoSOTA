from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .delta_graph import DeltaGraph
from .model import ModelProfile, Variant


@dataclass(frozen=True)
class AtlasConfig:
    pruning_levels: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    accuracy_cap: float = 0.01
    lambda_memory: float = 0.15
    lambda_accuracy: float = 1000.0
    latency_scale: float = 1.0
    memory_scale: float = 1.0
    seed: int = 7


def estimate_variant(
    model: ModelProfile,
    pruning: float,
    exit_index: int,
    delta_graph: DeltaGraph,
    config: AtlasConfig = AtlasConfig(),
) -> Variant:
    raw_latency = model.cumulative_latency_us(exit_index)
    raw_memory = model.cumulative_memory_mib(exit_index)
    pruning = max(0.0, min(0.9, pruning))
    latency_scale = 1.0 - 0.58 * pruning
    memory_scale = 1.0 - 0.72 * pruning
    exit_accuracy = model.exits[exit_index].accuracy
    pruning_loss = 0.018 * (pruning / 0.1) ** 1.6 if pruning > 0 else 0.0
    calibrated_recovery = 0.45 * pruning_loss
    accuracy = max(0.0, min(1.0, exit_accuracy - pruning_loss + calibrated_recovery))
    chunks = delta_graph.chunks_for_variant(model, pruning, exit_index)
    bytes_to_load = int(sum(chunk.size_bytes for chunk in chunks) * config.memory_scale)
    return Variant(
        model_name=model.name,
        pruning=round(pruning, 4),
        exit_index=exit_index,
        latency_us=max(1, int(raw_latency * latency_scale * config.latency_scale)),
        memory_mib=max(1.0, raw_memory * memory_scale * config.memory_scale),
        accuracy=accuracy,
        bytes_to_load=bytes_to_load,
        chunk_keys=tuple(chunk.key for chunk in chunks),
    )


def pareto_filter(variants: Iterable[Variant]) -> List[Variant]:
    items = list(variants)
    keep: List[Variant] = []
    for candidate in items:
        dominated = False
        for other in items:
            if other is candidate:
                continue
            no_worse = (
                other.latency_us <= candidate.latency_us
                and other.memory_mib <= candidate.memory_mib
                and other.accuracy >= candidate.accuracy
                and other.bytes_to_load <= candidate.bytes_to_load
            )
            strictly_better = (
                other.latency_us < candidate.latency_us
                or other.memory_mib < candidate.memory_mib
                or other.accuracy > candidate.accuracy
                or other.bytes_to_load < candidate.bytes_to_load
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(candidate)
    keep.sort(key=lambda v: (v.model_name, v.memory_mib, v.latency_us, -v.accuracy))
    return keep


def build_variant_atlas(
    models: Dict[str, ModelProfile],
    delta_graph: DeltaGraph,
    config: AtlasConfig = AtlasConfig(),
) -> Dict[str, List[Variant]]:
    atlas: Dict[str, List[Variant]] = {}
    for model in models.values():
        variants: List[Variant] = []
        for pruning in config.pruning_levels:
            for exit_index in range(len(model.exits)):
                variant = estimate_variant(model, pruning, exit_index, delta_graph, config)
                accuracy_drop = model.full_accuracy - variant.accuracy
                if accuracy_drop <= max(config.accuracy_cap, model.full_accuracy):
                    variants.append(variant)
        atlas[model.name] = pareto_filter(variants)
    return atlas


def genetic_select(
    variants: Sequence[Variant],
    memory_budget_mib: float,
    population: int = 32,
    generations: int = 32,
    seed: int = 7,
) -> List[Variant]:
    if not variants:
        return []
    rng = random.Random(seed)
    base = list(variants)

    def score(chosen: Sequence[Variant]) -> float:
        latency = sum(v.latency_us for v in chosen) / 1000.0
        memory = max((v.memory_mib for v in chosen), default=0.0)
        accuracy_loss = sum(1.0 - v.accuracy for v in chosen) * 100.0
        overflow = max(0.0, memory - memory_budget_mib) * 1000.0
        return latency + 0.15 * memory + accuracy_loss + overflow

    candidates: List[List[Variant]] = []
    for _ in range(population):
        sample_size = rng.randint(1, min(len(base), 8))
        candidates.append(rng.sample(base, sample_size))
    for _ in range(generations):
        candidates.sort(key=score)
        survivors = candidates[: max(2, population // 4)]
        children = list(survivors)
        while len(children) < population:
            left, right = rng.sample(survivors, 2)
            merged = list({variant.key: variant for variant in left + right}.values())
            if rng.random() < 0.5:
                merged.append(rng.choice(base))
            if len(merged) > 1 and rng.random() < 0.3:
                merged.pop(rng.randrange(len(merged)))
            children.append(merged)
        candidates = children
    best = min(candidates, key=score)
    return pareto_filter(best)
