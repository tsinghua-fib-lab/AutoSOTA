from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"

MEM_BUDGET_MIB = 4096
RESERVE_IF_MIB = 512
HORIZON_MS = 760


@dataclass(frozen=True)
class TaskSpec:
    task: str
    label: str
    model: str
    period_ms: int
    arrival_ms: int
    deadline_ms: int


@dataclass(frozen=True)
class VariantChoice:
    task: str
    system: str
    variant: str
    pruning: float | str
    exit_point: int | str
    latency_ms: int
    memory_mib: int
    accuracy: float
    missing_mib: int
    selected: bool
    note: str


@dataclass(frozen=True)
class BufferEvent:
    system: str
    job: str
    task: str
    label: str
    start_ms: int
    end_ms: int
    addr_mib: int
    memory_mib: int
    kind: str
    stage: str
    variant: str
    exit_point: int | str
    pruning: float | str
    accuracy: float | str
    detail: str
    load_type: str = "-"
    missing_mib_delta_graph: int | str = "-"

    @property
    def top_mib(self) -> int:
        return self.addr_mib + self.memory_mib


TASKS = (
    TaskSpec("task_i", "Task I / Traffic light detection", "MobileNetv2-SSDLite-300", 180, 0, 260),
    TaskSpec("task_ii", "Task II / High-res object detection", "YOLOv8-L-1080p", 220, 35, 320),
    TaskSpec("task_iii", "Task III / UAV scene recognition", "MobileViT-S", 240, 70, 340),
    TaskSpec("task_iv", "Task IV / Large scene recognition", "ViT-L-1024", 300, 105, 420),
    TaskSpec("task_v", "Task V / Edge command generation", "GPT-2-small KV", 360, 130, 500),
    TaskSpec("task_vi", "Task VI / Wildfire detection", "ResNet152-512", 260, 160, 380),
)

TASK_SHORT = {
    "task_i": "T1",
    "task_ii": "T2",
    "task_iii": "T3",
    "task_iv": "T4",
    "task_v": "T5",
    "task_vi": "T6",
    "be": "BE",
    "reserve": "IF",
}

VARIANTS = (
    VariantChoice("task_i", "Pantheon", "E3", 0.0, 3, 140, 900, 0.88, 900, True, "Task I is first only because A_I=0 and no other RT job is active yet."),
    VariantChoice("task_i", "RTInfer", "P25-E4", 0.25, 4, 215, 1180, 0.93, 180, True, "ALC keeps a deeper exit under memory pressure."),
    VariantChoice("task_i", "RTInfer-periodic", "P50-E2", 0.50, 2, 135, 520, 0.87, 90, True, "Second-period compact variant."),
    VariantChoice("task_ii", "Pantheon", "E2", 0.0, 2, 120, 1150, 0.68, 1150, True, "YOLOv8-L is forced from E4 to E2 after queueing."),
    VariantChoice("task_ii", "RTInfer", "P25-E4", 0.25, 4, 220, 920, 0.90, 220, True, "YOLOv8-L keeps the deep exit through pruning and layout."),
    VariantChoice("task_ii", "RTInfer-periodic", "P50-E2", 0.50, 2, 145, 700, 0.82, 120, True, "Second-period compact YOLO variant."),
    VariantChoice("task_iii", "Pantheon", "E1", 0.0, 1, 85, 760, 0.62, 760, True, "MobileViT takes a shallow exit due to remaining slack."),
    VariantChoice("task_iii", "RTInfer", "P25-E3", 0.25, 3, 195, 610, 0.89, 140, True, "MobileViT keeps a deeper calibrated exit."),
    VariantChoice("task_iii", "RTInfer-periodic", "P25-E2", 0.25, 2, 140, 560, 0.84, 120, True, "Second-period compact MobileViT variant."),
    VariantChoice("task_iv", "Pantheon", "E1", 0.0, 1, 120, 1150, 0.58, 1150, True, "ViT-L falls back to E1."),
    VariantChoice("task_iv", "RTInfer", "P50-E3", 0.50, 3, 235, 630, 0.88, 180, True, "ViT-L uses pruned transformer blocks."),
    VariantChoice("task_v", "Pantheon", "E1-worst-KV", "0.0", 1, 145, 1600, 0.70, 1600, True, "SOTA reserves worst-case KV cache as one rectangle."),
    VariantChoice("task_v", "RTInfer", "P25-E3-stepped-KV", 0.25, 3, 300, 650, 0.84, 160, True, "RTInfer represents KV as stepped buffers."),
    VariantChoice("task_vi", "Pantheon", "E1", 0.0, 1, 80, 900, 0.55, 900, True, "Wildfire detection misses after a long queue."),
    VariantChoice("task_vi", "RTInfer", "P50-E2", 0.50, 2, 170, 340, 0.82, 100, True, "Compact stress-task variant fits the packed burst."),
)


PANTHEON_EVENTS = (
    BufferEvent("Pantheon", "be#0", "be", "BE0", 0, 35, 3440, 520, "be", "BE0", "AlexNet-like", "-", "-", "-", "BE runs with Task I until the RT burst starts"),
    BufferEvent("Pantheon", "full_load_i#0", "task_i", "Full load I", 0, 18, 0, 0, "full_load", "load", "E3", "-", "-", "-", "first use must load the selected Task I variant", "full", 900),
    BufferEvent("Pantheon", "task_i#0", "task_i", "T1-C1", 18, 55, 0, 900, "rt", "C1", "E3", 3, 0.0, 0.88, "queue head because A_I=0 and it is the only RT job"),
    BufferEvent("Pantheon", "full_load_ii#0", "task_ii", "Full load II", 35, 55, 0, 0, "full_load", "load", "E2", "-", "-", "-", "full H2D load before YOLO first use", "full", 1150),
    BufferEvent("Pantheon", "task_i#0", "task_i", "T1-IF", 55, 155, 0, 420, "if_hold", "IF", "E3", 3, 0.0, 0.88, "T1-IF remains in T1's original address range; T2 is allocated above the held IF."),
    BufferEvent("Pantheon", "task_ii#0", "task_ii", "T2-C1", 55, 105, 420, 1150, "rt", "C1", "E2", 2, 0.0, 0.68, "YOLO preempts the older T1 job and is allocated above T1-IF"),
    BufferEvent("Pantheon", "task_ii#0", "task_ii", "T2-C2", 105, 155, 420, 1150, "rt", "C2", "E2", 2, 0.0, 0.68, "serial RT chunk above held T1-IF"),
    BufferEvent("Pantheon", "task_i#0", "task_i", "T1-C2", 155, 205, 0, 900, "rt", "C2", "E3", 3, 0.0, 0.88, "T1 resumes from held IF instead of restarting"),
    BufferEvent("Pantheon", "be#1", "be", "BE1", 205, 250, 3440, 520, "be", "BE1", "AlexNet-like", "-", "-", "-", "BE resumes in a gap while Task I finishes"),
    BufferEvent("Pantheon", "task_i#0", "task_i", "T1-C3", 205, 250, 0, 900, "rt", "C3", "E3", 3, 0.0, 0.88, "T1 completes after preemption"),
    BufferEvent("Pantheon", "be#2", "be", "BE2", 260, 345, 3440, 520, "be", "BE2", "AlexNet-like", "-", "-", "-", "BE runs with a shallow MobileViT RT chunk"),
    BufferEvent("Pantheon", "task_iii#0", "task_iii", "T3-C1", 260, 345, 0, 760, "rt", "C1", "E1", 1, 0.0, 0.62, "MobileViT shallow exit"),
    BufferEvent("Pantheon", "task_iv#0", "task_iv", "T4-C1", 345, 465, 0, 1150, "rt", "C1", "E1", 1, 0.0, 0.58, "ViT-L shallow exit"),
    BufferEvent("Pantheon", "full_load_v#0", "task_v", "Full load V", 430, 465, 0, 0, "full_load", "load", "E1-worst-KV", "-", "-", "-", "full reload before GPT/KV execution", "full", 1600),
    BufferEvent("Pantheon", "task_v#0", "task_v", "T5-C1", 465, 610, 0, 1600, "kv", "C1", "E1-worst-KV", 1, "0.0", 0.70, "KV=max: worst-case KV rectangle squeezes later RT jobs"),
    BufferEvent("Pantheon", "task_vi#0", "task_vi", "T6-C1", 610, 690, 0, 900, "rt", "C1", "E1", 1, 0.0, 0.55, "MISS: wildfire detection starts too late"),
    BufferEvent("Pantheon", "full_reload_i#1", "task_i", "Full reload I'", 650, 690, 0, 0, "full_load", "load", "E1", "-", "-", "-", "SOTA reloads the second-period Task I variant in full", "full", 520),
    BufferEvent("Pantheon", "be#3", "be", "BE3", 690, 760, 3440, 520, "be", "BE3", "AlexNet-like", "-", "-", "-", "BE resumes after the RT queue drains"),
    BufferEvent("Pantheon", "task_i#1", "task_i", "T1'-C1", 690, 760, 0, 520, "rt", "C1", "E1", 1, 0.50, 0.60, "MISS: second-period Task I starts too late"),
)


RTINFER_EVENTS = (
    BufferEvent("RTInfer", "be#0", "be", "BE", 0, 70, 3922, 160, "be", "BE", "AlexNet-like", "-", "-", "-", "BE runs before the RT burst becomes dense"),
    BufferEvent("RTInfer", "load_i#0", "task_i", "Full load I", 0, 20, 0, 0, "full_load", "load", "P25-E4", "-", "-", "-", "first use full load before T1-C1", "full", 1180),
    BufferEvent("RTInfer", "task_i#0", "task_i", "T1-C1", 20, 95, 0, 560, "rt", "C1", "P25-E4", 4, 0.25, 0.93, "traffic-light detection stream"),
    BufferEvent("RTInfer", "load_ii#0", "task_ii", "Full load II", 35, 55, 0, 0, "full_load", "load", "P25-E4", "-", "-", "-", "first use full load before T2-C1", "full", 920),
    BufferEvent("RTInfer", "task_ii#0", "task_ii", "T2-C1", 55, 145, 560, 760, "rt", "C1", "P25-E4", 4, 0.25, 0.90, "YOLO high-resolution stream"),
    BufferEvent("RTInfer", "load_iii#0", "task_iii", "Full load III", 70, 85, 0, 0, "full_load", "load", "P25-E3", "-", "-", "-", "first use full load before T3-C1", "full", 610),
    BufferEvent("RTInfer", "task_iii#0", "task_iii", "T3-C1", 85, 175, 1480, 610, "rt", "C1", "P25-E3", 3, 0.25, 0.89, "MobileViT stream"),
    BufferEvent("RTInfer", "load_iv#0", "task_iv", "Full load IV", 105, 125, 0, 0, "full_load", "load", "P50-E3", "-", "-", "-", "first use full load before T4-C1", "full", 630),
    BufferEvent("RTInfer", "task_iv#0", "task_iv", "T4-C1", 125, 230, 2090, 580, "rt", "C1", "P50-E3", 3, 0.50, 0.88, "ViT-L first packed segment"),
    BufferEvent("RTInfer", "load_v#0", "task_v", "Full load V", 130, 150, 0, 0, "full_load", "load", "P25-E3-stepped-KV", "-", "-", "-", "first use full load before T5-C1", "full", 650),
    BufferEvent("RTInfer", "task_i#0", "task_i", "T1-C2", 95, 155, 0, 560, "rt", "C2", "P25-E4", 4, 0.25, 0.93, "traffic-light detection chunk"),
    BufferEvent("RTInfer", "task_ii#0", "task_ii", "T2-C2", 145, 255, 560, 920, "rt", "C2", "P25-E4", 4, 0.25, 0.90, "YOLO deep exit continues"),
    BufferEvent("RTInfer", "task_i#0", "task_i", "T1-C3", 155, 185, 0, 560, "rt", "C3", "P25-E4", 4, 0.25, 0.93, "keeps deeper exit"),
    BufferEvent("RTInfer", "task_i#0", "task_i", "T1-C4", 185, 215, 0, 560, "rt", "C4", "P25-E4", 4, 0.25, 0.93, "exit E4 completes"),
    BufferEvent("RTInfer", "task_v#0", "task_v", "T5-C1", 150, 230, 2670, 300, "kv", "C1", "P25-E3-stepped-KV", 3, 0.25, 0.84, "KV step: early decoding"),
    BufferEvent("RTInfer", "be#1", "be", "BE", 150, 210, 3922, 160, "be", "BE", "AlexNet-like", "-", "-", "-", "BE placed in the remaining top address gap"),
    BufferEvent("RTInfer", "load_vi#0", "task_vi", "Full load VI", 160, 170, 0, 0, "full_load", "load", "P50-E2", "-", "-", "-", "first use full load before T6-C1", "full", 340),
    BufferEvent("RTInfer", "task_vi#0", "task_vi", "T6-C1", 170, 230, 3140, 340, "rt", "C1", "P50-E2", 2, 0.50, 0.82, "wildfire detection fits in address gap"),
    BufferEvent("RTInfer", "task_iii#0", "task_iii", "T3-C2", 175, 280, 1480, 610, "rt", "C2", "P25-E3", 3, 0.25, 0.89, "MobileViT deeper exit"),
    BufferEvent("RTInfer", "load_i#1", "task_i", "T1' Delta 90MiB", 180, 195, 0, 0, "delta_load", "load", "P50-E2", "-", "-", "-", "Delta-Graph loads only missing chunks for second-period T1'", "delta", 90),
    BufferEvent("RTInfer", "task_i#1", "task_i", "T1'-C1", 225, 360, 0, 520, "rt", "C1", "P50-E2", 2, 0.50, 0.87, "second-period compact traffic-light task"),
    BufferEvent("RTInfer", "task_iv#0", "task_iv", "T4-C2", 230, 340, 2090, 580, "rt", "C2", "P50-E3", 3, 0.50, 0.88, "ViT-L deeper exit"),
    BufferEvent("RTInfer", "task_v#0", "task_v", "T5-C2", 230, 330, 2670, 460, "kv", "C2", "P25-E3-stepped-KV", 3, 0.25, 0.84, "KV step: cache grows as decoding proceeds"),
    BufferEvent("RTInfer", "task_vi#0", "task_vi", "T6-C2", 260, 330, 3140, 300, "rt", "C2", "P50-E2", 2, 0.50, 0.82, "second compact wildfire chunk"),
    BufferEvent("RTInfer", "load_ii#1", "task_ii", "T2' Delta 120MiB", 255, 285, 0, 0, "delta_load", "load", "P50-E2", "-", "-", "-", "Delta-Graph loads only missing chunks for second-period YOLO", "delta", 120),
    BufferEvent("RTInfer", "task_ii#1", "task_ii", "T2'-C1", 285, 430, 560, 700, "rt", "C1", "P50-E2", 2, 0.50, 0.82, "second-period compact YOLO"),
    BufferEvent("RTInfer", "load_iii#1", "task_iii", "T3' Delta 120MiB", 310, 330, 0, 0, "delta_load", "load", "P25-E2", "-", "-", "-", "Delta-Graph loads only missing chunks for second-period MobileViT", "delta", 120),
    BufferEvent("RTInfer", "task_v#0", "task_v", "T5-C3", 330, 430, 2670, 650, "kv", "C3", "P25-E3-stepped-KV", 3, 0.25, 0.84, "KV step: final cache footprint"),
    BufferEvent("RTInfer", "task_iii#1", "task_iii", "T3'-C1", 360, 500, 1480, 560, "rt", "C1", "P25-E2", 2, 0.25, 0.84, "second-period MobileViT"),
    BufferEvent("RTInfer", "be#2", "be", "BE", 500, 650, 512, 420, "be", "BE", "AlexNet-like", "-", "-", "-", "BE resumes when RT pressure drops"),
)


JOB_RESULTS = (
    ("Pantheon", "task_i#0", 260, 250, 0.88),
    ("Pantheon", "task_ii#0", 355, 155, 0.68),
    ("Pantheon", "task_iii#0", 410, 345, 0.62),
    ("Pantheon", "task_iv#0", 525, 465, 0.58),
    ("Pantheon", "task_v#0", 630, 610, 0.70),
    ("Pantheon", "task_vi#0", 540, 690, 0.55),
    ("Pantheon", "task_i#1", 440, 760, 0.60),
    ("RTInfer", "task_i#0", 260, 215, 0.93),
    ("RTInfer", "task_ii#0", 355, 255, 0.90),
    ("RTInfer", "task_iii#0", 410, 280, 0.89),
    ("RTInfer", "task_iv#0", 525, 340, 0.88),
    ("RTInfer", "task_v#0", 630, 430, 0.84),
    ("RTInfer", "task_vi#0", 540, 330, 0.82),
    ("RTInfer", "task_i#1", 440, 360, 0.87),
    ("RTInfer", "task_ii#1", 575, 430, 0.82),
    ("RTInfer", "task_iii#1", 650, 500, 0.84),
)


COLORS = {
    "task_i": "#e4572e",
    "task_ii": "#b2182b",
    "task_iii": "#f4a261",
    "task_iv": "#d95f02",
    "task_v": "#8c510a",
    "task_vi": "#f6c85f",
    "be": "#9dbf75",
    "if_hold": "#eeeeee",
    "load": "#c9c2bc",
}

TIME_MARKERS = (
    ("A_I", 0, "arrival"),
    ("A_II", 35, "arrival"),
    ("A_III", 70, "arrival"),
    ("A_IV", 105, "arrival"),
    ("A_V", 130, "arrival"),
    ("A_VI", 160, "arrival"),
    ("A_I'", 180, "arrival"),
    ("A_II'", 255, "arrival"),
    ("A_III'", 310, "arrival"),
    ("d_I", 260, "deadline"),
    ("d_II", 355, "deadline"),
    ("d_III", 410, "deadline"),
    ("d_I'", 440, "deadline"),
    ("d_IV", 525, "deadline"),
    ("d_VI", 540, "deadline"),
    ("d_II'", 575, "deadline"),
    ("d_V", 630, "deadline"),
    ("d_III'", 650, "deadline"),
)


def time_overlap(a: BufferEvent, b: BufferEvent) -> bool:
    return a.start_ms < b.end_ms and b.start_ms < a.end_ms


def address_overlap(a: BufferEvent, b: BufferEvent) -> bool:
    return a.addr_mib < b.top_mib and b.addr_mib < a.top_mib


def validate_events(events: Sequence[BufferEvent]) -> list[str]:
    errors: list[str] = []
    nonzero = [event for event in events if event.memory_mib > 0 and event.kind not in {"full_load", "delta_load"}]
    for event in nonzero:
        if event.top_mib > MEM_BUDGET_MIB:
            errors.append(f"{event.system}:{event.job}:{event.label} exceeds memory budget")
        if event.start_ms >= event.end_ms:
            errors.append(f"{event.system}:{event.job}:{event.label} has invalid interval")
    for index, left in enumerate(nonzero):
        for right in nonzero[index + 1 :]:
            if left.system == right.system and time_overlap(left, right) and address_overlap(left, right):
                errors.append(f"{left.system}: overlap {left.job}/{left.label} with {right.job}/{right.label}")
    first_use = {
        "full_load_i#0": 18,
        "full_load_ii#0": 55,
        "full_load_v#0": 465,
        "full_reload_i#1": 690,
        "load_i#0": 20,
        "load_ii#0": 55,
        "load_iii#0": 85,
        "load_iv#0": 125,
        "load_v#0": 150,
        "load_vi#0": 170,
        "load_i#1": 225,
        "load_ii#1": 285,
        "load_iii#1": 360,
    }
    for event in events:
        if event.kind in {"full_load", "delta_load"} and event.job in first_use and event.end_ms > first_use[event.job]:
            errors.append(f"{event.job} ends after first use")
    return errors


def active_at(events: Sequence[BufferEvent], at_ms: int, kinds: set[str]) -> list[BufferEvent]:
    return [event for event in events if event.kind in kinds and event.start_ms <= at_ms < event.end_ms]


def validate_semantics() -> list[str]:
    errors: list[str] = []
    all_events = PANTHEON_EVENTS + RTINFER_EVENTS
    for system, events in (("Pantheon", PANTHEON_EVENTS), ("RTInfer", RTINFER_EVENTS)):
        if any(event.kind == "reserve" for event in events):
            errors.append(f"{system}: global Reserve IF should not be modeled as a persistent buffer")

        expected_misses = missed_jobs(system)
        actual_misses = {
            event.job
            for event in events
            if event.kind in {"rt", "kv"} and event.detail.upper().startswith("MISS")
        }
        if actual_misses != expected_misses:
            errors.append(f"{system}: miss labels {sorted(actual_misses)} != JOB_RESULTS {sorted(expected_misses)}")

    sota_rt_be = any(
        rt.system == "Pantheon"
        and rt.kind in {"rt", "kv"}
        and be.kind == "be"
        and time_overlap(rt, be)
        for rt in PANTHEON_EVENTS
        for be in PANTHEON_EVENTS
    )
    if not sota_rt_be:
        errors.append("Pantheon: expected at least one RT+BE concurrency interval")

    sota_be = sorted([event for event in PANTHEON_EVENTS if event.kind == "be"], key=lambda event: event.start_ms)
    if len(sota_be) < 3 or not any(left.end_ms < right.start_ms for left, right in zip(sota_be, sota_be[1:])):
        errors.append("Pantheon: BE should be segmented to show preempt/pause/resume")

    if_holds = [event for event in PANTHEON_EVENTS if event.kind == "if_hold"]
    if not if_holds:
        errors.append("Pantheon: expected at least one IF hold interval for a preempted RT job")
    for hold in if_holds:
        before_compute = [
            event
            for event in PANTHEON_EVENTS
            if event.job == hold.job and event.kind in {"rt", "kv"} and event.end_ms <= hold.start_ms
        ]
        after_compute = [
            event
            for event in PANTHEON_EVENTS
            if event.job == hold.job and event.kind in {"rt", "kv"} and event.start_ms >= hold.end_ms
        ]
        has_before = any(
            True for _ in before_compute
        )
        has_after = any(
            True for _ in after_compute
        )
        if not (has_before and has_after):
            errors.append(f"Pantheon: IF hold {hold.label} must have compute before and after it")
        if before_compute:
            base = before_compute[-1].addr_mib
            full_memory = before_compute[-1].memory_mib
            if hold.addr_mib != base:
                errors.append(f"Pantheon: IF hold {hold.label} should keep the original base address {base}")
            if hold.memory_mib > full_memory:
                errors.append(f"Pantheon: IF hold {hold.label} should not exceed original compute memory")
        for active in active_at(PANTHEON_EVENTS, hold.start_ms, {"rt", "kv"}):
            if active.job != hold.job and active.addr_mib < hold.top_mib:
                errors.append(f"Pantheon: preemptor {active.label} should be allocated above held IF {hold.label}")

    for marker in range(0, HORIZON_MS + 1, 5):
        if len(active_at(PANTHEON_EVENTS, marker, {"rt", "kv"})) > 1:
            errors.append("Pantheon: RT-RT events should remain queue-like, not globally packed")
            break

    packed = active_at(RTINFER_EVENTS, 180, {"rt", "kv"})
    be_live = active_at(RTINFER_EVENTS, 180, {"be"})
    if len(packed) < 6 or not be_live:
        errors.append("RTInfer: expected at least 6 RT/KV objects plus BE live around 170-210 ms")

    if not any(event.system == "Pantheon" and event.task == "task_v" and "KV=max" in event.detail for event in all_events):
        errors.append("Pantheon: GPT-2 should be shown as a worst-case KV rectangle")
    rtinfer_kv_steps = [event for event in RTINFER_EVENTS if event.task == "task_v" and event.kind == "kv"]
    if len(rtinfer_kv_steps) < 3 or not all(event.label.startswith("T5-C") for event in rtinfer_kv_steps):
        errors.append("RTInfer: GPT-2 KV cache should be shown as T5-C* stepped buffers")

    for task in ("task_i", "task_ii", "task_iii"):
        full_jobs = [event for event in RTINFER_EVENTS if event.task == task and event.job.endswith("#0") and event.kind in {"rt", "kv"}]
        compact_jobs = [event for event in RTINFER_EVENTS if event.task == task and event.job.endswith("#1") and event.kind in {"rt", "kv"}]
        full_runtime = sum(event.end_ms - event.start_ms for event in full_jobs)
        compact_runtime = sum(event.end_ms - event.start_ms for event in compact_jobs)
        if compact_runtime >= full_runtime:
            errors.append(f"RTInfer: compact second-period {task} should be visually shorter than first-period variant")

    if not any(event.system == "RTInfer" and event.kind == "full_load" and event.job.endswith("#0") for event in all_events):
        errors.append("RTInfer: first-use model loading should be shown as Full load")
    if not any(event.system == "RTInfer" and event.kind == "delta_load" and event.job.endswith("#1") for event in all_events):
        errors.append("RTInfer: second-period variant switching should be shown as Delta load")
    sota_reload = next((event for event in PANTHEON_EVENTS if event.job == "full_reload_i#1"), None)
    rtinfer_delta = next((event for event in RTINFER_EVENTS if event.job == "load_i#1"), None)
    if not sota_reload or not rtinfer_delta or (rtinfer_delta.end_ms - rtinfer_delta.start_ms) >= (sota_reload.end_ms - sota_reload.start_ms):
        errors.append("RTInfer: Delta load strip should be shorter than SOTA full reload")

    old_blue_colors = {"#d9e8ff", "#b7c9f2", "#c9daf8", "#9dc3e6"}
    if any(COLORS[key] in old_blue_colors for key in ("task_i", "task_ii", "task_iii", "task_iv", "task_v", "task_vi", "load")):
        errors.append("Color palette: RT/load colors should avoid the old blue palette")

    pantheon_summary = summarize("Pantheon")
    rtinfer_summary = summarize("RTInfer")
    if rtinfer_summary["dmr"] > pantheon_summary["dmr"]:
        errors.append("Metrics: RTInfer DMR should not exceed SOTA")
    if rtinfer_summary["deadline_weighted_accuracy"] <= pantheon_summary["deadline_weighted_accuracy"]:
        errors.append("Metrics: RTInfer deadline-weighted accuracy should exceed SOTA")
    if rtinfer_summary["completed_only_accuracy"] <= pantheon_summary["completed_only_accuracy"]:
        errors.append("Metrics: RTInfer completed-only accuracy should exceed SOTA")

    return errors


def summarize(system: str) -> dict[str, float]:
    rows = [row for row in JOB_RESULTS if row[0] == system]
    misses = sum(1 for _, _, deadline, finish, _ in rows if finish > deadline)
    weighted = sum(0.0 if finish > deadline else acc for _, _, deadline, finish, acc in rows)
    completed = [acc for _, _, deadline, finish, acc in rows if finish <= deadline]
    return {
        "jobs": float(len(rows)),
        "misses": float(misses),
        "dmr": misses / len(rows),
        "deadline_weighted_accuracy": weighted / len(rows),
        "completed_only_accuracy": sum(completed) / len(completed) if completed else 0.0,
        "makespan_ms": float(max(finish for _, _, _, finish, _ in rows)),
    }


def missed_jobs(system: str) -> set[str]:
    return {job for row_system, job, deadline, finish, _ in JOB_RESULTS if row_system == system and finish > deadline}


def event_missed(event: BufferEvent) -> bool:
    return event.job in missed_jobs(event.system)


def variant_tag(event: BufferEvent) -> str:
    if event.kind not in {"rt", "kv"}:
        return ""
    pruning = "p=0" if str(event.pruning) in {"0.0", "0", "-"} else f"p={float(event.pruning):.2f}"
    if isinstance(event.exit_point, int):
        return f"{pruning},E{event.exit_point}"
    if str(event.exit_point).isdigit():
        return f"{pruning},E{event.exit_point}"
    return pruning


def task_corner_label(event: BufferEvent) -> str:
    if event.kind in {"reserve", "be", "load"}:
        return event.label
    return event.label.split(":")[0]


def load_label(event: BufferEvent) -> str:
    text = event.label.replace("Full load ", "Full ").replace("Delta ", "Δ")
    return text.replace("MiB", "")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tables() -> None:
    write_csv(
        OUT / "modern_variant_table.csv",
        [
            {
                "task": choice.task,
                "system": choice.system,
                "variant": choice.variant,
                "pruning": choice.pruning,
                "exit_point": choice.exit_point,
                "latency_ms": choice.latency_ms,
                "memory_mib": choice.memory_mib,
                "accuracy": choice.accuracy,
                "missing_mib_delta_graph": choice.missing_mib,
                "selected": int(choice.selected),
                "note": choice.note,
            }
            for choice in VARIANTS
        ],
    )
    for filename, events in (
        ("modern_pantheon_trace.csv", PANTHEON_EVENTS),
        ("modern_rtinfer_trace.csv", RTINFER_EVENTS),
    ):
        write_csv(
            OUT / filename,
            [
                {
                    "system": event.system,
                    "job": event.job,
                    "task": event.task,
                    "label": event.label,
                    "start_ms": event.start_ms,
                    "end_ms": event.end_ms,
                    "addr_mib": event.addr_mib,
                    "memory_mib": event.memory_mib,
                    "kind": event.kind,
                    "stage": event.stage,
                    "variant": event.variant,
                    "exit_point": event.exit_point,
                    "pruning": event.pruning,
                    "accuracy": event.accuracy,
                    "load_type": event.load_type,
                    "missing_mib_delta_graph": event.missing_mib_delta_graph,
                    "detail": event.detail,
                }
                for event in events
            ],
        )


def write_decisions() -> None:
    text = """# Modern Mixed Workload Online Decisions

## SOTA / Pantheon

Fairness assumption: Pantheon can run RT tasks concurrently with the BE task. The limitation in this case is not RT-vs-BE overlap; it is the lack of RTInfer-style cross-RT variant selection, Delta-Graph switching, and global time-address packing.

1. `t=0-18 ms`: Task I performs a first-use full H2D load. There is no Delta reuse yet.
2. `t=18 ms`: Task I (Traffic light detection / MobileNetv2-SSDLite-300) runs first because `A_I=0 ms` and it is the only RT job at that instant. This is normal arrival/urgency behavior, not a hand-picked advantage.
3. `t=18-35 ms`: BE analytics is live in the high address range while Task I runs, so SOTA does support RT+BE concurrency.
4. `t=55-155 ms`: YOLOv8-L preempts Task I. Pantheon keeps `T1-IF` at Task I's original base address, and allocates T2 above the held IF so Task I can resume after T2 instead of restarting.
5. `t=155-250 ms`: Task I resumes from the held IF. Pantheon still keeps RT compute mostly queue-ordered rather than globally packing multiple RT chunks.
6. `t=260 ms`: MobileViT-S starts late and takes shallow `E1`.
7. `t=345 ms`: ViT-L starts late and takes shallow `E1`.
8. `t=465 ms`: GPT-2-small KV is represented by one `T5-C1` worst-case `KV=max` rectangle, consuming a large address interval.
9. `t=650-690 ms`: SOTA reloads the second-period Task I variant in full. Task VI and second-period Task I miss.

## RTInfer

1. Offline ALC has already built a modern mixed Variant Atlas: MobileNetv2-SSDLite-300, YOLOv8-L-1080p, MobileViT-S, ViT-L-1024, GPT-2-small KV, and ResNet152-512 variants.
2. Every arrival triggers active-set replanning. The scheduler jointly considers deadline slack, memory budget, accuracy, and missing Delta chunks.
3. First-use model arrivals still use `Full load` strips. Delta-Graph advantage appears on second-period variant switching, where `T1'`, `T2'`, and `T3'` load only missing chunks.
4. `t=170-210 ms`: MobileNetv2-SSDLite-300, YOLOv8-L-1080p, MobileViT-S, ViT-L-1024, GPT-2-small KV step, ResNet152 wildfire detection, and BE are simultaneously live but occupy disjoint address ranges below 4096 MiB.
5. GPT-2 uses `T5-C1/C2/C3` stepped KV-cache buffers rather than a single worst-case `KV=max` rectangle.
6. Periodic second arrivals are admitted with visibly shorter compact variants instead of collapsing the queue.
"""
    (OUT / "modern_online_decisions.md").write_text(text)


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "#222", extra: str = "") -> str:
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="1" {extra}/>'


def label(x: float, y: float, value: str, size: int = 11, anchor: str = "middle", weight: str = "normal") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" font-family="Arial" text-anchor="{anchor}" font-weight="{weight}">{value}</text>'


def event_fill(event: BufferEvent) -> str:
    if event.kind == "if_hold":
        return COLORS["if_hold"]
    if event.kind in {"full_load", "delta_load"}:
        return COLORS["load"]
    return COLORS.get(event.task, "#ddd")


def panel(events: Sequence[BufferEvent], x0: int, y0: int, title: str) -> list[str]:
    panel_w = 880
    panel_h = 245
    sx = panel_w / HORIZON_MS
    sy = panel_h / MEM_BUDGET_MIB
    out: list[str] = []
    out.append(f'<line x1="{x0}" y1="{y0 + panel_h}" x2="{x0 + panel_w}" y2="{y0 + panel_h}" stroke="#222"/>')
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + panel_h}" stroke="#222"/>')
    out.append(label(x0 - 42, y0 + panel_h / 2, "Addr", 13, weight="bold"))
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + panel_w}" y2="{y0}" stroke="#d7191c" stroke-width="2" stroke-dasharray="6 4"/>')
    out.append(label(x0 + panel_w / 2, y0 - 6, "Memory Budget 4096 MiB", 11, weight="bold"))
    h2d_y = y0 - 28
    out.append(f'<line x1="{x0}" y1="{h2d_y:.2f}" x2="{x0 + panel_w}" y2="{h2d_y:.2f}" stroke="#bdbdbd" stroke-width="1"/>')
    out.append(label(x0 + 8, h2d_y - 19, "H2D loads", 10, anchor="start", weight="bold"))
    for name, at_ms, kind in TIME_MARKERS:
        x = x0 + at_ms * sx
        color = "#555555" if kind == "arrival" else "#d7191c"
        out.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0 + panel_h}" stroke="{color}" stroke-width="0.8" stroke-dasharray="3 5" opacity="0.45"/>')
        if name in {"A_I", "A_II", "A_III", "A_IV", "A_V", "A_VI", "A_I'", "d_I", "d_II", "d_IV", "d_V"}:
            out.append(label(x, y0 + panel_h + 17, name, 9, weight="bold"))
    out.append(label(x0 + panel_w / 2, y0 + panel_h + 43, title, 16, weight="bold"))
    for event in events:
        x = x0 + event.start_ms * sx
        w = max(2.0, (event.end_ms - event.start_ms) * sx)
        if event.kind in {"full_load", "delta_load"}:
            h = 5.0
            y = h2d_y - h / 2
        else:
            h = max(2.0, event.memory_mib * sy)
            y = y0 + panel_h - event.top_mib * sy
        extra = 'style="fill:url(#hatch)"' if event.kind == "if_hold" else ""
        stroke = "#6f625c" if event.kind in {"full_load", "delta_load"} else ("#d7191c" if event_missed(event) and event.kind in {"rt", "kv"} else "#222")
        stroke_width = "2" if event_missed(event) and event.kind in {"rt", "kv"} else "1"
        if event.kind in {"full_load", "delta_load"}:
            out.append(rect(x, y, w, h, event_fill(event), stroke=stroke))
            if w >= 30:
                out.append(label(x + w / 2, y - 2, load_label(event), 7, weight="bold"))
            continue
        out.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{event_fill(event)}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" {extra}/>'
        )
        if event_missed(event) and event.kind in {"rt", "kv"}:
            out.append(rect(x, y, w, min(7, h), "#f4cccc", stroke="#d7191c"))
        if w > 18 and h > 11:
            center_value = event.stage if event.kind in {"rt", "kv", "if_hold"} else event.label
            compact = w < 52 or h < 22
            if not (event.kind == "kv" and h < 24):
                out.append(label(x + w / 2, y + h / 2 + 4, center_value, 9, weight="bold"))
            if not compact:
                out.append(label(x + 3, y + 10, task_corner_label(event), 7, anchor="start", weight="bold"))
            elif event.kind == "kv":
                out.append(label(x + 3, y + min(11, h - 3), task_corner_label(event), 7, anchor="start", weight="bold"))
            tag = variant_tag(event)
            if tag and w >= 72 and h >= 20:
                out.append(label(x + w - 3, y + 10, tag, 7, anchor="end", weight="bold"))
            if event.kind == "kv" and h >= 28 and w >= 58:
                kv_tag = "KV=max" if "KV=max" in event.detail else "KV step"
                out.append(label(x + 3, y + h - 4, kv_tag, 7, anchor="start", weight="bold"))
        if event_missed(event) and event.kind in {"rt", "kv"}:
            out.append(label(x + w + 6, y + 10, "MISS", 9, anchor="start", weight="bold"))
    return out


def write_svg() -> None:
    width = 1230
    height = 750
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<pattern id="hatch" patternUnits="userSpaceOnUse" width="8" height="8">',
        '<rect width="8" height="8" fill="#ffffff"/>',
        '<path d="M0 8 L8 0" stroke="#222" stroke-width="1"/>',
        "</pattern>",
        "</defs>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        label(width / 2, 28, "Rebuttal-Aligned Modern Mixed Workload on Simulated Jetson Nano", 19, weight="bold"),
    ]
    out.extend(panel(PANTHEON_EVENTS, 70, 85, "(a) Pantheon / SOTA: RT+BE concurrency, but RT queueing causes shallow exits"))
    out.extend(panel(RTINFER_EVENTS, 70, 435, "(b) RTInfer / Ours: ALC variants + time-address packing + Delta-Graph loading"))
    legend_x = 990
    legend_y = 92
    out.append(label(legend_x, legend_y - 20, "Legend", 14, anchor="start", weight="bold"))
    legend = [
        ("T1 MobileNetv2-SSDLite-300", COLORS["task_i"], ""),
        ("T2 YOLOv8-L-1080p", COLORS["task_ii"], ""),
        ("T3 MobileViT-S", COLORS["task_iii"], ""),
        ("T4 ViT-L-1024", COLORS["task_iv"], ""),
        ("T5 GPT-2-small KV", COLORS["task_v"], ""),
        ("T6 ResNet152-512 wildfire", COLORS["task_vi"], ""),
        ("BE task", COLORS["be"], ""),
        ("Full/Delta H2D strip", COLORS["load"], ""),
        ("IF hold after preemption", COLORS["if_hold"], 'style="fill:url(#hatch)"'),
    ]
    for idx, (name, fill, extra) in enumerate(legend):
        y = legend_y + idx * 27
        out.append(rect(legend_x, y, 24, 16, fill, extra=extra))
        out.append(label(legend_x + 34, y + 12, name, 11, anchor="start"))
    pantheon = summarize("Pantheon")
    rtinfer = summarize("RTInfer")
    result_y = 370
    out.append(label(legend_x, result_y, "Result", 14, anchor="start", weight="bold"))
    out.append(label(legend_x, result_y + 25, f"SOTA DMR {pantheon['misses']:.0f}/{pantheon['jobs']:.0f}", 11, anchor="start"))
    out.append(label(legend_x, result_y + 47, f"SOTA acc {pantheon['deadline_weighted_accuracy']:.2f}", 11, anchor="start"))
    out.append(label(legend_x, result_y + 66, "missed: T6, T1'", 10, anchor="start"))
    out.append(label(legend_x, result_y + 94, f"Ours DMR {rtinfer['misses']:.0f}/{rtinfer['jobs']:.0f}", 11, anchor="start"))
    out.append(label(legend_x, result_y + 116, f"Ours acc {rtinfer['deadline_weighted_accuracy']:.2f}", 11, anchor="start"))
    out.append(label(legend_x, result_y + 146, "p=pruning, E=exit", 10, anchor="start", weight="bold"))
    out.append(label(legend_x, result_y + 170, "width=time, height=memory", 10, anchor="start"))
    out.append(label(legend_x, result_y + 192, "170-210 ms: 6 RT + BE live", 10, anchor="start"))
    out.append("</svg>")
    (OUT / "modern_mixed_case.svg").write_text("\n".join(out))


def write_summary() -> None:
    pantheon = summarize("Pantheon")
    rtinfer = summarize("RTInfer")
    lines = [
        "Modern mixed workload case summary",
        f"memory_budget_mib={MEM_BUDGET_MIB}",
        "global_reserve_if=disabled",
        "if_hold_semantics=only preempted jobs retain intermediate features",
        "device=simulated Jetson Nano shared-memory budget",
        "",
        "Pantheon / SOTA:",
        f"  jobs={pantheon['jobs']:.0f}",
        f"  DMR={pantheon['dmr']:.4f} ({pantheon['misses']:.0f}/{pantheon['jobs']:.0f})",
        f"  deadline_weighted_accuracy={pantheon['deadline_weighted_accuracy']:.4f}",
        f"  completed_only_accuracy={pantheon['completed_only_accuracy']:.4f}",
        f"  makespan_ms={pantheon['makespan_ms']:.0f}",
        "",
        "RTInfer / Ours:",
        f"  jobs={rtinfer['jobs']:.0f}",
        f"  DMR={rtinfer['dmr']:.4f} ({rtinfer['misses']:.0f}/{rtinfer['jobs']:.0f})",
        f"  deadline_weighted_accuracy={rtinfer['deadline_weighted_accuracy']:.4f}",
        f"  completed_only_accuracy={rtinfer['completed_only_accuracy']:.4f}",
        f"  makespan_ms={rtinfer['makespan_ms']:.0f}",
        "",
        "Semantic checks:",
        "  SOTA includes RT+BE concurrency and segmented BE preemption/resume.",
        "  SOTA includes local IF hold in the preempted job's original address range, not a global Reserve IF.",
        "  RTInfer includes 6 RT tasks plus BE live during 170-210 ms.",
        "  GPT-2 KV cache is a SOTA worst-case rectangle and an RTInfer stepped footprint.",
        "  First-use loads are full; second-period RTInfer switches use shorter Delta loads.",
        "  H2D strips finish before first use and are drawn outside the address plane.",
    ]
    (OUT / "modern_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\noutputs={OUT}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    errors = validate_events(PANTHEON_EVENTS) + validate_events(RTINFER_EVENTS) + validate_semantics()
    if errors:
        raise SystemExit("layout validation failed:\n" + "\n".join(errors))
    write_tables()
    write_decisions()
    write_svg()
    write_summary()


if __name__ == "__main__":
    main()
