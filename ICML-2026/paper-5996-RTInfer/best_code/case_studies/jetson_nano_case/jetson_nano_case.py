from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"

MEM_BUDGET_MIB = 4096
RESERVE_IF_MIB = 512
HORIZON_MS = 280


@dataclass(frozen=True)
class Variant:
    task: str
    variant: str
    pruning: float
    exit_point: int
    latency_ms: int
    memory_mib: int
    accuracy: float
    missing_mib: int
    note: str


@dataclass(frozen=True)
class Task:
    name: str
    label: str
    arrival_ms: int
    deadline_ms: int
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class Event:
    system: str
    task: str
    label: str
    start_ms: int
    end_ms: int
    addr_mib: int
    memory_mib: int
    variant: str
    exit_point: int | str
    pruning: float | str
    accuracy: float | str
    kind: str
    detail: str


TASKS = (
    Task(
        name="task_i",
        label="Task I / Traffic detection",
        arrival_ms=0,
        deadline_ms=220,
        variants=(
            Variant("task_i", "E1-tiny", 0.50, 1, 70, 520, 0.76, 80, "SOTA fallback if queue is tight"),
            Variant("task_i", "E2-mid", 0.00, 2, 105, 1200, 0.86, 1200, "Pantheon selected"),
            Variant("task_i", "P25-E3", 0.25, 3, 135, 1150, 0.91, 220, "RTInfer selected"),
        ),
    ),
    Task(
        name="task_ii",
        label="Task II / Sign classification",
        arrival_ms=20,
        deadline_ms=210,
        variants=(
            Variant("task_ii", "E1-tiny", 0.00, 1, 52, 700, 0.80, 700, "Pantheon selected due queue slack"),
            Variant("task_ii", "E2-mid", 0.25, 2, 75, 620, 0.89, 110, "Alternative"),
            Variant("task_ii", "P25-E3", 0.25, 3, 95, 760, 0.94, 140, "RTInfer selected"),
        ),
    ),
    Task(
        name="task_iii",
        label="Task III / Scene recognition",
        arrival_ms=40,
        deadline_ms=260,
        variants=(
            Variant("task_iii", "E1-tiny", 0.00, 1, 54, 850, 0.58, 850, "Pantheon selected due queue slack"),
            Variant("task_iii", "P25-E2", 0.25, 2, 120, 900, 0.86, 180, "RTInfer selected"),
            Variant("task_iii", "P25-E3", 0.25, 3, 150, 1250, 0.90, 260, "Too slow for this case"),
        ),
    ),
)


PANTHEON_EVENTS = [
    Event("Pantheon", "reserve", "Reserve IF", 0, HORIZON_MS, 0, RESERVE_IF_MIB, "-", "-", "-", "-", "reserve", "reserved intermediate-feature space"),
    Event("Pantheon", "be", "BE Task", 0, 20, 512, 500, "BE", "-", "-", "-", "be", "low-priority task runs until RT burst"),
    Event("Pantheon", "task_i", "I / E2", 0, 105, 512, 1200, "E2-mid", 2, 0.00, 0.86, "rt", "serial RT execution"),
    Event("Pantheon", "task_ii", "II / E1", 105, 157, 512, 700, "E1-tiny", 1, 0.00, 0.80, "rt", "forced shallow exit after waiting"),
    Event("Pantheon", "task_iii", "III / E1", 157, 211, 512, 850, "E1-tiny", 1, 0.00, 0.58, "rt", "forced shallow exit after waiting"),
    Event("Pantheon", "be", "BE Task", 211, HORIZON_MS, 512, 500, "BE", "-", "-", "-", "be", "best-effort resumes after RT queue drains"),
]


RTINFER_EVENTS = [
    Event("RTInfer", "reserve", "Reserve IF", 0, HORIZON_MS, 0, RESERVE_IF_MIB, "-", "-", "-", "-", "reserve", "reserved intermediate-feature space"),
    Event("RTInfer", "be", "BE Task", 0, 20, 3322, 500, "BE", "-", "-", "-", "be", "low-priority stream, preempted at burst"),
    Event("RTInfer", "task_i", "I / P25-E3", 8, 143, 512, 1150, "P25-E3", 3, 0.25, 0.91, "rt", "high-priority stream 0"),
    Event("RTInfer", "task_ii", "II / P25-E3", 26, 121, 1662, 760, "P25-E3", 3, 0.25, 0.94, "rt", "high-priority stream 1"),
    Event("RTInfer", "task_iii", "III / P25-E2", 48, 168, 2422, 900, "P25-E2", 2, 0.25, 0.86, "rt", "high-priority stream 2"),
    Event("RTInfer", "be", "BE Task", 168, HORIZON_MS, 512, 500, "BE", "-", "-", "-", "be", "best-effort resumes after RT finishes"),
    Event("RTInfer", "load_i", "Delta load I", 0, 8, 3720, 120, "load", "-", "-", "-", "load", "load missing chunks: 220 MiB before first use"),
    Event("RTInfer", "load_ii", "Delta load II", 20, 26, 3720, 120, "load", "-", "-", "-", "load", "load missing chunks: 140 MiB before first use"),
    Event("RTInfer", "load_iii", "Delta load III", 40, 48, 3720, 120, "load", "-", "-", "-", "load", "load missing chunks: 180 MiB before first use"),
]


ARRIVALS_AND_DEADLINES = [
    ("A_I", 0),
    ("A_II", 20),
    ("A_III", 40),
    ("d_II", 210),
    ("d_I", 220),
    ("d_III", 260),
]


COLORS = {
    "task_i": "#f26b2f",
    "task_ii": "#f4b183",
    "task_iii": "#d9e8ff",
    "be": "#b6d7a8",
    "reserve": "#ffffff",
    "load": "#9dc3e6",
}


def selected_accuracy(events: Iterable[Event]) -> tuple[int, float, float]:
    rt_events = [event for event in events if event.kind == "rt"]
    misses = 0
    weighted_acc = 0.0
    raw_acc = 0.0
    for event in rt_events:
        task = next(task for task in TASKS if task.name == event.task)
        acc = float(event.accuracy)
        raw_acc += acc
        if event.end_ms > task.deadline_ms:
            misses += 1
        else:
            weighted_acc += acc
    count = len(rt_events)
    return misses, weighted_acc / count, raw_acc / count


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tables() -> None:
    variant_rows = []
    for task in TASKS:
        for variant in task.variants:
            variant_rows.append(
                {
                    "task": task.label,
                    "arrival_ms": task.arrival_ms,
                    "deadline_ms": task.deadline_ms,
                    "variant": variant.variant,
                    "pruning": variant.pruning,
                    "exit_point": variant.exit_point,
                    "latency_ms": variant.latency_ms,
                    "memory_mib": variant.memory_mib,
                    "accuracy": variant.accuracy,
                    "missing_mib_delta_graph": variant.missing_mib,
                    "note": variant.note,
                }
            )
    write_csv(OUT / "variant_table.csv", variant_rows)

    for name, events in (("pantheon_trace.csv", PANTHEON_EVENTS), ("rtinfer_trace.csv", RTINFER_EVENTS)):
        write_csv(
            OUT / name,
            [
                {
                    "system": event.system,
                    "task": event.task,
                    "label": event.label,
                    "start_ms": event.start_ms,
                    "end_ms": event.end_ms,
                    "addr_mib": event.addr_mib,
                    "memory_mib": event.memory_mib,
                    "variant": event.variant,
                    "exit_point": event.exit_point,
                    "pruning": event.pruning,
                    "accuracy": event.accuracy,
                    "kind": event.kind,
                    "detail": event.detail,
                }
                for event in events
            ],
        )


def write_decisions() -> None:
    text = """# Online Decisions

## SOTA / Pantheon

1. `t=0 ms`: Task I arrives. Pantheon starts the RT queue with Task I and uses exit point `E2` because it has enough immediate slack.
2. `t=20 ms`: Task II arrives while Task I is running. Pantheon does not run it concurrently; Task II waits.
3. `t=40 ms`: Task III arrives while Task I is still running. Task III also waits.
4. `t=105 ms`: Task II starts with only 105 ms of slack left before `d_II=210 ms`; deeper exits are unsafe, so it selects `E1`.
5. `t=157 ms`: Task III starts with 103 ms of slack left before `d_III=260 ms`; it also selects `E1`.
6. Result: no deadline miss in this tiny example, but accuracy is low because the queue forces shallow early exits.

## RTInfer

1. `t=0 ms`: Task I arrives. Accuracy-Calibrated Variant Co-Optimization selects `P25-E3`: pruning ratio `0.25`, exit point `3`, latency `135 ms`, memory `1150 MiB`, accuracy `0.91`.
2. `t=0-8 ms`: Delta-Graph loads only the missing chunks for Task I (`220 MiB`) before first use.
3. `t=20 ms`: Task II arrives. The online scheduler recomputes the active set and selects `P25-E3`, accuracy `0.94`. Delta-Graph loads `140 MiB` of missing chunks in `t=20-26 ms`.
4. `t=40 ms`: Task III arrives. Full unpruned concurrent placement would exceed `4096 MiB` with Reserve IF, so RTInfer selects `P25-E2`: memory `900 MiB`, accuracy `0.86`.
5. `t=40-48 ms`: Delta-Graph loads `180 MiB` missing chunks for Task III before first use.
6. `t=48-121 ms`: Task I, Task II, and Task III are all live concurrently in the 2D Time x Address layout.
7. Result: all RT tasks meet deadlines while using deeper exits than Pantheon.
"""
    (OUT / "online_decisions.md").write_text(text)


def svg_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "#222", extra: str = "") -> str:
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="1" {extra}/>'


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "middle", weight: str = "normal") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" font-family="Arial" text-anchor="{anchor}" font-weight="{weight}">{text}</text>'


def panel_svg(events: list[Event], x0: int, y0: int, title: str) -> list[str]:
    width = 720
    height = 240
    scale_x = width / HORIZON_MS
    scale_y = height / MEM_BUDGET_MIB
    out: list[str] = []
    out.append(svg_text(x0 + width / 2, y0 - 22, title, 16, weight="bold"))
    out.append(f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#222"/>')
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#222"/>')
    out.append(svg_text(x0 - 45, y0 + height / 2, "Addr", 13, anchor="middle", weight="bold"))
    out.append(svg_text(x0 + width / 2, y0 + height + 42, "Time (ms)", 13, weight="bold"))
    out.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x0 + width}" y2="{y0}" stroke="#d7191c" stroke-width="2" stroke-dasharray="5 4"/>'
    )
    out.append(svg_text(x0 + 90, y0 - 6, "Memory Budget = 4096 MiB", 12, anchor="start", weight="bold"))
    for label, t in ARRIVALS_AND_DEADLINES:
        x = x0 + t * scale_x
        out.append(f'<line x1="{x:.2f}" y1="{y0 + height}" x2="{x:.2f}" y2="{y0 + height + 12}" stroke="#222"/>')
        color = "#d7191c" if label.startswith("d") else "#1f78b4"
        out.append(svg_text(x, y0 + height + 28, label, 11, weight="bold"))
        out.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0 + height}" stroke="{color}" stroke-width="1" stroke-dasharray="3 4" opacity="0.45"/>')
    for event in events:
        x = x0 + event.start_ms * scale_x
        w = max(2, (event.end_ms - event.start_ms) * scale_x)
        y = y0 + height - (event.addr_mib + event.memory_mib) * scale_y
        h = event.memory_mib * scale_y
        fill = COLORS.get(event.task, COLORS.get(event.kind, "#ddd"))
        extra = ""
        if event.kind == "reserve":
            extra = 'style="fill:url(#hatch)"'
        out.append(svg_rect(x, y, w, h, fill, extra=extra))
        if w > 28 and h > 14:
            out.append(svg_text(x + w / 2, y + h / 2 + 4, event.label, 11, weight="bold"))
    return out


def write_svg() -> None:
    width = 980
    height = 700
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<pattern id="hatch" patternUnits="userSpaceOnUse" width="8" height="8">',
        '<rect width="8" height="8" fill="#ffffff"/>',
        '<path d="M0 8 L8 0" stroke="#222" stroke-width="1"/>',
        "</pattern>",
        "</defs>",
        '<rect x="0" y="0" width="980" height="700" fill="#ffffff"/>',
        svg_text(490, 28, "Minimal Jetson Nano Case: Pantheon vs RTInfer", 20, weight="bold"),
    ]
    out.extend(panel_svg(PANTHEON_EVENTS, 80, 80, "(a) Pantheon / SOTA: serial RT queue forces shallow exits"))
    out.extend(panel_svg(RTINFER_EVENTS, 80, 400, "(b) RTInfer: ALC variants + memory layout + Delta-Graph loading"))
    legend_x = 830
    legend_y = 90
    out.append(svg_text(legend_x, legend_y - 20, "Legend", 14, anchor="start", weight="bold"))
    legend = [
        ("Task I", COLORS["task_i"]),
        ("Task II", COLORS["task_ii"]),
        ("Task III", COLORS["task_iii"]),
        ("BE Task", COLORS["be"]),
        ("Delta load", COLORS["load"]),
        ("Reserve IF", "#ffffff"),
    ]
    for i, (label, color) in enumerate(legend):
        y = legend_y + i * 28
        extra = 'style="fill:url(#hatch)"' if label == "Reserve IF" else ""
        out.append(svg_rect(legend_x, y, 24, 16, color, extra=extra))
        out.append(svg_text(legend_x + 34, y + 13, label, 12, anchor="start"))
    pantheon_miss, pantheon_weighted, pantheon_raw = selected_accuracy(PANTHEON_EVENTS)
    rtinfer_miss, rtinfer_weighted, rtinfer_raw = selected_accuracy(RTINFER_EVENTS)
    out.append(svg_text(legend_x, 300, "Result", 14, anchor="start", weight="bold"))
    out.append(svg_text(legend_x, 324, f"Pantheon DMR {pantheon_miss}/3", 12, anchor="start"))
    out.append(svg_text(legend_x, 346, f"Pantheon acc {pantheon_weighted:.2f}", 12, anchor="start"))
    out.append(svg_text(legend_x, 376, f"RTInfer DMR {rtinfer_miss}/3", 12, anchor="start"))
    out.append(svg_text(legend_x, 398, f"RTInfer acc {rtinfer_weighted:.2f}", 12, anchor="start"))
    out.append("</svg>")
    (OUT / "jetson_nano_case.svg").write_text("\n".join(out))


def write_summary() -> None:
    pantheon_miss, pantheon_weighted, pantheon_raw = selected_accuracy(PANTHEON_EVENTS)
    rtinfer_miss, rtinfer_weighted, rtinfer_raw = selected_accuracy(RTINFER_EVENTS)
    pantheon_rt = [event for event in PANTHEON_EVENTS if event.kind == "rt"]
    rtinfer_rt = [event for event in RTINFER_EVENTS if event.kind == "rt"]
    lines = [
        "Jetson Nano case summary",
        f"memory_budget_mib={MEM_BUDGET_MIB}",
        f"reserve_if_mib={RESERVE_IF_MIB}",
        "",
        f"Pantheon: DMR={pantheon_miss}/3, deadline_weighted_accuracy={pantheon_weighted:.4f}, raw_completed_accuracy={pantheon_raw:.4f}, makespan_ms={max(e.end_ms for e in pantheon_rt)}",
        f"RTInfer:  DMR={rtinfer_miss}/3, deadline_weighted_accuracy={rtinfer_weighted:.4f}, raw_completed_accuracy={rtinfer_raw:.4f}, makespan_ms={max(e.end_ms for e in rtinfer_rt)}",
        "",
        "Key interval: RTInfer runs Task I, Task II, and Task III concurrently during t=48-121 ms.",
        "Why better: Pantheon spends slack in a serial queue; RTInfer spends memory budget more efficiently with pruned deeper-exit variants and 2D packing.",
    ]
    (OUT / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\noutputs={OUT}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_tables()
    write_decisions()
    write_svg()
    write_summary()


if __name__ == "__main__":
    main()
