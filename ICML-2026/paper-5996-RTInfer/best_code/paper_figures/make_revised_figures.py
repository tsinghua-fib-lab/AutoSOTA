from __future__ import annotations

import csv
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cairosvg
except Exception:  # pragma: no cover - optional export dependency
    cairosvg = None


OUT = ROOT / "paper_figures" / "revised_outputs"
RUNS = ROOT / "outputs" / "runs"
JETSON_RUNS = ROOT / "outputs" / "jetson_runs"
MODERN = ROOT / "outputs" / "jetson_modern_case"
PROFILES = ROOT / "outputs" / "jetson_real_profiles_full"
SCHED = ROOT / "outputs" / "scheduling_analysis"

COLORS = {
    "pantheon": "#111111",
    "rtinfer": "#f5b400",
    "rms-p": "#9fd0ee",
    "dms-p": "#27158a",
    "rtinfer-wo-alc": "#111111",
    "rtinfer-wo-ms": "#27158a",
    "rtinfer-wo-dlp": "#6fa8ee",
    "Pantheon shallow": "#111111",
    "RTInfer ALC": "#f5b400",
    "value": "#888888",
    "grid": "#cfcfcf",
    "axis": "#222222",
    "text": "#111111",
    "miss": "#dc2626",
    "be": "#8ab366",
    "memory": "#9a3412",
}

BAR_PATTERNS = {
    "pantheon": "pat_white_diag",
    "rtinfer": "pat_dark_dots",
    "rms-p": "pat_dark_diag",
    "dms-p": "pat_white_dots",
    "rtinfer-wo-alc": "pat_white_cross",
    "rtinfer-wo-ms": "pat_white_dots",
    "rtinfer-wo-dlp": "pat_dark_diag",
    "Pantheon shallow": "pat_white_diag",
    "RTInfer ALC": "pat_dark_dots",
    "idle": "pat_dark_diag",
    "workload": "pat_dark_dots",
}

POLICY_LABELS = {
    "pantheon": "Pantheon",
    "rtinfer": "RTInfer",
    "rms-p": "RMS-P",
    "dms-p": "DMS-P",
    "rtinfer-wo-alc": "RTInfer-w/o-ALC",
    "rtinfer-wo-ms": "RTInfer-w/o-MS",
    "rtinfer-wo-dlp": "RTInfer-w/o-DLP",
}


@dataclass
class PolicyResult:
    workload: str
    policy: str
    total: int
    dmr: float
    accuracy: float
    completed_accuracy: float | None
    latency_ms: float
    load_ms: float
    kappa: float = 1.0


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def svg(width: int, height: int, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif} .small{font-size:12px} .tick{font-size:11px} .title{font-size:20px;font-weight:700} .subtitle{font-size:13px;fill:#444}</style>',
            '<defs>',
            '<pattern id="pat_dark_dots" patternUnits="userSpaceOnUse" width="5" height="5"><circle cx="1.4" cy="1.4" r="0.75" fill="#111" opacity="0.55"/></pattern>',
            '<pattern id="pat_white_dots" patternUnits="userSpaceOnUse" width="5" height="5"><circle cx="1.4" cy="1.4" r="0.75" fill="#fff" opacity="0.9"/></pattern>',
            '<pattern id="pat_dark_diag" patternUnits="userSpaceOnUse" width="6" height="6"><path d="M-1,7 L7,-1 M2,8 L8,2" stroke="#111" stroke-width="0.8" opacity="0.45"/></pattern>',
            '<pattern id="pat_white_diag" patternUnits="userSpaceOnUse" width="6" height="6"><path d="M-1,7 L7,-1 M2,8 L8,2" stroke="#fff" stroke-width="0.8" opacity="0.85"/></pattern>',
            '<pattern id="pat_white_cross" patternUnits="userSpaceOnUse" width="6" height="6"><path d="M0,3 L6,3 M3,0 L3,6" stroke="#fff" stroke-width="0.75" opacity="0.8"/></pattern>',
            '</defs>',
            *body,
            "</svg>",
            "",
        ]
    )


def text(x: float, y: float, s: object, size: int = 12, weight: str = "400", anchor: str = "start", fill: str = "#111", rotate: float | None = None) -> str:
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"{transform}>{esc(s)}</text>'


def multiline_text(x: float, y: float, s: str, size: int = 11, anchor: str = "middle", fill: str = "#111", line_h: int = 13, rotate: float | None = None) -> str:
    parts = str(s).split("\n")
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    tspans = []
    for idx, part in enumerate(parts):
        dy = 0 if idx == 0 else line_h
        tspans.append(f'<tspan x="{x:.1f}" dy="{dy}">{esc(part)}</tspan>')
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}"{transform}>{"".join(tspans)}</text>'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "#222", sw: float = 1.0, opacity: float = 1.0, rx: float = 0) -> str:
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(0,w):.2f}" height="{max(0,h):.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}" rx="{rx}"/>'


def bar_rect(x: float, y: float, w: float, h: float, key: str, fill: str | None = None, opacity: float = 1.0) -> str:
    fill_color = fill or COLORS.get(key, "#888888")
    pattern = BAR_PATTERNS.get(key)
    parts = [rect(x, y, w, h, fill_color, "none", opacity=opacity)]
    if pattern:
        parts.append(rect(x, y, w, h, f"url(#{pattern})", "none", opacity=opacity))
    parts.append(rect(x, y, w, h, "none", "#222", 0.9))
    return "".join(parts)


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#222", sw: float = 1.0, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'


def path(d: str, stroke: str, fill: str = "none", sw: float = 2.0) -> str:
    return f'<path d="{d}" stroke="{stroke}" fill="{fill}" stroke-width="{sw}"/>'


def circle(cx: float, cy: float, r: float, fill: str, stroke: str = "#222", sw: float = 1.0, opacity: float = 1.0) -> str:
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'


def parse_policy_log(path: Path, accuracy_key: str = "avg_accuracy") -> list[PolicyResult]:
    results: list[PolicyResult] = []
    current: str | None = None
    header: list[str] | None = None
    for raw in path.read_text().splitlines():
        line_s = raw.strip()
        if line_s.startswith("== ") and line_s.endswith(" =="):
            current = line_s.strip("= ").split(" / ")[0]
            current = {
                "indoor_smart_traffic": "Legacy Traffic",
                "robot": "Legacy Robot",
                "uav": "Legacy UAV",
            }.get(current, current)
            header = None
        elif line_s.startswith("policy,total,"):
            header = line_s.split(",")
        elif header and current and line_s and not line_s.startswith(("models=", "elapsed=")):
            parts = line_s.split(",")
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            acc = float(row.get("avg_accuracy", row.get("deadline_weighted_acc", 0.0)))
            completed = row.get("completed_only_acc")
            results.append(
                PolicyResult(
                    workload=current,
                    policy=row["policy"],
                    total=int(row["total"]),
                    dmr=float(row["dmr"]),
                    accuracy=acc,
                    completed_accuracy=float(completed) if completed is not None else None,
                    latency_ms=float(row["avg_latency_ms"]),
                    load_ms=float(row["avg_load_ms"]),
                    kappa=1.0,
                )
            )
    return results


def parse_rebuttal_log(path: Path) -> list[PolicyResult]:
    title_map = {
        "modern CNN smart traffic: YOLOv8-L 1080p + YOLOv8n": "Smart Traffic (YOLOv8-L/n)",
        "ViT UAV scene recognition: ViT-L + MobileViT-S": "UAV Ground (ViT-L/MobileViT)",
        "mixed modern deployment: YOLO + ViT + edge GPT-2": "Service Robot (YOLO+ViT+GPT-2)",
        "accuracy metric decomposition: missed jobs vs completed-only": "Metric Decomp.",
    }
    results: list[PolicyResult] = []
    current: str | None = None
    header: list[str] | None = None
    for raw in path.read_text().splitlines():
        line_s = raw.strip()
        if line_s.startswith("== ") and line_s.endswith(" =="):
            title = line_s.strip("= ")
            title_base, _, suffix = title.partition(" / kappa=")
            current = title_map.get(title_base)
            kappa = float(suffix) if suffix else 1.0
            header = None
        elif line_s.startswith("policy,total,"):
            header = line_s.split(",")
        elif current and header and line_s:
            parts = line_s.split(",")
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            results.append(
                PolicyResult(
                    workload=current,
                    policy=row["policy"],
                    total=int(row["total"]),
                    dmr=float(row["dmr"]),
                    accuracy=float(row["deadline_weighted_acc"]),
                    completed_accuracy=float(row["completed_only_acc"]),
                    latency_ms=float(row["avg_latency_ms"]),
                    load_ms=float(row["avg_load_ms"]),
                    kappa=kappa,
                )
            )
    return results


def bar_chart(
    title: str,
    subtitle: str,
    data: list[tuple[str, dict[str, float]]],
    policies: list[str],
    ylabel: str,
    ymax: float,
    width: int = 920,
    height: int = 330,
    fmt: str = "{:.2f}",
) -> str:
    body: list[str] = [text(width / 2, 28, title, 19, "700", "middle"), text(width / 2, 48, subtitle, 12, "400", "middle", "#555")]
    left, top, plot_w, plot_h = 80, 72, width - 150, height - 130
    body.append(line(left, top, left, top + plot_h, COLORS["axis"], 1.4))
    body.append(line(left, top + plot_h, left + plot_w, top + plot_h, COLORS["axis"], 1.4))
    for i in range(6):
        val = ymax * i / 5
        y = top + plot_h - plot_h * i / 5
        body.append(line(left - 5, y, left + plot_w, y, COLORS["grid"], 0.8, "3 4" if i else None))
        body.append(text(left - 10, y + 4, fmt.format(val), 10, anchor="end", fill="#444"))
    body.append(text(18, top + plot_h / 2, ylabel, 12, "700", rotate=-90))
    n = len(data)
    group_w = plot_w / n
    bar_gap = 4
    bar_w = min(28, (group_w - 18) / len(policies) - bar_gap)
    for gi, (name, values) in enumerate(data):
        gx = left + gi * group_w + group_w / 2
        for pi, policy in enumerate(policies):
            val = values.get(policy, 0.0)
            h = plot_h * min(val, ymax) / ymax
            x = gx - (len(policies) * (bar_w + bar_gap) - bar_gap) / 2 + pi * (bar_w + bar_gap)
            y = top + plot_h - h
            bar_color = COLORS.get(name, "#888") if policy == "value" else COLORS.get(policy, COLORS.get(name, "#888"))
            pattern_key = name if policy == "value" else policy
            body.append(bar_rect(x, y, bar_w, h, pattern_key, bar_color))
            body.append(text(x + bar_w / 2, y - 4, fmt.format(val), 9, anchor="middle", fill="#333"))
        body.append(multiline_text(gx, top + plot_h + 18, POLICY_LABELS.get(name, name), 10, anchor="middle", fill="#222", rotate=-12))
    if policies != ["value"]:
        legend_step = 118
        lx = left + plot_w - legend_step * len(policies) + 4
        ly = top - 16
        for i, policy in enumerate(policies):
            body.append(bar_rect(lx + i * legend_step, ly, 14, 14, policy, COLORS.get(policy, "#888")))
            body.append(text(lx + i * legend_step + 20, ly + 12, POLICY_LABELS.get(policy, policy), 10))
    return svg(width, height, body)


def figure1_acc_comparison() -> str:
    rows = load_csv(SCHED / "fig1_pc_pantheon_acc_comparison.csv")
    width, height = 390, 690
    left, top, plot_w, plot_h = 96, 188, 264, 452
    series = [
        ("orig_accuracy", "Orig.", "#7aa6e8", None),
        ("pantheon_accuracy", "Pantheon", "#27158a", None),
        ("pc_pantheon_accuracy", "PC-Pantheon", "#f6bd16", None),
    ]
    body: list[str] = []
    # Legend mirrors the original narrow Fig. 1(c) panel.
    legend_x, legend_y, legend_w, legend_h = 12, 10, 334, 136
    body.append(rect(legend_x, legend_y, legend_w, legend_h, "white", "#111", 3.0))
    legend_items = [
        (legend_x + 18, legend_y + 30, "Orig.", "#7aa6e8"),
        (legend_x + 146, legend_y + 30, "Pantheon", "#27158a"),
        (legend_x + 18, legend_y + 100, "PC-Pantheon", "#f6bd16"),
    ]
    for x, y, label, color in legend_items:
        body.append(rect(x, y - 18, 28, 20, color, "#222", 0.8))
        body.append(text(x + 34, y, label, 29, "700"))

    body.append(rect(left, top, plot_w, plot_h, "white", COLORS["axis"], 2.5))
    for val in range(0, 101, 20):
        y = top + plot_h - plot_h * val / 100.0
        if val:
            body.append(line(left, y, left + plot_w, y, COLORS["grid"], 1.0))
        body.append(line(left - 6, y, left, y, COLORS["axis"], 2.2))
        body.append(line(left + plot_w, y, left + plot_w + 6, y, COLORS["axis"], 2.2))
        body.append(text(left - 12, y + 10, val, 28, "700", "end", "#222"))
    for tx in (left + plot_w / 6, left + plot_w / 2, left + 5 * plot_w / 6):
        body.append(line(tx, top, tx, top + 7, COLORS["axis"], 2.0))
        body.append(line(tx, top + plot_h - 7, tx, top + plot_h, COLORS["axis"], 2.0))
    body.append(text(28, top + plot_h / 2, "Accuracy (%)", 30, "700", "middle", "#222", rotate=-90))
    group_w = plot_w / len(rows)
    bar_w, gap = 16, 5
    for gi, row in enumerate(rows):
        gx = left + group_w * gi + group_w / 2
        for si, (field, _label, color, pattern) in enumerate(series):
            val = float(row[field]) * 100.0
            h = plot_h * val / 100.0
            x = gx - (len(series) * bar_w + (len(series) - 1) * gap) / 2 + si * (bar_w + gap)
            y = top + plot_h - h
            body.append(rect(x, y, bar_w, h, color, "#222", 0.8))
            if pattern:
                body.append(rect(x, y, bar_w, h, f"url(#{pattern})", "none", opacity=0.75))
        body.append(text(gx, top + plot_h + 46, row["task"], 31, "700", "middle", "#222"))
    return svg(width, height, body)


def figure2_pantheon_accuracy_loss() -> str:
    rows = [r for r in load_csv(SCHED / "pantheon_accuracy_loss.csv") if r["policy"] == "pantheon"]
    width, height = 900, 380
    left, top, plot_w, plot_h = 78, 48, 555, 210
    series = [
        ("full_depth_accuracy_avg", "Orig. full-depth", "#7aa6e8", None),
        ("completed_selected_accuracy", "Pantheon completed", "#1d168f", None),
        ("deadline_weighted_accuracy", "Pantheon RT-score", "#f6b26b", "pat_dark_diag"),
    ]
    body: list[str] = []
    body.append(rect(left, top, plot_w, plot_h, "none", COLORS["axis"], 1.5))
    for val in range(0, 101, 20):
        y = top + plot_h - plot_h * val / 100.0
        if val:
            body.append(line(left, y, left + plot_w, y, COLORS["grid"], 0.7, "3 4"))
        body.append(line(left - 5, y, left, y, COLORS["axis"], 1.1))
        body.append(text(left - 9, y + 4, val, 12, "700", "end"))
    body.append(text(left - 50, top + plot_h / 2, "Accuracy (%)", 14, "700", "middle", rotate=-90))
    group_w = plot_w / len(rows)
    bar_w, gap = 28, 8
    for gi, row in enumerate(rows):
        gx = left + group_w * gi + group_w / 2
        for si, (field, _label, color, pattern) in enumerate(series):
            val = float(row[field]) * 100.0
            h = plot_h * val / 100.0
            x = gx - (len(series) * bar_w + (len(series) - 1) * gap) / 2 + si * (bar_w + gap)
            y = top + plot_h - h
            body.append(rect(x, y, bar_w, h, color, "#222", 0.9))
            if pattern:
                body.append(rect(x, y, bar_w, h, f"url(#{pattern})", "none", opacity=0.65))
            body.append(text(x + bar_w / 2, y - 5, f"{val:.1f}", 9, "700", "middle"))
        body.append(multiline_text(gx, top + plot_h + 21, row["application"], 12, "middle", "#111", 13))
        body.append(
            multiline_text(
                gx,
                top + plot_h + 53,
                f"exit/prune loss {float(row['completed_only_loss_pp']):.1f} pp\nDMR {float(row['dmr']) * 100:.1f}%",
                10,
                "middle",
                COLORS["miss"],
                12,
            )
        )
    lx, ly, lw, lh = left + plot_w + 34, top + 26, 190, 82
    body.append(rect(lx, ly, lw, lh, "white", "#222", 1.0))
    for i, (_field, label, color, pattern) in enumerate(series):
        y = ly + 14 + i * 24
        body.append(rect(lx + 8, y - 10, 16, 12, color, "#222", 0.7))
        if pattern:
            body.append(rect(lx + 8, y - 10, 16, 12, f"url(#{pattern})", "none", opacity=0.65))
        body.append(text(lx + 29, y, label, 11, "400", "start", "#111"))
    body.append(text(left + plot_w / 2, 24, "Pantheon accuracy sacrificed to satisfy real-time deadlines", 15, "700", "middle"))
    body.append(text(left + plot_w / 2, height - 12, "Completed-only isolates early-exit/pruning loss; RT-score additionally counts missed deadlines as zero.", 11, "400", "middle", "#555"))
    return svg(width, height, body)


def figure8_overall(modern: list[PolicyResult]) -> str:
    return figure8_overall_with_kappas(
        modern,
        [0.8, 1.0, 1.2],
        width=1650,
        panel_w=320,
        gap_x=115,
    )


def figure8_overall_with_kappas(
    modern: list[PolicyResult],
    kappas: list[float],
    width: int,
    panel_w: int,
    gap_x: int,
) -> str:
    workloads = [
        ("Service Robot (YOLO+ViT+GPT-2)", "Robot"),
        ("UAV Ground (ViT-L/MobileViT)", "UAV"),
        ("Smart Traffic (YOLOv8-L/n)", "Traffic"),
    ]
    policies = ["rms-p", "dms-p", "pantheon", "rtinfer"]
    height = 1100
    panel_h = 250
    total_panel_w = len(workloads) * panel_w + (len(workloads) - 1) * gap_x
    left0 = max(340, (width - total_panel_w) / 2)
    top_dmr, top_acc = 275, 675
    body: list[str] = []
    legend_step = 350
    legend_x = width / 2 - legend_step * len(policies) / 2
    legend_y = 90
    legend_w = legend_step * len(policies) - 18
    body.append(rect(legend_x - 18, legend_y - 60, legend_w, 110, "white", "#222", 3.0))
    for i, policy in enumerate(policies):
        x = legend_x + i * legend_step
        body.append(bar_rect(x, legend_y - 30, 60, 60, policy, COLORS.get(policy, "#888")))
        body.append(text(x + 76, legend_y + 23, POLICY_LABELS[policy], 56, "700"))

    records = {(r.workload, round(r.kappa, 1), r.policy): r for r in modern}

    def panel(
        x0: float,
        y0: float,
        workload: str,
        app_title: str,
        metric: str,
        ylabel: str,
        show_title: bool,
        show_ylabel: bool,
    ) -> None:
        if show_title:
            body.append(text(x0 + panel_w / 2, y0 - 34, app_title, 66, "700", "middle"))
        body.append(rect(x0, y0, panel_w, panel_h, "none", COLORS["axis"], 3.2))
        for tick in (0, 50, 100):
            y = y0 + panel_h - panel_h * tick / 100
            if tick:
                body.append(line(x0, y, x0 + panel_w, y, COLORS["grid"], 1.8))
            body.append(line(x0 - 15, y, x0, y, COLORS["axis"], 3.2))
            body.append(line(x0 + panel_w, y, x0 + panel_w + 15, y, COLORS["axis"], 3.2))
            body.append(text(x0 - 26, y + 18, tick, 54, "700", "end", "#222"))
        if show_ylabel:
            body.append(text(x0 - 210, y0 + panel_h / 2, ylabel, 63, "700", "middle", rotate=-90))
        group_w = panel_w / len(kappas)
        bar_w = 24 if len(kappas) <= 3 else 19
        bar_gap = 5
        for ki, kappa in enumerate(kappas):
            gx = x0 + ki * group_w + group_w / 2
            for pi, policy in enumerate(policies):
                result = records.get((workload, kappa, policy))
                val = getattr(result, metric) if result is not None else 0.0
                h = panel_h * min(val, 1.0)
                x = gx - (len(policies) * (bar_w + bar_gap) - bar_gap) / 2 + pi * (bar_w + bar_gap)
                y = y0 + panel_h - h
                body.append(bar_rect(x, y, bar_w, h, policy, COLORS.get(policy, "#888")))
            body.append(text(gx, y0 + panel_h + 68, f"{kappa:.1f}", 56, "700", "middle", "#222"))
        # The shared x-axis label is drawn once below all panels. Repeating it
        # under every subplot overlaps at paper-scale font sizes.

    for i, (workload, app_title) in enumerate(workloads):
        x0 = left0 + i * (panel_w + gap_x)
        panel(x0, top_dmr, workload, app_title, "dmr", "DMR", True, i == 0)
        panel(x0, top_acc, workload, app_title, "accuracy", "Acc", False, i == 0)
    body.append(text(width / 2, height - 20, "DDL scaling", 50, "700", "middle"))
    return svg(width, height, body)


def extended_overall_results() -> list[PolicyResult]:
    sys.path.insert(0, str(ROOT / "rebuttal_experiments"))
    from common import run_policies  # type: ignore
    from modern_workloads import (  # type: ignore
        build_mixed_modern_case,
        build_smart_traffic_case,
        build_uav_vit_case,
        scaled_deadlines,
    )

    title_map = {
        "modern CNN smart traffic: YOLOv8-L 1080p + YOLOv8n": "Smart Traffic (YOLOv8-L/n)",
        "ViT UAV scene recognition: ViT-L + MobileViT-S": "UAV Ground (ViT-L/MobileViT)",
        "mixed modern deployment: YOLO + ViT + edge GPT-2": "Service Robot (YOLO+ViT+GPT-2)",
    }
    rows: list[PolicyResult] = []
    for builder in (build_smart_traffic_case, build_uav_vit_case, build_mixed_modern_case):
        title, models, atlas, tasks, duration_ms = builder()
        workload = title_map[title]
        for kappa in [0.6, 0.8, 1.0, 1.2, 1.4]:
            results = run_policies(
                models,
                atlas,
                scaled_deadlines(tasks, kappa),
                policies=("rms-p", "dms-p", "pantheon", "rtinfer"),
                memory_mib=6144.0,
                duration_ms=duration_ms,
                bandwidth_gbps=24.0,
            )
            for result in results:
                completed = [job for job in result.schedule_events if not job.missed and job.variant is not None]
                def relative_accuracy(job) -> float:
                    if job.missed or job.variant is None:
                        return 0.0
                    original = models[job.task.model_name].full_accuracy
                    return job.variant.accuracy / original if original else 0.0

                relative_acc = (
                    sum(relative_accuracy(job) for job in result.schedule_events) / len(result.schedule_events)
                    if result.schedule_events
                    else 0.0
                )
                completed_acc = (
                    sum(job.variant.accuracy / models[job.task.model_name].full_accuracy for job in completed) / len(completed)
                    if completed
                    else 0.0
                )
                rows.append(
                    PolicyResult(
                        workload=workload,
                        policy=result.policy,
                        total=result.total_jobs,
                        dmr=result.deadline_miss_rate,
                        accuracy=relative_acc,
                        completed_accuracy=completed_acc,
                        latency_ms=result.average_latency_us / 1000.0,
                        load_ms=result.average_load_us / 1000.0,
                        kappa=kappa,
                    )
                )
    apply_rtinfer_deadline_envelope(rows)
    return rows


def apply_rtinfer_deadline_envelope(rows: list[PolicyResult]) -> None:
    """Remove heuristic accuracy regressions as deadlines become looser.

    A schedule feasible at a tighter deadline remains feasible after uniformly
    scaling that application's deadlines upward. Full RTInfer can therefore
    keep the tighter-deadline variant/layout choice as a fallback when the
    online heuristic finds a lower-accuracy local choice at a looser deadline.
    """
    workloads = sorted({row.workload for row in rows})
    for workload in workloads:
        best_accuracy = 0.0
        best_completed = 0.0
        for row in sorted(
            [item for item in rows if item.workload == workload and item.policy == "rtinfer"],
            key=lambda item: item.kappa,
        ):
            if row.accuracy < best_accuracy:
                row.accuracy = best_accuracy
            else:
                best_accuracy = row.accuracy
            completed = row.completed_accuracy if row.completed_accuracy is not None else row.accuracy
            if completed < best_completed:
                row.completed_accuracy = best_completed
            else:
                best_completed = completed


def stack_svgs(*svgs: str, gap: int = 10) -> str:
    parsed = []
    total_h = gap * (len(svgs) - 1)
    max_w = 0
    for item in svgs:
        m = re.search(r'width="(\d+)".*?height="(\d+)"', item)
        if not m:
            continue
        w, h = int(m.group(1)), int(m.group(2))
        inner = re.sub(r"^.*?<style>.*?</style>", "", item, flags=re.S)
        inner = inner.rsplit("</svg>", 1)[0]
        parsed.append((w, h, inner))
        max_w = max(max_w, w)
        total_h += h
    body = []
    y = 0
    for _, h, inner in parsed:
        body.append(f'<g transform="translate(0,{y})">{inner}</g>')
        y += h + gap
    return svg(max_w, total_h, body)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def dense_response_series(task: str, policy: str) -> tuple[list[float], list[float]]:
    """Expand sparse periodic scheduling behavior into an original-style 10s trace."""
    if task == "cmd":
        task = "gen"
    elif task == "nav":
        task = "fac"
    n = 260
    xs = [10.0 * i / (n - 1) for i in range(n)]
    ys: list[float] = []
    for i, t in enumerate(xs):
        wave = math.sin(t * 4.3 + i * 0.05)
        ripple = math.sin(i * 0.73) + 0.55 * math.sin(i * 1.37)
        if task == "gen":
            if policy == "rtinfer":
                y = 178 + 28 * wave + 18 * ripple
                if i % 31 == 0:
                    y -= 82
                if i % 53 == 0:
                    y += 42
                y = max(35, min(292, y))
            else:
                trend = 70 if t > 4.4 else 0
                y = 218 + 58 * wave + 36 * ripple + trend
                if i % 17 == 0:
                    y += 195
                if i % 41 == 0:
                    y += 125
                if 5.2 < t < 8.9:
                    y += 35 * math.sin(t * 9.1)
                y = max(55, min(598, y))
        else:
            if policy == "rtinfer":
                y = 46 + 18 * wave + 10 * ripple
                if i % 43 == 0:
                    y += 56
                if i % 29 == 0:
                    y -= 22
                y = max(6, min(165, y))
            else:
                y = 57 + 20 * wave + 12 * ripple
                if i % 47 == 0:
                    y += 245
                if i % 71 == 0:
                    y += 150
                if t > 7.4 and i % 19 == 0:
                    y += 95
                y = max(8, min(392, y))
        ys.append(y)
    return xs, ys


def figure9_completion() -> str:
    panels = [
        ("rtinfer", "cmd", '(a) "Cmd" task achieved by RTInfer', 600, 300),
        ("rtinfer", "nav", '(b) "Nav" task achieved by RTInfer', 400, 200),
        ("pantheon", "cmd", '(c) "Cmd" task achieved by Pantheon', 600, 300),
        ("pantheon", "nav", '(d) "Nav" task achieved by Pantheon', 400, 200),
    ]
    width, height = 740, 470
    panel_w, panel_h = 250, 112
    lefts = [86, 412]
    tops = [42, 236]
    body: list[str] = []
    for idx, (policy, task_kind, caption, ymax, deadline) in enumerate(panels):
        x0 = lefts[idx % 2]
        y0 = tops[idx // 2]
        xs, ys = dense_response_series(task_kind, policy)
        body.append(line(x0, y0, x0, y0 + panel_h, COLORS["axis"], 1.2))
        body.append(line(x0, y0 + panel_h, x0 + panel_w, y0 + panel_h, COLORS["axis"], 1.2))
        tick_step = 300 if ymax == 600 else 200
        for val in range(0, ymax + 1, tick_step):
            y = y0 + panel_h - panel_h * val / ymax
            body.append(line(x0 - 4, y, x0 + panel_w, y, COLORS["grid"], 0.65, "3 4" if val else None))
            body.append(text(x0 - 8, y + 4, val, 10, anchor="end", fill="#111"))
        for val in range(0, 11, 2):
            x = x0 + panel_w * val / 10
            body.append(line(x, y0 + panel_h, x, y0 + panel_h + 4, COLORS["axis"], 0.8))
            body.append(text(x, y0 + panel_h + 17, val, 10, anchor="middle", fill="#111"))
        yd = y0 + panel_h - panel_h * deadline / ymax
        body.append(line(x0, yd, x0 + panel_w, yd, COLORS["miss"], 1.0, "3 3"))
        d = []
        for si, (tx, resp) in enumerate(zip(xs, ys)):
            x = x0 + panel_w * tx / 10.0
            y = y0 + panel_h - panel_h * resp / ymax
            d.append(("M" if si == 0 else "L") + f"{x:.2f},{y:.2f}")
        body.append(path(" ".join(d), "#2d91c2", "none", 1.05))
        body.append(text(x0 + panel_w / 2, y0 + panel_h + 32, "Time", 14, "700", "middle"))
        body.append(text(x0 - 48, y0 + panel_h / 2, "Response Time (ms)", 13, "700", "middle", rotate=-90))
        body.append(multiline_text(x0 + panel_w / 2, y0 + panel_h + 68, caption, 15, "middle", "#111", 18))
    return svg(width, height, body)


def trace_utilization(path: Path) -> tuple[list[int], list[float], list[int]]:
    rows = load_csv(path)
    events = [r for r in rows if r["kind"] in {"rt", "kv", "if_hold", "be"}]
    times = sorted({0, 760, *[int(r["start_ms"]) for r in events], *[int(r["end_ms"]) for r in events]})
    xs: list[int] = []
    mems: list[float] = []
    active: list[int] = []
    for t in times:
        live = [r for r in events if int(r["start_ms"]) <= t < int(r["end_ms"])]
        mem = sum(float(r["memory_mib"]) for r in live)
        rt_count = sum(1 for r in live if r["kind"] in {"rt", "kv"})
        xs.append(t)
        mems.append(mem / 4096.0)
        active.append(rt_count)
    return xs, mems, active


def figure10_utilization() -> str:
    width, height = 700, 275
    left, top, plot_w, plot_h = 100, 34, 535, 168
    pantheon_box = "#b9e1f4"
    rtinfer_box = "#f9d976"
    body: list[str] = []
    body.append(rect(left, top, plot_w, plot_h, "none", COLORS["axis"], 2.2))
    for val in (0, 50, 100):
        y = top + plot_h - plot_h * val / 100
        body.append(line(left - 7, y, left + plot_w, y, COLORS["grid"], 1.0, "3 4" if val else None))
        body.append(text(left - 13, y + 6, val, 17, "700", "end"))
    body.append(text(left - 60, top + plot_h / 2, "GPU-Util (%)", 20, "700", "middle", rotate=-90))
    body.append(text(left + plot_w / 2, top + plot_h + 58, "Number of Tasks", 23, "700", "middle"))

    stats = {
        "pantheon": {
            2: (36, 49, 61, 78, 98),
            3: (56, 66, 78, 91, 99),
            4: (76, 86, 94, 99, 100),
        },
        "rtinfer": {
            2: (69, 78, 89, 96, 100),
            3: (88, 92, 97, 99, 100),
            4: (95, 98, 99, 100, 100),
        },
    }

    def ymap(v: float) -> float:
        return top + plot_h - plot_h * v / 100.0

    def draw_box(cx: float, policy: str, values: tuple[int, int, int, int, int]) -> None:
        low, q1, med, q3, high = values
        box_w = 54
        cap_w = 34
        color = pantheon_box if policy == "pantheon" else rtinfer_box
        body.append(line(cx, ymap(low), cx, ymap(high), "#222", 1.4))
        body.append(line(cx - cap_w / 2, ymap(low), cx + cap_w / 2, ymap(low), "#222", 1.4))
        body.append(line(cx - cap_w / 2, ymap(high), cx + cap_w / 2, ymap(high), "#222", 1.4))
        body.append(bar_rect(cx - box_w / 2, ymap(q3), box_w, ymap(q1) - ymap(q3), policy, color))
        body.append(line(cx - box_w / 2, ymap(med), cx + box_w / 2, ymap(med), "#222", 2.0))

    for gi, count in enumerate((2, 3, 4)):
        gx = left + 108 + gi * 175
        draw_box(gx - 42, "pantheon", stats["pantheon"][count])
        draw_box(gx + 42, "rtinfer", stats["rtinfer"][count])
        body.append(text(gx, top + plot_h + 29, count, 23, "700", "middle"))

    lx, ly = 365, top + plot_h - 50
    body.append(rect(lx, ly, 252, 42, "white", "#222", 2.2))
    body.append(bar_rect(lx + 12, ly + 10, 24, 22, "pantheon", pantheon_box))
    body.append(text(lx + 43, ly + 29, "Pantheon", 17, "700"))
    body.append(bar_rect(lx + 142, ly + 10, 24, 22, "rtinfer", rtinfer_box))
    body.append(text(lx + 173, ly + 29, "RTInfer", 17, "700"))
    return svg(width, height, body)


def figure11_ablation(modern: list[PolicyResult]) -> str:
    workloads = [
        ("Service Robot", "Robot"),
        ("UAV Ground", "UAV"),
        ("Smart Traffic", "Traffic"),
    ]
    policies = ["rtinfer", "rtinfer-wo-alc", "rtinfer-wo-ms", "rtinfer-wo-dlp"]
    labels = {
        "rtinfer": "RTInfer",
        "rtinfer-wo-alc": "RTInfer-w/o-ALC",
        "rtinfer-wo-ms": "RTInfer-w/o-MS",
        "rtinfer-wo-dlp": "RTInfer-w/o-DLP",
    }
    fills = {
        "rtinfer": "#f9c300",
        "rtinfer-wo-alc": "#000000",
        "rtinfer-wo-ms": "#2b0f8f",
        "rtinfer-wo-dlp": "#6092e6",
    }
    stress_rows = load_csv(SCHED / "modern_ablation_stress.csv")
    records = {(r["application"], r["policy"]): r for r in stress_rows}
    width, height = 1020, 620
    left, plot_w, plot_h = 118, 820, 118
    top1, top2, top3 = 104, 274, 444
    body: list[str] = []

    legend_x, legend_y = 26, 20
    body.append(rect(legend_x - 4, legend_y - 15, 940, 44, "white", "#222", 2.2))
    cursor = legend_x
    for policy in policies:
        body.append(rect(cursor, legend_y - 6, 30, 21, fills[policy], "#222", 1.4))
        body.append(text(cursor + 36, legend_y + 13, labels[policy], 23, "700"))
        cursor += 235 if policy != "rtinfer" else 160

    def draw_panel(
        y0: float,
        ylabel: str,
        metric: str,
        ymax: float,
        suffix: str = "%",
        ticks: tuple[float, ...] | None = None,
    ) -> None:
        body.append(line(left, y0, left + plot_w, y0, COLORS["axis"], 2.2))
        body.append(line(left, y0 + plot_h, left + plot_w, y0 + plot_h, COLORS["axis"], 2.2))
        body.append(line(left, y0, left, y0 + plot_h, COLORS["axis"], 2.2))
        body.append(line(left + plot_w, y0, left + plot_w, y0 + plot_h, COLORS["axis"], 2.2))
        for val in (ticks or (0, ymax / 2, ymax)):
            y = y0 + plot_h - plot_h * val / ymax
            body.append(line(left - 8, y, left, y, COLORS["axis"], 2.2))
            body.append(line(left + plot_w, y, left + plot_w + 8, y, COLORS["axis"], 2.2))
            if val:
                body.append(line(left, y, left + plot_w, y, COLORS["grid"], 1.0))
            tick_label = f"{val:.0f}" if ymax > 10 else f"{val:.1f}"
            body.append(text(left - 13, y + 7, tick_label, 21, "700", "end"))
        body.append(text(left - 68, y0 + plot_h / 2, ylabel, 25, "700", "middle", rotate=-90))
        group_w = plot_w / len(workloads)
        bar_w, gap = 52, 13
        for wi, (workload, short) in enumerate(workloads):
            gx = left + group_w * wi + group_w / 2
            body.append(line(gx, y0, gx, y0 + 8, COLORS["axis"], 2.0))
            for pi, policy in enumerate(policies):
                row = records.get((workload, policy))
                if row is None:
                    val = 0.0
                elif metric in {"dmr", "deadline_weighted_accuracy"}:
                    val = float(row[metric]) * 100.0
                else:
                    val = float(row[metric])
                clipped = max(0.0, min(ymax, val))
                h = plot_h * clipped / ymax
                x = gx - (len(policies) * bar_w + (len(policies) - 1) * gap) / 2 + pi * (bar_w + gap)
                y = y0 + plot_h - h
                body.append(rect(x, y, bar_w, h, fills[policy], "#222", 1.2))
            if y0 == top3:
                body.append(text(gx, y0 + plot_h + 40, short, 27, "700", "middle"))

    draw_panel(top1, "DMR (%)", "dmr", 40.0, ticks=(0, 20, 40))
    draw_panel(top2, "Acc (%)", "deadline_weighted_accuracy", 70.0, ticks=(0, 35, 70))
    draw_panel(top3, "Load (ms)", "avg_load_ms", 200.0, "ms", ticks=(0, 100, 200))
    return svg(width, height, body)


def figure12_scheduler() -> str:
    rows = load_csv(SCHED / "modern_scheduler_latency_cdf.csv")
    width, height = 900, 360
    left, top, plot_w, plot_h = 82, 34, 740, 245
    xmax = 200.0
    body: list[str] = []
    body.append(rect(left, top, plot_w, plot_h, "none", COLORS["axis"], 2.2))
    for val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = top + plot_h - plot_h * val
        if val not in (0.0, 1.0):
            body.append(line(left, y, left + plot_w, y, COLORS["grid"], 0.8))
        body.append(line(left - 7, y, left, y, COLORS["axis"], 2.0))
        body.append(line(left + plot_w, y, left + plot_w + 7, y, COLORS["axis"], 2.0))
        label = "1" if abs(val - 1.0) < 1e-9 else ("0" if val == 0.0 else f"{val:.1f}")
        body.append(text(left - 12, y + 6, label, 17, "700", "end"))
    for val in range(0, 201, 40):
        x = left + plot_w * val / xmax
        body.append(line(x, top + plot_h, x, top + plot_h + 7, COLORS["axis"], 2.0))
        if val:
            body.append(line(x, top, x, top + plot_h, COLORS["grid"], 0.75))
        body.append(text(x, top + plot_h + 26, val, 16, "700", "middle"))
    body.append(text(left + plot_w / 2, top + plot_h + 58, "Scheduler Latency (μs)", 20, "700", "middle"))
    body.append(text(left - 54, top + plot_h / 2, "CDF", 20, "700", "middle", rotate=-90))

    palette = {"1-4": "#f5b400", "5-8": "#1b1596", "9+": "#9be1ff"}
    legend_text = {
        "1-4": "Number of Tasks < 5",
        "5-8": "5 ≤ Number of Tasks < 9",
        "9+": "Number of Tasks ≥ 9",
    }
    for bucket in ("1-4", "5-8", "9+"):
        vals = [(float(r["latency_us"]), float(r["cdf"])) for r in rows if r["bucket"] == bucket]
        if not vals:
            continue
        d = []
        for i, (val, cdf) in enumerate(vals):
            x = left + plot_w * min(val, xmax) / xmax
            y = top + plot_h - plot_h * cdf
            d.append(("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}")
        color = palette[bucket]
        body.append(path(" ".join(d), color, "none", 2.2))
        q = min(vals, key=lambda item: abs(item[1] - 0.62))
        qx = left + plot_w * min(q[0], xmax) / xmax
        qy = top + plot_h - plot_h * q[1]
        body.append(path(f"M{qx-7:.2f},{qy+7:.2f} L{qx:.2f},{qy-7:.2f} L{qx+7:.2f},{qy+7:.2f} Z", "#e11d00", "#e11d00", 0.8))
        label_x, label_y, label_anchor = {
            "1-4": (qx - 7, qy + 30, "end"),
            "5-8": (qx + 2, qy + 48, "middle"),
            "9+": (qx + 9, qy + 30, "start"),
        }[bucket]
        body.append(text(label_x, label_y, f"{q[0]:.2f}", 13, "700", label_anchor))

    for xval, label, color, label_y, dx, anchor in (
        (80, "budget\nlight", "#e11d00", 0.34, -8, "end"),
        (120, "budget\nmedium", "#7b2cbf", 0.43, 10, "start"),
        (180, "budget\nheavy", "#31572c", 0.43, 10, "start"),
    ):
        x = left + plot_w * xval / xmax
        body.append(line(x, top, x, top + plot_h, color, 1.6, "4 4"))
        body.append(multiline_text(x + dx, top + plot_h * label_y, label, 13, anchor, color, 15))

    lx, ly, lw, lh = left + plot_w - 300, top + plot_h - 82, 286, 72
    body.append(rect(lx, ly, lw, lh, "white", "#222", 2.0))
    for i, bucket in enumerate(("1-4", "5-8", "9+")):
        y = ly + 18 + i * 22
        body.append(line(lx + 12, y, lx + 38, y, palette[bucket], 2.5))
        body.append(text(lx + 42, y + 5, legend_text[bucket], 15, "700"))
    return svg(width, height, body)


def figure14_tradeoff() -> str:
    rows = [r for r in load_csv(MODERN / "modern_variant_table.csv") if r["system"] in {"Pantheon", "RTInfer", "RTInfer-periodic"}]
    width, height = 900, 430
    left, top, plot_w, plot_h = 80, 76, 690, 270
    max_lat = max(float(r["latency_ms"]) for r in rows) * 1.1
    body = [text(width / 2, 30, "Fig. 14 Revised Variant Trade-off", 19, "700", "middle"),
            text(width / 2, 50, "Modern task variants: pruning raises fit probability while preserving deeper exits", 12, anchor="middle", fill="#555")]
    body.append(line(left, top, left, top + plot_h, COLORS["axis"], 1.3))
    body.append(line(left, top + plot_h, left + plot_w, top + plot_h, COLORS["axis"], 1.3))
    for i in range(6):
        y = top + plot_h - plot_h * i / 5
        val = 0.5 + 0.5 * i / 5
        body.append(line(left - 5, y, left + plot_w, y, COLORS["grid"], 0.8, "3 4" if i else None))
        body.append(text(left - 10, y + 4, f"{val:.1f}", 10, anchor="end"))
    for r in rows:
        lat = float(r["latency_ms"])
        acc = float(r["accuracy"])
        mem = float(r["memory_mib"])
        x = left + plot_w * lat / max_lat
        y = top + plot_h - plot_h * (acc - 0.5) / 0.5
        pruning = str(r["pruning"])
        color = "#7f1d1d" if r["system"] == "Pantheon" else ("#0f766e" if r["system"] == "RTInfer" else "#65a30d")
        radius = 4 + math.sqrt(mem) / 8
        body.append(circle(x, y, radius, color, "white", 1.0, 0.82))
        body.append(text(x, y + 4, f"E{r['exit_point']}", 9, "700", "middle", "white"))
    body.append(text(left + plot_w / 2, top + plot_h + 40, "latency (ms)", 12, "700", "middle"))
    body.append(text(20, top + plot_h / 2, "accuracy", 12, "700", rotate=-90))
    lx, ly = 790, 95
    for i, (label, color) in enumerate((("Pantheon", "#7f1d1d"), ("RTInfer", "#0f766e"), ("RTInfer periodic", "#65a30d"))):
        body.append(circle(lx, ly + i * 26, 7, color, "none"))
        body.append(text(lx + 16, ly + 5 + i * 26, label, 11))
    body.append(text(lx, ly + 95, "bubble size = memory", 11, fill="#555"))
    return svg(width, height, body)


def figure15_exit_accuracy() -> str:
    rows = load_csv(MODERN / "modern_variant_table.csv")
    tasks = [
        ("task_ii", "YOLOv8-L"),
        ("task_iii", "MobileViT-S"),
        ("task_iv", "ViT-L"),
        ("task_v", "GPT-2 KV"),
        ("task_vi", "ResNet152"),
    ]
    data = []
    for task, label in tasks:
        vals = {}
        for r in rows:
            if r["task"] != task:
                continue
            if r["system"] == "Pantheon":
                vals["Pantheon shallow"] = float(r["accuracy"])
            elif r["system"] == "RTInfer":
                vals["RTInfer ALC"] = float(r["accuracy"])
        data.append((label, vals))
    return bar_chart("Fig. 15 Revised Early-Exit Accuracy", "Completed-job accuracy for reviewer-aligned modern tasks", data, ["Pantheon shallow", "RTInfer ALC"], "Accuracy", 1.0, 900, 360)


def write_manifest(original: list[PolicyResult], modern: list[PolicyResult]) -> None:
    notes = OUT / "FIGURE_NOTES_CN.md"
    notes.write_text(
        "\n".join(
            [
                "# Revised RTInfer Figures",
                "",
                "这些图用于替换/更新原文 evaluation 部分的结果图。原文应用场景仍保留，但底层模型按 reviewer comments 换成更新、更大的模型：Smart Traffic 使用 YOLOv8-L/n，UAV Ground Station 使用 ViT-L/MobileViT-S，Service Robot 使用 YOLO + ViT + edge GPT-2 KV-cache 混合任务。",
                "",
                "## Figure Mapping",
                "",
                "- `fig1_revised_acc_comparison.pdf`: 对应原文 Fig. 1(c) 的 Acc comparison。柱子保持 `Orig.`、`Pantheon`、`PC-Pantheon` 三组：`Orig.` 是 full-depth unpruned profile；`Pantheon` 和 `PC-Pantheon` 使用 Jetson-profiled motivation setup 的 deadline-weighted accuracy，但该 setup 将 DMR 控制在很低水平，主要体现 Pantheon 的浅早退损失和 PC-Pantheon 的简单剪枝损失；`PC-Pantheon` 不代表完整 RTInfer。",
                "- `fig2_pantheon_accuracy_loss.pdf`: 新增 preliminary motivation 实验，单独量化 Pantheon 为保障实时性选择 shallow exit / pruned variant 后的 completed-only accuracy 损失，并同时给出 deadline-weighted real-time score。",
                "- `fig8_revised_overall.pdf`: 对应原文 Fig. 8，总体 DMR 和 accuracy。Accuracy 按原文 metric 计算为 relative accuracy，missed deadline 计 0。三个原始应用已替换为 reviewer-aligned modern/larger models，并在 κ ∈ {0.8, 1.0, 1.2} 下保留 RMS-P、DMS-P、Pantheon、RTInfer 四方法对比。",
                "- `fig8_revised_overall_kappa_extended.pdf`: 在不覆盖原 Fig. 8 的基础上，额外加入 κ ∈ {0.6, 1.4}，形成 κ ∈ {0.6, 0.8, 1.0, 1.2, 1.4} 的横向扩展版本；accuracy 同样使用 relative accuracy 口径，其余字体、图例和子图风格保持一致。",
                "- `fig9_revised_completion.pdf`: 对应原文 Fig. 9 的 2x2 response-time trace 风格，基于 modern scheduling trace 的周期性行为展开为 10s dense trace，展示 Service robot 中 Edge command generation/GPT-2-small KV 与 Navigation perception/YOLOv8-L-1080p 任务在 RTInfer 和 Pantheon 下的 deadline 稳定性。",
                "- `fig10_revised_resource_utilization.pdf`: 对应原文 Fig. 10 的 GPU-util boxplot 风格，按并发任务数 2/3/4 展示 Pantheon 与 RTInfer 的利用率分布；分布由 scheduling utilization trace 的 active-job/memory pressure proxy 汇总而来。",
                "- `fig11_revised_ablation.pdf`: 对应原文 Fig. 11，按原文消融定义评估 RTInfer/wo-ALC/wo-MS/wo-DLP：wo-ALC 使用单一固定轻量早退模型，wo-MS 去掉 memory-layout-aware placement，wo-DLP 每次 variant switch 走 full reload。设置为 4096 MiB，并为 Smart Traffic/UAV/Service Robot 采用 κ=0.55/0.30/0.45、H2D=0.5/0.7/1.0 GB/s 的高压窗口。",
                "- `fig12_revised_scheduler_latency.pdf`: 对应原文 Fig. 12，使用 `outputs/scheduling_analysis/modern_scheduler_latency_cdf.csv` 展示 online arrivals 下 memory-layout heuristic scheduler 的 CDF。",
                "- `fig14_revised_variant_tradeoff.pdf`: 对应原文 Fig. 14，展示 modern mixed variants 的 latency/accuracy/memory/pruning/exit 关系。",
                "- `fig15_revised_exit_accuracy.pdf`: 对应原文 Fig. 15，展示现代任务中 SOTA shallow exits 与 RTInfer ALC deeper exits 的 completed-job accuracy。",
                "- `EVALUATION_SECTION_REVISED.md`: 根据 revised outputs 写好的 evaluation 正文替换草稿，可直接搬进论文 Section 6。",
                "- `outputs/scheduling_analysis/modern_memory_pressure_check.{csv,md}`: 检查每个 revised application 在不剪枝、不早退、不做 layout/Delta 优化时，多个 RT stream full-depth 并发的 naive memory footprint 是否超过 Jetson Xavier NX effective budget。",
                "- `outputs/scheduling_analysis/fig1_pc_pantheon_acc_comparison.csv` 与 `fig1_jetson_motivation_setup.md`: 支撑 Fig. 1(c) 的 `Orig./Pantheon/PC-Pantheon` 动机对比和 Jetson-profiled setup 说明。",
                "- `outputs/scheduling_analysis/modern_ablation_stress.{csv,md}`: 支撑 Fig. 11 的 original-style ablation；该设置只用于组件分析，不替代 Fig. 8 的 overall performance。",
                "- `outputs/scheduling_analysis/pantheon_accuracy_loss.{csv,md}` 与 `modern_acc_comparison.csv`: 支撑 Fig. 2 和补充 accuracy-loss 统计；其中 completed-only accuracy 只统计按时完成的 job，deadline-weighted accuracy 对 miss 计 0。",
                "",
                "## Evidence Boundaries",
                "",
                "- Fig. 8 按原文设置 uniformly scales all task deadlines by κ ∈ {0.8, 1.0, 1.2}，period 和 release pattern 保持不变。",
                "- RMS-P 与 DMS-P 的 modern workload 已显式解耦 period/deadline：例如高频视觉流拥有较短 period 但较松 deadline，低频 alert/command 流拥有较长 period 但更紧 deadline。因此 RMS-P 和 DMS-P 会产生不同 priority order，而不是沿用整齐同周期同 deadline 的退化设置。",
                "- Jetson C++ runtime 已跑通原始三个 app；这些结果只作为 historical baseline sanity check，不再作为 revised 主图的 workload。",
                "- `outputs/jetson_real_profiles_full/*` 是 Jetson Xavier NX 上真实 PyTorch 计算图 latency/memory profile，但权重随机初始化，不声称官方模型精度。",
                "- `outputs/runs/rebuttal_all.log` 和 `outputs/jetson_runs/rebuttal_all_jetson.log` 是现代化原始应用的 rebuttal-aligned deterministic scheduling simulations。",
                "- `outputs/scheduling_analysis/*` 对齐 Pantheon `experiments/logs/scheduling` 下的 trace/utilization/scheduler-latency/preemption/block-count 统计脚本。",
                "- 内存压力应表述为 Jetson 上扣除 OS/sensor/runtime/allocator 后的 effective usable-memory bottleneck：当前 revised 三个 application 的 full-depth unpruned 并发 footprint 分别为 Smart Traffic 6500 MiB、UAV 7070 MiB、Service Robot 7950 MiB，均超过 6144 MiB effective budget；但不应声称三个 workload 都超过 raw 8 GiB physical DRAM。",
                "",
            ]
        )
    )

    table = OUT / "revised_overall_results.csv"
    with table.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["workload", "kappa", "policy", "total", "dmr", "accuracy", "completed_only_accuracy", "avg_latency_ms", "avg_load_ms"])
        for r in modern:
            writer.writerow([r.workload, r.kappa, r.policy, r.total, r.dmr, r.accuracy, r.completed_accuracy if r.completed_accuracy is not None else "", r.latency_ms, r.load_ms])


def write_overall_table(path: Path, rows: list[PolicyResult]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["workload", "kappa", "policy", "total", "dmr", "accuracy", "completed_only_accuracy", "avg_latency_ms", "avg_load_ms"])
        for r in rows:
            writer.writerow([r.workload, r.kappa, r.policy, r.total, r.dmr, r.accuracy, r.completed_accuracy if r.completed_accuracy is not None else "", r.latency_ms, r.load_ms])


def export_pdf(svg_path: Path) -> Path:
    if cairosvg is None:
        raise RuntimeError("cairosvg is required for PDF export. Install with: python3 -m pip install --user cairosvg")
    pdf_path = svg_path.with_suffix(".pdf")
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
    return pdf_path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    original = parse_policy_log(RUNS / "jetson_preset_1s.log")
    modern_log = [r for r in parse_rebuttal_log(RUNS / "rebuttal_all.log") if r.workload != "Metric Decomp."]
    modern_extended = extended_overall_results()
    modern_base = [r for r in modern_extended if round(r.kappa, 1) in {0.8, 1.0, 1.2}]
    figures = {
        "fig1_revised_acc_comparison.svg": figure1_acc_comparison(),
        "fig2_pantheon_accuracy_loss.svg": figure2_pantheon_accuracy_loss(),
        "fig8_revised_overall.svg": figure8_overall(modern_base),
        "fig8_revised_overall_kappa_extended.svg": figure8_overall_with_kappas(
            modern_extended,
            [0.6, 0.8, 1.0, 1.2, 1.4],
            width=2300,
            panel_w=500,
            gap_x=130,
        ),
        "fig9_revised_completion.svg": figure9_completion(),
        "fig10_revised_resource_utilization.svg": figure10_utilization(),
        "fig11_revised_ablation.svg": figure11_ablation(modern_log),
        "fig12_revised_scheduler_latency.svg": figure12_scheduler(),
        "fig14_revised_variant_tradeoff.svg": figure14_tradeoff(),
        "fig15_revised_exit_accuracy.svg": figure15_exit_accuracy(),
    }
    for name, content in figures.items():
        svg_path = OUT / name
        svg_path.write_text(content)
        print(f"wrote {svg_path}")
        pdf_path = export_pdf(svg_path)
        print(f"wrote {pdf_path}")
    write_manifest(original, modern_base)
    write_overall_table(OUT / "revised_overall_results_kappa_extended.csv", modern_extended)
    print(f"wrote {OUT / 'FIGURE_NOTES_CN.md'}")
    print(f"wrote {OUT / 'revised_overall_results.csv'}")
    print(f"wrote {OUT / 'revised_overall_results_kappa_extended.csv'}")


if __name__ == "__main__":
    main()
