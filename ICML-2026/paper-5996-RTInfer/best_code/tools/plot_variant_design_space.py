from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path


PRUNING_TIERS = (0.75, 0.5, 0.25, 0.0)
EXIT_INDICES = (1, 2, 3, 4)


def load_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def escape(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(
    x: float,
    y: float,
    value: object,
    size: int,
    weight: str = "400",
    anchor: str = "start",
    rotate: int | None = None,
    fill: str = "#111",
) -> str:
    rotate_attr = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"{rotate_attr}>'
        f"{escape(value)}</text>"
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str = "#222", width: float = 1.0) -> str:
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"/>'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", sw: float = 0.0) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def palette(value: float) -> str:
    value = (max(0.4, min(1.0, value)) - 0.4) / 0.6
    stops = [
        (0.0, (57, 36, 149)),
        (0.33, (43, 125, 246)),
        (0.66, (34, 188, 172)),
        (0.84, (247, 190, 60)),
        (1.0, (255, 245, 0)),
    ]
    for idx in range(len(stops) - 1):
        left, c0 = stops[idx]
        right, c1 = stops[idx + 1]
        if left <= value <= right:
            alpha = 0.0 if right == left else (value - left) / (right - left)
            rgb = tuple(round(c0[i] + alpha * (c1[i] - c0[i])) for i in range(3))
            return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    rgb = stops[-1][1]
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def matrix(rows: list[dict[str, float]], key: str) -> dict[tuple[float, int], float]:
    return {(row["pruning"], int(row["exit_index"])): row[key] for row in rows}


def draw_panel(
    body: list[str],
    rows: list[dict[str, float]],
    x0: float,
    y0: float,
    title: str,
    metric: str,
    label_values: bool,
) -> None:
    panel_w, panel_h = 225, 225
    cell_w = panel_w / len(EXIT_INDICES)
    cell_h = panel_h / len(PRUNING_TIERS)
    values = matrix(rows, metric)
    max_value = max(values.values())
    min_value = min(values.values())
    for ri, pruning in enumerate(PRUNING_TIERS):
        for ci, exit_index in enumerate(EXIT_INDICES):
            raw = values[(pruning, exit_index)]
            if max_value > min_value:
                value = 0.4 + 0.6 * (raw - min_value) / (max_value - min_value)
            else:
                value = 1.0
            x = x0 + ci * cell_w
            y = y0 + ri * cell_h
            body.append(rect(x, y, cell_w, cell_h, palette(value)))
            if label_values:
                label = f"{raw:.2f}" if metric == "accuracy" else f"{raw:.1f}"
                body.append(text(x + cell_w / 2, y + cell_h / 2 + 5, label, 10, "700", "middle"))
    body.append(rect(x0, y0, panel_w, panel_h, "none", "#222", 2.0))
    for ci in range(1, len(EXIT_INDICES)):
        body.append(line(x0 + ci * cell_w, y0, x0 + ci * cell_w, y0 + panel_h, "#ffffff", 0.55))
    for ri in range(1, len(PRUNING_TIERS)):
        body.append(line(x0, y0 + ri * cell_h, x0 + panel_w, y0 + ri * cell_h, "#ffffff", 0.55))
    for ci, exit_index in enumerate(EXIT_INDICES):
        body.append(text(x0 + (ci + 0.5) * cell_w, y0 + panel_h + 26, exit_index, 18, "700", "middle"))
    for ri, pruning in enumerate(PRUNING_TIERS):
        body.append(text(x0 - 12, y0 + (ri + 0.5) * cell_h + 6, f"{pruning:g}", 16, "700", "end"))
    body.append(text(x0 + panel_w / 2, y0 + panel_h + 54, "Early Exit Index", 18, "700", "middle"))
    body.append(text(x0 - 58, y0 + panel_h / 2, "Pruning Ratio p", 18, "700", "middle", -90))
    body.append(text(x0 + panel_w / 2, y0 + panel_h + 92, title, 19, "700", "middle"))


def draw_colorbar(body: list[str], x: float, y: float, h: float) -> None:
    steps = 80
    for i in range(steps):
        value = 1.0 - 0.6 * i / (steps - 1)
        body.append(rect(x, y + h * i / steps, 18, h / steps + 0.5, palette(value)))
    body.append(rect(x, y, 18, h, "none", "#222", 1.2))
    for value in (1.0, 0.8, 0.6, 0.4):
        yy = y + h * (1.0 - value) / 0.6
        if y <= yy <= y + h:
            body.append(line(x + 18, yy, x + 25, yy, "#222", 1.2))
            body.append(text(x + 30, yy + 5, f"{value:g}", 14, "700"))


def write_svg(rows: list[dict[str, float]], label_values: bool) -> str:
    width, height = 1040, 440
    y0 = 58
    xs = (92, 392, 692)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "white"),
    ]
    draw_panel(body, rows, xs[0], y0, "(a) Accuracy", "accuracy", label_values)
    draw_panel(body, rows, xs[1], y0, "(b) Latency", "normalized_latency", label_values)
    draw_panel(body, rows, xs[2], y0, "(c) Memory", "normalized_memory", label_values)
    draw_colorbar(body, 954, y0, 225)
    body.append("</svg>\n")
    return "\n".join(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Fig. 14 variant design-space heatmaps.")
    parser.add_argument("--csv", type=Path, default=Path("outputs/jetson_variant_design_space/variant_design_space.csv"))
    parser.add_argument("--out-prefix", type=Path, default=Path("paper_figures/revised_outputs/fig14_revised_variant_tradeoff"))
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--label-values", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    svg = write_svg(rows, args.label_values)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    svg_path = args.out_prefix.with_suffix(".svg")
    pdf_path = args.out_prefix.with_suffix(".pdf")
    png_path = args.out_prefix.with_suffix(".png")
    svg_path.write_text(svg)
    try:
        import cairosvg

        cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(pdf_path))
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png_path), output_width=2080, output_height=880)
    except Exception as exc:
        print(f"warning: wrote SVG, but export failed: {exc}")

    summary = {
        "accuracy_range": [min(row["accuracy"] for row in rows), max(row["accuracy"] for row in rows)],
        "latency_ms_range": [min(row["latency_ms"] for row in rows), max(row["latency_ms"] for row in rows)],
        "memory_mib_range": [min(row["tensor_footprint_mib"] for row in rows), max(row["tensor_footprint_mib"] for row in rows)],
        "note": "Latency is measured on Jetson Xavier NX; memory is parameter plus forward-activation tensor footprint; accuracy is recomputed from the calibrated ALC variant model.",
    }
    summary_path = args.summary_out or args.out_prefix.with_name(args.out_prefix.name + "_summary").with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    preview = args.out_prefix.parent / "png_preview" / args.out_prefix.with_suffix(".png").name
    preview.parent.mkdir(parents=True, exist_ok=True)
    if png_path.exists():
        shutil.copy2(png_path, preview)
    print(f"wrote {svg_path}")
    if pdf_path.exists():
        print(f"wrote {pdf_path}")
    if png_path.exists():
        print(f"wrote {png_path}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
