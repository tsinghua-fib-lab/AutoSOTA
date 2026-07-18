#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - fallback
    plt = None
    _matplotlib_err = exc


def load_results(paths):
    rows = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        rows.extend(data.get("results", []))
    return rows


def group_by(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return out


def ensure_sorted(rows):
    return sorted(rows, key=lambda r: (r["batch"], r["variant"]))


def fmt_float(x, places=1):
    return f"{x:.{places}f}"


def build_table(rows, model):
    rows = [r for r in rows if r["model"] == model]
    rows = ensure_sorted(rows)
    batches = sorted(set(r["batch"] for r in rows))
    lines = []
    lines.append(f"### {model} (seq={rows[0]['seq']})")
    lines.append("| Batch | Sigma ms/inf | FuseFSS ms/inf | Speedup | Sigma tokens/s | FuseFSS tokens/s | Sigma comm (GB/inf) | FuseFSS comm (GB/inf) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for b in batches:
        sigma = [r for r in rows if r["batch"] == b and r["variant"] == "sigma"][0]
        fusefss = [r for r in rows if r["batch"] == b and r["variant"] == "fusefss"][0]
        speed = sigma["online_ms_per_inf"] / fusefss["online_ms_per_inf"] if fusefss["online_ms_per_inf"] else 0.0
        lines.append(
            "| {} | {} | {} | {}x | {} | {} | {} | {} |".format(
                b,
                fmt_float(sigma["online_ms_per_inf"], 1),
                fmt_float(fusefss["online_ms_per_inf"], 1),
                fmt_float(speed, 2),
                fmt_float(sigma["tokens_per_s"], 1),
                fmt_float(fusefss["tokens_per_s"], 1),
                fmt_float(sigma["comm_gb_per_inf"], 3),
                fmt_float(fusefss["comm_gb_per_inf"], 3),
            )
        )
    lines.append("")
    return "\n".join(lines)


def plot_fig_matplotlib(rows, out_path, metric_key, ylabel, title_prefix):
    models = sorted(set(r["model"] for r in rows))
    ncols = len(models)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.2), dpi=150)
    if ncols == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        mr = [r for r in rows if r["model"] == model]
        batches = sorted(set(r["batch"] for r in mr))
        for variant, label, style in [
            ("sigma", "Sigma", "o-"),
            ("fusefss", "FuseFSS", "s-"),
        ]:
            vals = []
            for b in batches:
                row = [r for r in mr if r["batch"] == b and r["variant"] == variant][0]
                vals.append(row[metric_key])
            ax.plot(batches, vals, style, label=label, linewidth=2, markersize=5)
        ax.set_xticks(batches)
        ax.set_xlabel("Batch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix}: {model}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _svg_escape(text):
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&quot;")
                .replace("'", "&apos;"))


def plot_fig_svg(rows, out_path, metric_key, ylabel, title_prefix):
    models = sorted(set(r["model"] for r in rows))
    ncols = len(models)
    panel_w = 520
    panel_h = 360
    margin = 50
    width = ncols * panel_w + margin * 2
    height = panel_h + margin * 2

    def scale(val, vmin, vmax, pmin, pmax):
        if vmax == vmin:
            return (pmin + pmax) / 2.0
        return pmin + (val - vmin) * (pmax - pmin) / (vmax - vmin)

    svg = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"]
    svg.append("<rect width='100%' height='100%' fill='white' />")

    for idx, model in enumerate(models):
        mr = [r for r in rows if r["model"] == model]
        batches = sorted(set(r["batch"] for r in mr))
        vals = [r[metric_key] for r in mr]
        y_min = 0.0
        y_max = max(vals) * 1.1 if vals else 1.0

        x0 = margin + idx * panel_w
        y0 = margin
        plot_w = panel_w - 2 * margin
        plot_h = panel_h - 2 * margin
        x_axis_y = y0 + plot_h

        svg.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y0 + plot_h}' stroke='black' />")
        svg.append(f"<line x1='{x0}' y1='{x_axis_y}' x2='{x0 + plot_w}' y2='{x_axis_y}' stroke='black' />")

        for t in range(6):
            y_val = y_min + (y_max - y_min) * t / 5.0
            y_px = scale(y_val, y_min, y_max, x_axis_y, y0)
            svg.append(f"<line x1='{x0 - 5}' y1='{y_px}' x2='{x0}' y2='{y_px}' stroke='black' />")
            svg.append(f"<text x='{x0 - 10}' y='{y_px + 4}' font-size='10' text-anchor='end'>{y_val:.1f}</text>")

        for b in batches:
            x_px = scale(b, min(batches), max(batches), x0, x0 + plot_w)
            svg.append(f"<line x1='{x_px}' y1='{x_axis_y}' x2='{x_px}' y2='{x_axis_y + 5}' stroke='black' />")
            svg.append(f"<text x='{x_px}' y='{x_axis_y + 18}' font-size='10' text-anchor='middle'>{b}</text>")

        title = f"{title_prefix}: {model}"
        svg.append(f"<text x='{x0 + plot_w / 2}' y='{y0 - 12}' font-size='12' text-anchor='middle'>{_svg_escape(title)}</text>")
        svg.append(f"<text x='{x0 + plot_w / 2}' y='{y0 + plot_h + 34}' font-size='11' text-anchor='middle'>Batch</text>")
        svg.append(f"<text x='{x0 - 38}' y='{y0 + plot_h / 2}' font-size='11' text-anchor='middle' transform='rotate(-90 {x0 - 38},{y0 + plot_h / 2})'>{_svg_escape(ylabel)}</text>")

        for variant, color, marker in [
            ("sigma", "#1f77b4", "circle"),
            ("fusefss", "#d62728", "rect"),
        ]:
            series = []
            for b in batches:
                row = [r for r in mr if r["batch"] == b and r["variant"] == variant][0]
                x_px = scale(b, min(batches), max(batches), x0, x0 + plot_w)
                y_px = scale(row[metric_key], y_min, y_max, x_axis_y, y0)
                series.append((x_px, y_px))
            path = " ".join(f"{x},{y}" for x, y in series)
            svg.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{path}' />")
            for x_px, y_px in series:
                if marker == "circle":
                    svg.append(f"<circle cx='{x_px}' cy='{y_px}' r='3.5' fill='{color}' />")
                else:
                    svg.append(f"<rect x='{x_px - 3.5}' y='{y_px - 3.5}' width='7' height='7' fill='{color}' />")

        leg_x = x0 + plot_w - 90
        leg_y = y0 + 10
        svg.append(f"<rect x='{leg_x - 6}' y='{leg_y - 8}' width='86' height='36' fill='white' stroke='#ccc' />")
        svg.append(f"<circle cx='{leg_x}' cy='{leg_y}' r='4' fill='#1f77b4' />")
        svg.append(f"<text x='{leg_x + 10}' y='{leg_y + 4}' font-size='10'>Sigma</text>")
        svg.append(f"<rect x='{leg_x - 4}' y='{leg_y + 12}' width='8' height='8' fill='#d62728' />")
        svg.append(f"<text x='{leg_x + 10}' y='{leg_y + 20}' font-size='10'>FuseFSS</text>")

    svg.append("</svg>")
    Path(out_path).write_text("\n".join(svg))


def plot_fig(rows, out_path, metric_key, ylabel, title_prefix):
    if plt is not None:
        plot_fig_matplotlib(rows, out_path, metric_key, ylabel, title_prefix)
    else:
        if str(out_path).endswith(".png"):
            out_path = str(out_path).replace(".png", ".svg")
        plot_fig_svg(rows, out_path, metric_key, ylabel, title_prefix)


def main():
    ap = argparse.ArgumentParser(description="Plot batch scaling results and emit markdown tables.")
    ap.add_argument("--inputs", type=str, required=True,
                    help="Comma-separated JSON result files")
    ap.add_argument("--out-dir", type=str, default="artifacts/batch_plots")
    ap.add_argument("--table-out", type=str, default="artifacts/batch_scaling_tables.md")
    args = ap.parse_args()

    paths = [Path(p.strip()) for p in args.inputs.split(",") if p.strip()]
    rows = load_results(paths)
    if not rows:
        raise SystemExit("No rows loaded.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tables
    models = sorted(set(r["model"] for r in rows))
    tables = ["# Batch Scaling Tables", ""]
    for model in models:
        tables.append(build_table(rows, model))
    Path(args.table_out).write_text("\n".join(tables))

    # Plots
    plot_fig(rows, out_dir / "batch_throughput.png", "tokens_per_s", "Tokens/s", "Throughput")
    plot_fig(rows, out_dir / "batch_latency.png", "online_ms_per_inf", "ms / inference", "Latency")

    print(f"Wrote {args.table_out} and plots in {out_dir}")


if __name__ == "__main__":
    main()
