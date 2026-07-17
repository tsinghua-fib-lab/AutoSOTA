#!/usr/bin/env python3
"""Plot rate-distortion curves for Llama-3.2-1B: WikiText-2 PPL (left) and C4 PPL (right).

Three configurations: no finetuning, finetuning on WikiText-2, finetuning on RedPajama.
Also includes GPTQ baseline.

Reads eval.json and eval_c4.json from run directories.

Usage:
    python scripts/plot_rate_distortion_1B.py \
        --output_dir /home/semyon/quant-bucket/quant_runs/3.2-1B/plots

    # Custom base dir:
    python scripts/plot_rate_distortion_1B.py \
        --base_dir /home/semyon/quant-bucket/quant_runs/3.2-1B \
        --output_dir ./plots
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_ppl(run_dir: Path, dataset: str = "w2") -> float | None:
    """Load PPL from eval.json or eval_c4.json."""
    if dataset == "c4":
        eval_path = run_dir / "eval_c4.json"
    else:
        eval_path = run_dir / "eval.json"
    if not eval_path.exists():
        return None
    try:
        data = json.loads(eval_path.read_text())
        return data["eval"]["ppl_quant"]
    except (KeyError, json.JSONDecodeError):
        return None


def collect_data(base_dir: Path):
    """Collect PPL data for all methods/rates/suffixes."""
    rates = [1.00, 1.50, 2.00, 2.50, 3.00, 3.18, 3.50, 3.76, 4.00]
    gptq_rates = [1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00]

    methods = {
        "WaterSIC": {
            "pattern": "3.2-1B.zsic.wiki.qronos.rescomp.attnw_joint.qadapt.r{rate}",
            "rates": rates,
        },
        "WaterSIC-FT (W2)": {
            "pattern": "3.2-1B.zsic.wiki.qronos.rescomp.attnw_joint.qadapt.r{rate}_tuned_w2",
            "rates": rates,
        },
        "WaterSIC-FT (RP)": {
            "pattern": "3.2-1B.zsic.wiki.qronos.rescomp.attnw_joint.qadapt.r{rate}_tuned_rp",
            "rates": rates,
        },
        "GPTQ": {
            "pattern": "3.2-1B.gptq.r{rate}",
            "rates": gptq_rates,
        },
    }

    results = {}  # method -> {"w2": [(rate, ppl), ...], "c4": [(rate, ppl), ...]}

    for method_name, cfg in methods.items():
        w2_points = []
        c4_points = []
        for rate in cfg["rates"]:
            rate_str = f"{rate:.2f}"
            dir_name = cfg["pattern"].format(rate=rate_str)
            run_dir = base_dir / dir_name
            if not run_dir.exists():
                continue
            w2_ppl = load_ppl(run_dir, "w2")
            c4_ppl = load_ppl(run_dir, "c4")
            if w2_ppl is not None:
                w2_points.append((rate, w2_ppl))
            if c4_ppl is not None:
                c4_points.append((rate, c4_ppl))
        results[method_name] = {"w2": w2_points, "c4": c4_points}

    return results


def plot_rd(results: dict, output_dir: Path, ref_ppl_w2: float = 9.67, ref_ppl_c4: float = 12.77):
    """Generate side-by-side rate-distortion plots and save as SVG + PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)

    styles = {
        "WaterSIC": {"color": "#1f77b4", "marker": "o", "linewidth": 2},
        "WaterSIC-FT (W2)": {"color": "#2ca02c", "marker": "^", "linewidth": 2},
        "WaterSIC-FT (RP)": {"color": "#ff7f0e", "marker": "D", "linewidth": 2},
        "GPTQ": {"color": "#d62728", "marker": "s", "linewidth": 1.5, "linestyle": "--"},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for dataset, ax, ref_ppl, title in [
        ("w2", ax1, ref_ppl_w2, "WikiText-2 Test PPL"),
        ("c4", ax2, ref_ppl_c4, "C4 Test PPL"),
    ]:
        for method_name, data in results.items():
            points = data[dataset]
            if not points:
                continue
            rates, ppls = zip(*sorted(points))
            style = styles.get(method_name, {"color": "gray", "marker": "x", "linewidth": 1})
            ax.plot(rates, ppls, label=method_name,
                    marker=style["marker"], color=style["color"],
                    linewidth=style["linewidth"],
                    linestyle=style.get("linestyle", "-"),
                    markersize=7)

        # Reference line
        ax.axhline(y=ref_ppl, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                   label=f"FP16 baseline ({ref_ppl:.2f})")

        ax.set_xlabel("Rate (bits/param)", fontsize=11)
        ax.set_ylabel("Perplexity", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.8, 4.2)

    plt.tight_layout()

    # Save as SVG (vector, perfect for LaTeX \includegraphics)
    svg_path = output_dir / "rate_distortion_1B.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    print(f"Saved: {svg_path}")

    # Also save PDF (alternative vector format for LaTeX)
    pdf_path = output_dir / "rate_distortion_1B.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {pdf_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot rate-distortion curves for Llama-3.2-1B")
    parser.add_argument("--base_dir", type=str,
                        default="/home/semyon/quant-bucket/quant_runs/3.2-1B",
                        help="Base directory containing run directories")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: base_dir/plots)")
    parser.add_argument("--ref_ppl_w2", type=float, default=9.67,
                        help="FP16 baseline WikiText-2 PPL")
    parser.add_argument("--ref_ppl_c4", type=float, default=12.77,
                        help="FP16 baseline C4 PPL")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "plots"

    results = collect_data(base_dir)

    # Print summary
    for method, data in results.items():
        print(f"{method}: {len(data['w2'])} W2 points, {len(data['c4'])} C4 points")

    plot_rd(results, output_dir, ref_ppl_w2=args.ref_ppl_w2, ref_ppl_c4=args.ref_ppl_c4)


if __name__ == "__main__":
    main()
