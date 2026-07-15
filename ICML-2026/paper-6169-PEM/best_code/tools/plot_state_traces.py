#!/usr/bin/env python3
"""Plot internal optimizer state traces (temperature/noise proxy) saved by noisy-wrapper runners."""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def read_state_index(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["budget_multiplier"] = int(row["budget_multiplier"])
            row["function"] = int(row["function"])
            row["dimension"] = int(row["dimension"])
            row["instance"] = int(row["instance"])
            row["noise_sigma"] = float(row.get("noise_sigma", "nan"))
            rows.append(row)
    return rows


def read_state_trace(path: str) -> dict[str, list[float]]:
    cols = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key in {"evals", "generation", "reeval_count", "gate_closed", "noise_z_pool_size"}:
                    cols[key].append(int(float(val)))
                else:
                    cols[key].append(float(val))
    return cols


def shade_gate(ax, evals, gate_closed):
    if not gate_closed:
        return
    if all(v < 0 for v in gate_closed):
        return
    # Shade regions where gate_closed == 1.
    in_block = False
    start = None
    for i in range(len(evals)):
        closed = gate_closed[i] == 1
        if closed and not in_block:
            in_block = True
            start = evals[i]
        if in_block and (not closed or i == len(evals) - 1):
            end = evals[i] if not closed else evals[i]
            if start is not None:
                ax.axvspan(start, end, color="#CBD5E1", alpha=0.35, linewidth=0)
            in_block = False
            start = None


def plot_state_trace(row: dict, output_dir: str) -> str:
    state_file = row["state_file"]
    data = read_state_trace(state_file)

    evals = data.get("evals", [])
    if not evals:
        return ""

    def has_any_finite(key: str) -> bool:
        series = data.get(key, [])
        return any(v == v and abs(v) != float("inf") for v in series)

    has_hetero = any(has_any_finite(k) for k in ["noise_s0", "noise_s1", "noise_z_abs_median"]) or bool(
        data.get("noise_z_pool_size", [])
    )

    if has_hetero:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.2, 7.8), sharex=True)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.0), sharex=True)
        ax3 = None

    ax1.plot(evals, data.get("temp_scale", []), label="temp_scale", color="#1D4ED8", linewidth=2)
    if "n_eff" in data:
        ax1.plot(evals, data.get("n_eff", []), label="n_eff", color="#0F766E", linewidth=1.6, alpha=0.85)
    if "n_sched" in data:
        ax1.plot(evals, data.get("n_sched", []), label="n_sched", color="#64748B", linewidth=1.2, alpha=0.8)
    ax1.set_ylabel("Temperature / n")
    ax1.grid(True, alpha=0.25)

    ax1b = None
    if "mueff" in data or "mueff_target" in data:
        ax1b = ax1.twinx()
        if "mueff" in data:
            ax1b.plot(evals, data.get("mueff", []), label="mueff", color="#F59E0B", linewidth=1.3, alpha=0.9)
        if "mueff_target" in data:
            ax1b.plot(
                evals,
                data.get("mueff_target", []),
                label="mueff_target",
                color="#B45309",
                linewidth=1.1,
                alpha=0.85,
                linestyle="--",
            )
        ax1b.set_ylabel("mueff")

    ax2.plot(evals, data.get("noise_level", []), label="noise_level", color="#DC2626", linewidth=1.7, alpha=0.9)
    ax2.plot(evals, data.get("noise_ema", []), label="noise_ema", color="#991B1B", linewidth=1.2, alpha=0.8)
    if "reeval_count" in data:
        ax2.plot(
            evals,
            data.get("reeval_count", []),
            label="reeval_count",
            color="#7C3AED",
            linewidth=1.0,
            alpha=0.65,
        )
    ax2.set_xlabel("Evaluations")
    ax2.set_ylabel("Noise proxy / reeval")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8)

    if ax3 is not None:
        ax3.plot(evals, data.get("noise_s0", []), label="noise_s0", color="#1D4ED8", linewidth=1.4, alpha=0.85)
        ax3.plot(evals, data.get("noise_s1", []), label="noise_s1", color="#0F766E", linewidth=1.4, alpha=0.85)
        if "noise_z_abs_median" in data:
            ax3.plot(
                evals,
                data.get("noise_z_abs_median", []),
                label="|z| median",
                color="#64748B",
                linewidth=1.2,
                alpha=0.8,
                linestyle="--",
            )
        ax3.set_ylabel("Hetero noise model")
        ax3.grid(True, alpha=0.25)

        ax3b = None
        if "noise_z_pool_size" in data:
            ax3b = ax3.twinx()
            ax3b.plot(
                evals,
                data.get("noise_z_pool_size", []),
                label="z_pool_size",
                color="#B45309",
                linewidth=1.0,
                alpha=0.75,
            )
            ax3b.set_ylabel("pool size")

        if ax3b is not None:
            h1, l1 = ax3.get_legend_handles_labels()
            h2, l2 = ax3b.get_legend_handles_labels()
            ax3.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
        else:
            ax3.legend(fontsize=8, loc="upper right")

    if ax1b is not None:
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1b.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=8)
    else:
        ax1.legend(fontsize=8)

    gate_closed = data.get("gate_closed", [])
    shade_gate(ax1, evals, gate_closed)
    shade_gate(ax2, evals, gate_closed)
    if ax3 is not None:
        shade_gate(ax3, evals, gate_closed)

    title = (
        f"{row['algorithm']} | f{row['function']} D={row['dimension']} "
        f"B={row['budget_multiplier']}x i={row['instance']} | "
        f"noise={row['noise_model']}({row['noise_sigma']})"
    )
    fig.suptitle(title, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{row['state_id']}.png"
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=os.path.join(base_dir, "Results", "largescale_noisy_wrapper"),
        help="Directory containing state_index.csv and traces/",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    state_index_path = os.path.join(results_dir, "state_index.csv")
    state_rows = read_state_index(state_index_path)
    if not state_rows:
        raise SystemExit(f"No state traces found: {state_index_path}")

    plots_dir = os.path.join(results_dir, "plots", "state_traces")
    os.makedirs(plots_dir, exist_ok=True)

    written = 0
    for row in state_rows:
        try:
            out = plot_state_trace(row, plots_dir)
        except FileNotFoundError:
            continue
        if out:
            written += 1

    print("State trace plots saved to", plots_dir)
    print("Plotted traces:", written)


if __name__ == "__main__":
    main()
