#!/usr/bin/env python3
"""
Plot distribution shift experiment results.

Reads schedule stats from eb and eb_kratio runs, and generates:
  - Panel 1: θ* (k_ratio) over iteration — shows IFR adaptation vs fixed ratio
  - Panel 2: Sliding-window throughput (tokens/s) over time
  - Panel 3: Preemption count (memory safety indicator)

Supports both 2-phase (legacy) and multi-phase experiments.

Usage:
    python reproduce/eb_plus/non_stationary/ or long_context/ or disagg/plot_distribution_shift.py \
        outputs/distribution_shift_Qwen3-8B_20260325_120000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCHEDULER_DISPLAY = {
    "eb": "EB(k̂*) (adaptive)",
    "eb_kratio": r"EB(k̂*) (fixed $\theta^{*}\!=\!0.8$)",
}

SCHEDULER_COLORS = {
    "eb": "#2A6F97",
    "eb_kratio": "#B23A3A",
}

PHASE_COLORS = ["#E8F4FD", "#FFF3E0", "#FFEBEE", "#E8F5E9", "#F3E5F5"]

PHASE_DISPLAY = {
    "prefill-heavy": "Prefill-heavy",
    "balanced": "Balanced",
    "decode-heavy": "Decode-heavy",
}


def configure_style() -> None:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 13
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11


def load_stats(stats_path: Path) -> List[Dict[str, Any]]:
    """Load schedule stats JSON, handling possibly truncated files."""
    with open(stats_path) as f:
        content = f.read().strip()

    # Try parsing as-is
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "stats" in data:
            return data["stats"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try repairing truncated JSON (missing closing brackets)
    for suffix in ["]}", "]", "]}]}}"]:
        try:
            data = json.loads(content + suffix)
            if isinstance(data, dict) and "stats" in data:
                return data["stats"]
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Cannot parse stats file: {stats_path}")


def compute_sliding_throughput(
    stats: List[Dict[str, Any]],
    window_sec: float = 5.0,
) -> tuple[List[float], List[float]]:
    """Compute sliding-window throughput in tokens/s."""
    if not stats:
        return [], []

    timestamps = [s.get("timestamp", 0) for s in stats]
    tokens = [s.get("total_tokens", 0) for s in stats]

    t_out = []
    thr_out = []

    for i, t in enumerate(timestamps):
        # Find window start
        j = i
        while j > 0 and timestamps[j] > t - window_sec:
            j -= 1
        window_tokens = sum(tokens[j:i + 1])
        window_time = timestamps[i] - timestamps[j] if i > j else window_sec
        if window_time > 0:
            t_out.append(t)
            thr_out.append(window_tokens / window_time)

    return t_out, thr_out


def estimate_shift_iterations(
    stats: List[Dict[str, Any]],
    num_prompts_per_phase: int,
    num_phases: int = 2,
) -> List[int]:
    """Estimate iteration indices where distribution shifts occur.

    Returns a list of shift iterations (one per phase boundary).
    For num_phases=3, returns 2 shift points.
    """
    shift_iters = []
    cum_new = 0
    next_boundary = 1

    for i, s in enumerate(stats):
        cum_new += s.get("num_new_reqs", 0)
        if cum_new >= next_boundary * num_prompts_per_phase and next_boundary < num_phases:
            shift_iters.append(i)
            next_boundary += 1

    return shift_iters


def _add_phase_shading(
    ax,
    shift_iters: List[int],
    total_iters: int,
    phases: List[Dict[str, Any]],
    use_x_values: bool = True,
    x_values: Optional[List[float]] = None,
) -> None:
    """Add subtle background shading and labels for each phase."""
    boundaries = [0] + shift_iters + [total_iters]
    if x_values is not None:
        # Map iteration boundaries to x-axis values (e.g., timestamps)
        boundaries_x = []
        for b in boundaries:
            idx = min(b, len(x_values) - 1)
            boundaries_x.append(x_values[idx] if idx >= 0 else x_values[0])
    else:
        boundaries_x = boundaries

    for j in range(len(boundaries_x) - 1):
        color = PHASE_COLORS[j % len(PHASE_COLORS)]
        ax.axvspan(boundaries_x[j], boundaries_x[j + 1],
                   alpha=0.15, color=color, zorder=0)

        # Add phase label at top
        mid_x = (boundaries_x[j] + boundaries_x[j + 1]) / 2
        if j < len(phases):
            phase_info = phases[j]
            name = PHASE_DISPLAY.get(phase_info.get("name", ""), phase_info.get("name", f"Phase {j+1}"))
            label = f"{name}\n(in={phase_info.get('input_mean', '?')}, out={phase_info.get('output_mean', '?')})"
        else:
            label = f"Phase {j + 1}"
        ax.text(mid_x, 0.97, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, alpha=0.7,
                style="italic")


def plot_distribution_shift(
    exp_dir: Path,
    output_path: Path,
    window_sec: float = 5.0,
) -> None:
    """Generate the 3-panel distribution shift figure."""

    config_path = exp_dir / "experiment_config.json"
    with open(config_path) as f:
        config = json.load(f)

    num_prompts_per_phase = config.get("num_prompts_per_phase", 2000)
    window_size = config.get("ifr_window_size", 500)
    num_phases = config.get("num_phases", 2)
    phases = config.get("phases", [])

    schedulers = ["eb", "eb_kratio"]
    all_stats = {}
    for sched in schedulers:
        stats_path = exp_dir / f"{sched}_stats.json"
        if stats_path.exists():
            all_stats[sched] = load_stats(stats_path)
            print(f"  {sched}: {len(all_stats[sched])} iterations")
        else:
            print(f"  {sched}: stats file not found, skipping")

    if not all_stats:
        print("No stats data found!")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    # Find shift points from first available scheduler's stats
    shift_iters = []
    ref_stats = list(all_stats.values())[0]
    shift_iters = estimate_shift_iterations(
        ref_stats, num_prompts_per_phase, num_phases
    )
    total_iters = len(ref_stats)

    # ── Panel 1: θ* (k_ratio) over iteration ──
    ax1 = axes[0]

    # Phase shading
    if phases:
        _add_phase_shading(ax1, shift_iters, total_iters, phases)

    for sched, stats in all_stats.items():
        k_ratios = [s.get("k_ratio", None) for s in stats]
        iters = list(range(len(k_ratios)))

        # Filter out None values
        valid = [(i, k) for i, k in zip(iters, k_ratios) if k is not None]
        if valid:
            xi, yk = zip(*valid)
            label = SCHEDULER_DISPLAY.get(sched, sched)
            ax1.plot(
                xi, yk,
                color=SCHEDULER_COLORS.get(sched, "gray"),
                linewidth=1.5, alpha=0.8,
                label=label,
            )

    # Mark shift points
    for si in shift_iters:
        ax1.axvline(
            si, color="black", linestyle="--",
            linewidth=1.5, alpha=0.7,
        )
    ax1.set_ylabel(r"$\theta^{*}$ (k_ratio)")
    ax1.set_title(
        r"$\theta^{*}$ Convergence Under Distribution Shift"
        f" (W={window_size})",
        fontweight="bold",
    )
    # Build legend with shift-line entry
    handles, labels = ax1.get_legend_handles_labels()
    if shift_iters:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=1.5))
        labels.append("Distribution shift")
    ax1.legend(handles, labels, loc="best", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Sliding-window throughput ──
    ax2 = axes[1]

    # Phase shading (using timestamps)
    if phases and ref_stats:
        timestamps = [s.get("timestamp", 0) for s in ref_stats]
        _add_phase_shading(
            ax2, shift_iters, total_iters, phases,
            x_values=timestamps,
        )

    for sched, stats in all_stats.items():
        t_vals, thr_vals = compute_sliding_throughput(stats, window_sec)
        if t_vals:
            label = SCHEDULER_DISPLAY.get(sched, sched)
            ax2.plot(
                t_vals, thr_vals,
                color=SCHEDULER_COLORS.get(sched, "gray"),
                linewidth=1.5, alpha=0.8,
                label=label,
            )

    # Mark shift times
    for si in shift_iters:
        if si < len(ref_stats):
            shift_time = ref_stats[si].get("timestamp", 0)
            ax2.axvline(
                shift_time, color="black", linestyle="--",
                linewidth=1.5, alpha=0.7,
            )

    ax2.set_ylabel("Throughput (tokens/s)")
    ax2.set_title("Sliding-Window Throughput", fontweight="bold")
    ax2.legend(loc="best", framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Cumulative preemptions (memory safety) ──
    ax3 = axes[2]

    # Phase shading
    if phases:
        _add_phase_shading(ax3, shift_iters, total_iters, phases)

    for sched, stats in all_stats.items():
        preemptions = [s.get("num_preempted_reqs", 0) for s in stats]
        cum_preempt = np.cumsum(preemptions)
        iters = list(range(len(cum_preempt)))

        label = SCHEDULER_DISPLAY.get(sched, sched)
        ax3.plot(
            iters, cum_preempt,
            color=SCHEDULER_COLORS.get(sched, "gray"),
            linewidth=1.5, alpha=0.8,
            label=label,
        )

    for si in shift_iters:
        ax3.axvline(
            si, color="black", linestyle="--",
            linewidth=1.5, alpha=0.7,
        )

    ax3.set_xlabel("Scheduling Iteration")
    ax3.set_ylabel("Cumulative Preemptions")
    ax3.set_title("Memory Safety (Preemptions)", fontweight="bold")
    ax3.legend(loc="best", framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()

    for fmt in ["pdf", "png"]:
        out = output_path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


def print_summary(exp_dir: Path) -> None:
    """Print convergence summary statistics."""
    config_path = exp_dir / "experiment_config.json"
    with open(config_path) as f:
        config = json.load(f)

    num_prompts_per_phase = config.get("num_prompts_per_phase", 2000)
    num_phases = config.get("num_phases", 2)
    phases = config.get("phases", [])

    for sched in ["eb", "eb_kratio"]:
        stats_path = exp_dir / f"{sched}_stats.json"
        if not stats_path.exists():
            continue

        stats = load_stats(stats_path)
        display = SCHEDULER_DISPLAY.get(sched, sched)

        # Find shift points
        shift_iters = estimate_shift_iterations(
            stats, num_prompts_per_phase, num_phases
        )

        # Compute k_ratio stats
        k_ratios = [s.get("k_ratio", None) for s in stats]
        k_ratios_clean = [k for k in k_ratios if k is not None]

        # Track cumulative new requests
        new_reqs = [s.get("num_new_reqs", 0) for s in stats]
        cum_new = np.cumsum(new_reqs)

        print(f"\n{display}:")
        print(f"  Total iterations: {len(stats)}")
        print(f"  Total new reqs: {int(cum_new[-1]) if len(cum_new) > 0 else 0}")

        # Per-phase statistics
        boundaries = [0] + shift_iters + [len(stats)]
        for p_idx in range(len(boundaries) - 1):
            start, end = boundaries[p_idx], boundaries[p_idx + 1]
            phase_k = [k for k in k_ratios[start:end] if k is not None]
            phase_preempt = sum(
                s.get("num_preempted_reqs", 0) for s in stats[start:end]
            )
            phase_new_reqs = sum(new_reqs[start:end])

            phase_label = ""
            if p_idx < len(phases):
                p = phases[p_idx]
                phase_label = f" ({p.get('name', '')} in={p.get('input_mean','?')},out={p.get('output_mean','?')})"

            print(f"\n  Phase {p_idx + 1}{phase_label}:")
            print(f"    Iterations: {end - start}")
            print(f"    New reqs: {phase_new_reqs}")
            if phase_k:
                print(f"    k_ratio: mean={np.mean(phase_k):.4f}, "
                      f"std={np.std(phase_k):.4f}, "
                      f"range=[{min(phase_k):.4f}, {max(phase_k):.4f}]")
            print(f"    Preemptions: {phase_preempt}")

            # Convergence analysis within this phase
            if phase_k and len(phase_k) > 100:
                final_val = np.mean(phase_k[-100:])
                conv_iter = None
                for ci in range(len(phase_k)):
                    if ci + 50 <= len(phase_k):
                        window = phase_k[ci:ci + 50]
                        if all(abs(k - final_val) / max(abs(final_val), 0.01) < 0.05
                               for k in window):
                            conv_iter = ci
                            break
                if conv_iter is not None:
                    # Count new reqs in convergence period
                    conv_new_reqs = sum(new_reqs[start:start + conv_iter])
                    print(f"    Convergence: {conv_iter} iters "
                          f"({conv_new_reqs} new reqs) to θ*={final_val:.4f}")
                else:
                    print(f"    Convergence: not reached (final θ*≈{final_val:.4f})")

        # Overall preemptions
        total_preempt = sum(s.get("num_preempted_reqs", 0) for s in stats)
        print(f"\n  Total preemptions: {total_preempt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot distribution shift experiment results."
    )
    parser.add_argument(
        "exp_dir", type=Path,
        help="Experiment output directory",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for figures (default: exp_dir parent / figures)",
    )
    parser.add_argument(
        "--window-sec", type=float, default=5.0,
        help="Sliding window size for throughput (seconds)",
    )
    args = parser.parse_args()

    configure_style()

    print(f"Loading: {args.exp_dir}")
    output_dir = args.output_dir or (args.exp_dir.parent / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print_summary(args.exp_dir)

    print("\n--- Generating figure ---")
    plot_distribution_shift(
        args.exp_dir,
        output_dir / "distribution_shift_convergence",
        window_sec=args.window_sec,
    )

    print(f"\nFigures saved to: {output_dir}")


if __name__ == "__main__":
    main()
