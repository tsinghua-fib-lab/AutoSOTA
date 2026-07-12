"""
Plot execution_time figures for §3 (execution_time.pdf / execution_time_gpus.pdf /
prefill_linearity_all_models.png).

Demonstrates the linear iteration-time model: T_iter = alpha + beta_p * L_p + beta_d * N_d.

Input:
  experiments/iteration_time/results.json (and results2/3/4*.json variants)

Schema (per file):
  {
    "config": {
      "model": "...",
      "prefill_chunk_sizes": [...],
      "decode_counts": [...],
      "pure_prefill_sizes": [...],
      "pure_decode_counts": [...],
      ...
    },
    "results": [
      {"description", "num_decode", "num_prefill", "prefill_chunk_size",
       "decode_context_len", "total_tokens", "mean_time_ms", "std_time_ms",
       "throughput_tokens_per_sec"},
      ...
    ]
  }

Output:
  execution_time.pdf — 2-panel figure:
    (left)  Prefill: mean_time_ms vs num_prefill   — should be linear (T_p ~ alpha_p + beta_p * L)
    (right) Decode:  mean_time_ms vs num_decode    — should be linear (T_d ~ alpha_d + beta_d * N)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def split_results(records: list[dict]) -> tuple[list, list, list]:
    """Split into pure_prefill, pure_decode, mixed lists."""
    pure_p, pure_d, mixed = [], [], []
    for r in records:
        np_ = r.get("num_prefill", 0)
        nd = r.get("num_decode", 0)
        if np_ > 0 and nd == 0:
            pure_p.append(r)
        elif np_ == 0 and nd > 0:
            pure_d.append(r)
        else:
            mixed.append(r)
    return pure_p, pure_d, mixed


def plot(records: list[dict], model: str, output: Path) -> None:
    pure_p, pure_d, _mixed = split_results(records)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    if pure_p:
        # For pure prefill measurements, num_prefill is typically 1 (a single request)
        # and prefill_chunk_size varies — that's the "prefill length L" we want as x.
        xs = np.array([r.get("prefill_chunk_size") or r["total_tokens"] for r in pure_p])
        ys = np.array([r["mean_time_ms"] for r in pure_p])
        err = np.array([r["std_time_ms"] for r in pure_p])
        # Sort by x for clean line
        order = np.argsort(xs)
        xs, ys, err = xs[order], ys[order], err[order]
        ax1.errorbar(xs, ys, yerr=err, fmt="o", capsize=3, label="Measured")
        # Linear fit
        if len(xs) >= 2:
            slope, intercept = np.polyfit(xs, ys, 1)
            xfit = np.array([xs.min(), xs.max()])
            ax1.plot(
                xfit,
                slope * xfit + intercept,
                linestyle="--",
                label=f"Fit: T = {intercept:.2f} + {slope:.5f} L",
            )
        ax1.set_xlabel("Prefill length L (tokens)")
        ax1.set_ylabel("Iteration time (ms)")
        ax1.set_title(f"Prefill linearity — {model}")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    if pure_d:
        xs = np.array([r["num_decode"] for r in pure_d])
        ys = np.array([r["mean_time_ms"] for r in pure_d])
        err = np.array([r["std_time_ms"] for r in pure_d])
        order = np.argsort(xs)
        xs, ys, err = xs[order], ys[order], err[order]
        ax2.errorbar(xs, ys, yerr=err, fmt="s", capsize=3, color="C1", label="Measured")
        if len(xs) >= 2:
            slope, intercept = np.polyfit(xs, ys, 1)
            xfit = np.array([xs.min(), xs.max()])
            ax2.plot(
                xfit,
                slope * xfit + intercept,
                linestyle="--",
                color="C1",
                label=f"Fit: T = {intercept:.2f} + {slope:.5f} N",
            )
        ax2.set_xlabel("Decode batch size N")
        ax2.set_ylabel("Iteration time (ms)")
        ax2.set_title(f"Decode linearity — {model}")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(f"Wrote {output}")


def main() -> None:
    p = argparse.ArgumentParser()
    # Default assumes CWD is this directory (reproduce/cost_model/linear_model/).
    script_dir = Path(__file__).parent
    p.add_argument(
        "--input",
        default=str(script_dir / "results.json"),
        help="iteration_time results JSON",
    )
    p.add_argument("--output", default="execution_time.pdf")
    args = p.parse_args()

    data = load(Path(args.input))
    model = data.get("config", {}).get("model", "<unknown>")
    plot(data["results"], model, Path(args.output))


if __name__ == "__main__":
    main()
