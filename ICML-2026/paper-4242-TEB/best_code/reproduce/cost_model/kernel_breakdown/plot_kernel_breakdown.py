"""
Plot kernel timing breakdown for §3 (kernel_breakdown_*.pdf in the paper).

Inputs (committed in this directory):
  results_h200.json
  results_a6000.json

Each file has the schema:
  {
    "gpu": "<GPU name>",
    "results": {
      "pure_prefill": [{seq_len, total_tokens, time_ms}, ...],
      "pure_decode":  [{context_len, total_query_tokens, time_ms}, ...],
      "mixed":        [{... varying schema ...}, ...],
    },
    "slopes": {...}    # pre-fit linear slopes
  }

Output:
  kernel_breakdown.pdf  — 2-panel figure (H200 | RTX 6000 / A6000),
                         each panel showing pure_prefill, pure_decode, mixed
                         attention time as function of token count.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_panel(ax, data: dict, title: str) -> None:
    res = data["results"]

    # pure prefill: time vs seq_len
    if res.get("pure_prefill"):
        xs = [r["seq_len"] for r in res["pure_prefill"]]
        ys = [r["time_ms"] for r in res["pure_prefill"]]
        ax.plot(xs, ys, marker="o", label="Pure prefill")

    # pure decode: time vs context_len  (fixed query count)
    if res.get("pure_decode"):
        xs = [r["context_len"] for r in res["pure_decode"]]
        ys = [r["time_ms"] for r in res["pure_decode"]]
        ax.plot(xs, ys, marker="s", label="Pure decode (B=512)")

    # mixed: dict keyed by decode percentage ("10pct", "20pct", ...), each a list of
    # {prefill_len, num_prefill, num_decode, total_query_tokens, time_ms}
    if isinstance(res.get("mixed"), dict):
        cmap = plt.get_cmap("viridis")
        keys = sorted(res["mixed"].keys(), key=lambda k: int(k.replace("pct", "")))
        for i, k in enumerate(keys):
            rows = res["mixed"][k]
            if not rows:
                continue
            xs = [r["prefill_len"] for r in rows]
            ys = [r["time_ms"] for r in rows]
            ax.plot(
                xs, ys,
                marker="^",
                linestyle="--",
                color=cmap(0.2 + 0.6 * i / max(1, len(keys) - 1)),
                label=f"Mixed ({k} decode)",
            )

    ax.set_xlabel("Tokens")
    ax.set_ylabel("Attention kernel time (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


def main() -> None:
    p = argparse.ArgumentParser()
    # Defaults assume CWD is this directory (reproduce/cost_model/kernel_breakdown/).
    script_dir = Path(__file__).parent
    p.add_argument(
        "--inputs",
        nargs="+",
        default=[
            str(script_dir / "results_h200.json"),
            str(script_dir / "results_a6000.json"),
        ],
        help="One or more attention benchmark JSON files.",
    )
    p.add_argument("--output", default="kernel_breakdown.pdf")
    args = p.parse_args()

    inputs = [Path(x) for x in args.inputs]
    inputs = [p for p in inputs if p.exists()]
    if not inputs:
        raise SystemExit("No input files found.")

    n = len(inputs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, path in zip(axes, inputs):
        data = load(path)
        gpu = data.get("gpu", path.stem)
        plot_panel(ax, data, gpu)

    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
