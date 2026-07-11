"""Embedding drift analysis (Algorithm 4, paper Appendix G).

For each family F spanning two or more years, and every ordered year pair
(y1, y2) with y2 > y1, and every sample pair (s_i in y1, s_j in y2)::

    r_ij = ||e_i - e_j||_2 / (y_j - y_i)

The minimum and maximum drift rate is recorded per family. High variance in
these rates across families indicates non-uniform evolution, which favours
Neighbor-Joining over UPGMA. This script also reproduces paper Figure 5.

Embeddings are L2-normalized before distances are computed so that drift
rates are comparable across modalities.

Usage:
    python embedding_drift.py --embeddings emb.json --timestamps ts.json \
        --families fam.json --output drift.json --figure figure5.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np


def load_year(timestamp) -> int | None:
    """Extract a 4-digit year from a timestamp string or {first_submission} dict."""
    if isinstance(timestamp, dict):
        timestamp = timestamp.get("first_submission") or timestamp.get("timestamp", "")
    if not timestamp:
        return None
    head = str(timestamp).strip().split("-")[0].split("/")[0]
    return int(head) if head.isdigit() and len(head) == 4 else None


def load_samples(embeddings_path: str, timestamps_path: str, families_path: str):
    """Load embeddings, years, and family labels into {sha: (vec, year, family)}."""
    with open(embeddings_path, encoding="utf-8") as fh:
        embeddings = json.load(fh)
    with open(timestamps_path, encoding="utf-8") as fh:
        timestamps = json.load(fh)
    with open(families_path, encoding="utf-8") as fh:
        families = json.load(fh)

    samples = {}
    for sha, value in embeddings.items():
        if isinstance(value, dict):
            vec = value.get("embedding") or value.get("embeddings")
        else:
            vec = value
        year = load_year(timestamps.get(sha))
        family = families.get(sha)
        if vec is not None and year is not None and family is not None:
            samples[sha] = (np.asarray(vec, dtype=np.float32), year, family)
    return samples


def compute_drift_rates(samples: dict, min_years: int = 2,
                        max_samples_per_year: int = 50, seed: int = 42) -> dict:
    """Algorithm 4: min/max pairwise drift rate per family (L2-normalized)."""
    rng = np.random.default_rng(seed)

    # Group L2-normalized embeddings by (family, year).
    bins: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for vec, year, family in samples.values():
        norm = np.linalg.norm(vec)
        bins[family][year].append(vec / norm if norm > 0 else vec)

    results = {}
    for family, years in bins.items():
        if len(years) < min_years:
            continue
        sorted_years = sorted(years)
        r_min, r_max = np.inf, -np.inf

        for i, y1 in enumerate(sorted_years[:-1]):
            for y2 in sorted_years[i + 1:]:
                v1, v2 = years[y1], years[y2]
                if len(v1) > max_samples_per_year:
                    v1 = [v1[k] for k in rng.choice(len(v1), max_samples_per_year, replace=False)]
                if len(v2) > max_samples_per_year:
                    v2 = [v2[k] for k in rng.choice(len(v2), max_samples_per_year, replace=False)]
                m1, m2 = np.stack(v1), np.stack(v2)
                # Pairwise Euclidean distances via the (a-b)^2 expansion.
                sq1 = np.sum(m1 ** 2, axis=1, keepdims=True)
                sq2 = np.sum(m2 ** 2, axis=1, keepdims=True)
                dists = np.sqrt(np.maximum(sq1 + sq2.T - 2.0 * m1 @ m2.T, 0.0))
                rates = dists / (y2 - y1)
                r_min = min(r_min, float(rates.min()))
                r_max = max(r_max, float(rates.max()))

        if np.isfinite(r_max):
            results[family] = {"min": r_min, "max": r_max,
                               "years": sorted_years, "n_samples": sum(len(v) for v in years.values())}
    return results


def make_figure5(results: dict, output_path: str, top_n: int = 10) -> None:
    """Reproduce paper Figure 5: per-family min/max drift rate (log scale)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = sorted(results.items(), key=lambda kv: kv[1]["max"], reverse=True)[:top_n]
    families = [name for name, _ in top]
    mins = [v["min"] for _, v in top]
    maxs = [v["max"] for _, v in top]
    y = np.arange(len(families))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y, maxs, color="#C0392B", alpha=0.85, label="Max")
    ax.barh(y, mins, color="#2980B9", alpha=0.85, label="Min")
    ax.set_yticks(y)
    ax.set_yticklabels(families, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Drift rate (L2-norm distance / year, log scale)")
    ax.set_title("Embedding drift varies substantially across families (Algorithm 4)")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"figure written to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", required=True, help="embeddings JSON (sha -> vector)")
    parser.add_argument("--timestamps", required=True, help="timestamps JSON (sha -> first_submission)")
    parser.add_argument("--families", required=True, help="family labels JSON (sha -> family)")
    parser.add_argument("--output", default="drift_analysis.json", help="output results JSON")
    parser.add_argument("--figure", help="optional path for the Figure 5 plot (.png)")
    parser.add_argument("--min-years", type=int, default=2, help="minimum distinct years per family")
    parser.add_argument("--max-samples", type=int, default=50, help="max samples per year bin")
    parser.add_argument("--seed", type=int, default=42, help="random seed for subsampling")
    args = parser.parse_args()

    print("loading samples ...", flush=True)
    samples = load_samples(args.embeddings, args.timestamps, args.families)
    print(f"loaded {len(samples)} samples", flush=True)

    results = compute_drift_rates(samples, args.min_years, args.max_samples, args.seed)
    print(f"computed drift rates for {len(results)} families", flush=True)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"results written to {args.output}")

    ranked = sorted(results.items(), key=lambda kv: kv[1]["max"], reverse=True)
    print(f"\n{'family':<28}{'min rate':>12}{'max rate':>12}")
    for name, value in ranked[:10]:
        print(f"{name:<28}{value['min']:>12.4f}{value['max']:>12.4f}")

    if args.figure:
        make_figure5(results, args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
