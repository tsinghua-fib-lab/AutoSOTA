#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

# Use a non-interactive backend so this works on headless machines/CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_k(k_key: str) -> int:
    """
    Parse keys like 'k-10' -> 10.
    """
    m = re.fullmatch(r"k-(\d+)", k_key)
    if not m:
        raise ValueError(f"Unexpected k key: {k_key!r} (expected 'k-<int>')")
    return int(m.group(1))


def sanitize_filename(s: str) -> str:
    # Keep it stable and filesystem-friendly
    s = s.replace("/", "_").replace("\\", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "plot"


def algo_label(algo_key: str) -> str:
    mapping = {
        "gb": "Guillory Bilmes",
        "cb": "Cesa-Bianchi et al.",
        "ca": "Cohen-Addad et al.",
    }
    if algo_key in mapping:
        return mapping[algo_key]
    if algo_key.startswith("ours-"):
        return "Ours" if algo_key == "ours" else f"Ours ({algo_key[len('ours-'):]})"
    return algo_key


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualize experiments from results JSON as per-(timestamp,dataset) PDFs."
    )
    ap.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        default=Path("results.json"),
        help="Input JSON path (default: results.json).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots/"),
        help="Output directory for PDFs (default: plots/).",
    )
    args = ap.parse_args()

    json_path: Path = args.json_path
    if not json_path.is_file():
        raise SystemExit(f"JSON file not found: {json_path}")

    out_dir: Path = args.out_dir if args.out_dir is not None else json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(json_path)

    # ICML-ish readability: larger fonts, vector PDF output
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 1.8,
            "lines.markersize": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Expected structure:
    # data[timestamp][dataset][algo][k-<int>] = [itr0, itr1, ...]
    for timestamp, datasets in data.items():
        if not isinstance(datasets, dict):
            continue

        for dataset_name, algos in datasets.items():
            if not isinstance(algos, dict):
                continue

            fig, ax = plt.subplots(figsize=(6.4, 4.0))  # good default; scale down in 2-col papers

            for algo_key, ks in algos.items():
                if not isinstance(ks, dict):
                    continue

                xs: list[int] = []
                means: list[float] = []
                stds: list[float] = []
                nvals: list[int] = []

                for k_key, itr_list in ks.items():
                    if not isinstance(itr_list, list):
                        continue

                    vals = [v for v in itr_list if v is not None]
                    if len(vals) == 0:
                        continue

                    k = parse_k(k_key)
                    xs.append(k)
                    vals_arr = np.asarray(vals, dtype=float)
                    means.append(float(np.mean(vals_arr)))
                    nvals.append(int(vals_arr.size))
                    if vals_arr.size > 1:
                        stds.append(float(np.std(vals_arr, ddof=1)))
                    else:
                        stds.append(0.0)

                if not xs:
                    continue

                order = np.argsort(xs)
                xs_arr = np.asarray(xs, dtype=int)[order]
                means_arr = np.asarray(means, dtype=float)[order]
                stds_arr = np.asarray(stds, dtype=float)[order]
                n_arr = np.asarray(nvals, dtype=int)[order]

                label = algo_label(algo_key)

                (line,) = ax.plot(xs_arr, means_arr, marker="o", label=label)
                c = line.get_color()

                mask = n_arr > 1
                if np.any(mask):
                    lower = means_arr - stds_arr
                    upper = means_arr + stds_arr
                    ax.fill_between(
                        xs_arr,
                        lower,
                        upper,
                        where=mask,
                        interpolate=True,
                        color=c,          # <-- match line color
                        alpha=0.2,
                        linewidth=0,
                        zorder=line.get_zorder() - 1,  # keep band behind the line
                    )


            ax.set_xlabel("k")
            ax.set_ylabel(r"$\Psi(L)$")
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
            ax.legend(frameon=True)
            fig.tight_layout()

            out_name = f"{timestamp}-{sanitize_filename(str(dataset_name))}.pdf"
            out_path = out_dir / out_name
            fig.savefig(out_path, format="pdf", bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
