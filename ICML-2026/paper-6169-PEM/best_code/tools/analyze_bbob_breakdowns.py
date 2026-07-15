#!/usr/bin/env python3
"""
Compute simple breakdown metrics (avg rank / win count) by dimension and function
from a `bbob_summary.csv` produced by the local runners.

This is meant for *exploration* and lightweight ablations: it does not compute
COCO's official ERT/ECDF metrics.
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from _project import repo_relpath

def read_summary(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "algorithm": row["algorithm"],
                    "budget_multiplier": int(row["budget_multiplier"]),
                    "function": int(row["function"]),
                    "dimension": int(row["dimension"]),
                    "instance": int(row["instance"]),
                    "best_f": float(row["best_f"]),
                    "final_target_hit": int(row.get("final_target_hit", "0")),
                }
            )
    return rows


def compute_ranks(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["budget_multiplier"], r["function"], r["dimension"], r["instance"])
        grouped[key].append(r)

    ranks: list[dict] = []
    for key, items in grouped.items():
        items_sorted = sorted(items, key=lambda x: x["best_f"])
        for i, item in enumerate(items_sorted, start=1):
            ranks.append(
                {
                    "algorithm": item["algorithm"],
                    "budget_multiplier": item["budget_multiplier"],
                    "function": item["function"],
                    "dimension": item["dimension"],
                    "instance": item["instance"],
                    "rank": i,
                }
            )
    return ranks


def summarize_by_key(ranks: list[dict], key_name: str) -> dict[tuple, dict]:
    by_key: dict[tuple, list[int]] = defaultdict(list)
    win_counts: dict[tuple, int] = defaultdict(int)
    total_counts: dict[tuple, int] = defaultdict(int)

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in ranks:
        problem_key = (r["budget_multiplier"], r["function"], r["dimension"], r["instance"])
        grouped[problem_key].append(r)

    for problem_key, items in grouped.items():
        # Mark winner for this problem group.
        best = min(items, key=lambda x: x["rank"])
        group_value = best[key_name]
        win_counts[(group_value, best["algorithm"])] += 1

    for r in ranks:
        group_value = r[key_name]
        by_key[(group_value, r["algorithm"])].append(int(r["rank"]))
        total_counts[(group_value, r["algorithm"])] += 1

    out = {}
    for (group_value, algo), rank_list in by_key.items():
        out[(group_value, algo)] = {
            key_name: group_value,
            "algorithm": algo,
            "avg_rank": sum(rank_list) / max(1, len(rank_list)),
            "win_count": int(win_counts.get((group_value, algo), 0)),
            "total_problems": int(total_counts.get((group_value, algo), 0)),
        }
    return out


def write_csv(rows: list[dict], path: str, key_name: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[key_name, "algorithm", "avg_rank", "win_count", "total_problems", "win_rate"]
        )
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["win_rate"] = (
                float(out["win_count"]) / float(out["total_problems"]) if out["total_problems"] else 0.0
            )
            writer.writerow(out)


def plot_grouped_bars(*, groups: list[int], series_by_algo: dict[str, list[float]], title: str, ylabel: str, out_path: str):
    algos = sorted(series_by_algo.keys())
    width = 0.8 / max(1, len(algos))
    x = list(range(len(groups)))
    plt.figure(figsize=(9.0, 4.6))
    for i, algo in enumerate(algos):
        values = series_by_algo[algo]
        offsets = [j + i * width for j in x]
        plt.bar(offsets, values, width=width, label=algo)
    plt.xticks([j + width * (len(algos) - 1) / 2 for j in x], [str(g) for g in groups])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Directory containing bbob_summary.csv")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    summary_path = os.path.join(results_dir, "bbob_summary.csv")
    if not os.path.isfile(summary_path):
        raise SystemExit(f"Missing: {summary_path}")

    rows = read_summary(summary_path)
    ranks = compute_ranks(rows)

    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for key_name, fname_prefix in [("dimension", "by_dimension"), ("function", "by_function")]:
        summary = summarize_by_key(ranks, key_name)
        out_rows = [v for _, v in sorted(summary.items(), key=lambda kv: (kv[1][key_name], kv[1]["algorithm"]))]
        out_csv = os.path.join(results_dir, f"{fname_prefix}.csv")
        write_csv(out_rows, out_csv, key_name)

        groups = sorted({int(r[key_name]) for r in out_rows})
        series_rank = defaultdict(list)
        series_win = defaultdict(list)
        for g in groups:
            per_g = [r for r in out_rows if int(r[key_name]) == g]
            algo_to_row = {r["algorithm"]: r for r in per_g}
            for algo, r in algo_to_row.items():
                series_rank[algo].append(float(r["avg_rank"]))
                series_win[algo].append(float(r["win_count"]) / max(1.0, float(r["total_problems"])))

        plot_grouped_bars(
            groups=groups,
            series_by_algo=series_rank,
            title=f"Average Rank by {key_name} (lower is better)",
            ylabel="avg_rank",
            out_path=os.path.join(plots_dir, f"{fname_prefix}_avg_rank.png"),
        )
        plot_grouped_bars(
            groups=groups,
            series_by_algo=series_win,
            title=f"Win Rate by {key_name} (higher is better)",
            ylabel="win_rate",
            out_path=os.path.join(plots_dir, f"{fname_prefix}_win_rate.png"),
        )

        print("Wrote:", repo_relpath(out_csv))


if __name__ == "__main__":
    main()
