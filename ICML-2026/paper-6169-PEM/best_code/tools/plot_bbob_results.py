#!/usr/bin/env python3
"""Generate plots and summary metrics from BBOB benchmark results."""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from _project import repo_relpath

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "bbob_summary.csv")
TRACE_INDEX_PATH = os.path.join(RESULTS_DIR, "trace_index.csv")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TRACES_PLOTS_DIR = os.path.join(PLOTS_DIR, "traces")


def read_summary():
    rows = []
    with open(SUMMARY_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["budget_multiplier"] = int(row["budget_multiplier"])
            row["function"] = int(row["function"])
            row["dimension"] = int(row["dimension"])
            row["instance"] = int(row["instance"])
            row["evaluations"] = int(row["evaluations"])
            row["best_f"] = float(row["best_f"])
            row["final_target_hit"] = int(row["final_target_hit"])
            row["elapsed_sec"] = float(row["elapsed_sec"])
            rows.append(row)
    return rows


def compute_rankings(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["budget_multiplier"],
            row["function"],
            row["dimension"],
            row["instance"],
        )
        grouped[key].append(row)

    ranks = []
    for key, items in grouped.items():
        items_sorted = sorted(items, key=lambda r: r["best_f"])
        for idx, row in enumerate(items_sorted, start=1):
            ranks.append(
                {
                    "algorithm": row["algorithm"],
                    "budget_multiplier": row["budget_multiplier"],
                    "rank": idx,
                    "best_f": row["best_f"],
                    "function": row["function"],
                    "dimension": row["dimension"],
                    "instance": row["instance"],
                }
            )
    return ranks


def summarize_metrics(rows, ranks):
    metrics = defaultdict(lambda: defaultdict(list))

    for row in rows:
        algo = row["algorithm"]
        budget = row["budget_multiplier"]
        metrics[algo]["hit"].append(row["final_target_hit"])
        metrics[algo]["budget_hit"].append((budget, row["final_target_hit"]))

    rank_stats = defaultdict(list)
    rank_stats_budget = defaultdict(list)

    for r in ranks:
        rank_stats[r["algorithm"]].append(r["rank"])
        rank_stats_budget[(r["algorithm"], r["budget_multiplier"])].append(r["rank"])

    win_counts = defaultdict(int)
    win_counts_budget = defaultdict(int)
    grouped = defaultdict(list)
    for r in ranks:
        key = (r["budget_multiplier"], r["function"], r["dimension"], r["instance"])
        grouped[key].append(r)

    for key, items in grouped.items():
        best = min(items, key=lambda x: x["best_f"])
        win_counts[best["algorithm"]] += 1
        win_counts_budget[(best["algorithm"], best["budget_multiplier"])] += 1

    return metrics, rank_stats, rank_stats_budget, win_counts, win_counts_budget


def write_metrics_csv(rows, ranks, metrics, rank_stats, rank_stats_budget, win_counts, win_counts_budget):
    output_path = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    budgets = sorted({row["budget_multiplier"] for row in rows})

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "algorithm",
                "avg_rank",
                "hit_rate",
                "win_count",
                "total_problems",
            ]
        )
        total_problems = len(
            {
                (r["budget_multiplier"], r["function"], r["dimension"], r["instance"])
                for r in ranks
            }
        )
        for algo in sorted(rank_stats.keys()):
            avg_rank = sum(rank_stats[algo]) / len(rank_stats[algo])
            hit_rate = sum(metrics[algo]["hit"]) / len(metrics[algo]["hit"]) if metrics[algo]["hit"] else 0.0
            writer.writerow(
                [
                    algo,
                    f"{avg_rank:.3f}",
                    f"{hit_rate:.3f}",
                    win_counts.get(algo, 0),
                    total_problems,
                ]
            )

        writer.writerow([])
        writer.writerow(["algorithm", "budget", "avg_rank", "hit_rate", "win_count", "total_budget_problems"])
        for algo in sorted(rank_stats.keys()):
            for budget in budgets:
                rank_list = rank_stats_budget.get((algo, budget), [])
                avg_rank = sum(rank_list) / len(rank_list) if rank_list else 0.0
                hit_list = [hit for b, hit in metrics[algo]["budget_hit"] if b == budget]
                hit_rate = sum(hit_list) / len(hit_list) if hit_list else 0.0
                win_count = win_counts_budget.get((algo, budget), 0)
                total_budget_problems = len(
                    {
                        (r["function"], r["dimension"], r["instance"])
                        for r in ranks
                        if r["budget_multiplier"] == budget
                    }
                )
                writer.writerow(
                    [
                        algo,
                        budget,
                        f"{avg_rank:.3f}",
                        f"{hit_rate:.3f}",
                        win_count,
                        total_budget_problems,
                    ]
                )

    return output_path


def plot_bar(values, labels, title, ylabel, output_path):
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color="#2b6cb0")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_grouped_bars(categories, series, series_labels, title, ylabel, output_path):
    width = 0.8 / len(series_labels)
    x = range(len(categories))
    plt.figure(figsize=(9, 4.5))
    for idx, (label, values) in enumerate(zip(series_labels, series)):
        offsets = [i + idx * width for i in x]
        plt.bar(offsets, values, width=width, label=label)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks([i + width * (len(series_labels) - 1) / 2 for i in x], categories)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_traces(trace_index_rows):
    os.makedirs(TRACES_PLOTS_DIR, exist_ok=True)
    grouped = defaultdict(list)
    for row in trace_index_rows:
        key = (row["budget_multiplier"], row["function"], row["dimension"])
        grouped[key].append(row)

    for key, items in grouped.items():
        budget, func, dim = key
        plt.figure(figsize=(7, 4.5))
        for item in items:
            with open(item["trace_file"], newline="") as f:
                reader = csv.DictReader(f)
                evals = []
                bests = []
                for row in reader:
                    evals.append(int(row["evals"]))
                    if "best_f" in row:
                        bests.append(float(row["best_f"]))
                    elif "best_true_f" in row:
                        bests.append(float(row["best_true_f"]))
                    else:
                        raise KeyError("Trace CSV must contain 'best_f' or 'best_true_f'")
            plt.plot(evals, bests, label=item["algorithm"])

        plt.title(f"Best-f vs Evaluations | f{func} D={dim} B={budget}x")
        plt.xlabel("Evaluations")
        plt.ylabel("Best f (lower is better)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        filename = f"trace_f{func}_d{dim}_b{budget}.png"
        plt.savefig(os.path.join(TRACES_PLOTS_DIR, filename), dpi=200)
        plt.close()


def read_trace_index():
    rows = []
    with open(TRACE_INDEX_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["budget_multiplier"] = int(row["budget_multiplier"])
            row["function"] = int(row["function"])
            row["dimension"] = int(row["dimension"])
            row["instance"] = int(row["instance"])
            rows.append(row)
    return rows


def main():
    global RESULTS_DIR, SUMMARY_PATH, TRACE_INDEX_PATH, PLOTS_DIR, TRACES_PLOTS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="Directory containing bbob_summary.csv and trace_index.csv",
    )
    args = parser.parse_args()

    RESULTS_DIR = args.results_dir
    SUMMARY_PATH = os.path.join(RESULTS_DIR, "bbob_summary.csv")
    TRACE_INDEX_PATH = os.path.join(RESULTS_DIR, "trace_index.csv")
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
    TRACES_PLOTS_DIR = os.path.join(PLOTS_DIR, "traces")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    rows = read_summary()
    ranks = compute_rankings(rows)
    metrics, rank_stats, rank_stats_budget, win_counts, win_counts_budget = summarize_metrics(
        rows, ranks
    )
    metrics_path = write_metrics_csv(
        rows, ranks, metrics, rank_stats, rank_stats_budget, win_counts, win_counts_budget
    )

    algorithms = sorted(rank_stats.keys())
    avg_ranks = [sum(rank_stats[a]) / len(rank_stats[a]) for a in algorithms]
    hit_rates = [
        sum(metrics[a]["hit"]) / len(metrics[a]["hit"]) if metrics[a]["hit"] else 0.0
        for a in algorithms
    ]
    wins = [win_counts.get(a, 0) for a in algorithms]

    plot_bar(avg_ranks, algorithms, "Average Rank (lower is better)", "Avg Rank", os.path.join(PLOTS_DIR, "avg_rank.png"))
    plot_bar(hit_rates, algorithms, "Final Target Hit Rate", "Hit Rate", os.path.join(PLOTS_DIR, "hit_rate.png"))
    plot_bar(wins, algorithms, "Win Count (best f per problem)", "Wins", os.path.join(PLOTS_DIR, "win_count.png"))

    budgets = sorted({row["budget_multiplier"] for row in rows})
    avg_rank_by_budget = []
    hit_rate_by_budget = []

    for algo in algorithms:
        ranks_for_algo = []
        hits_for_algo = []
        for budget in budgets:
            rank_list = rank_stats_budget.get((algo, budget), [])
            avg_rank = sum(rank_list) / len(rank_list) if rank_list else 0.0
            ranks_for_algo.append(avg_rank)

            hit_list = [hit for b, hit in metrics[algo]["budget_hit"] if b == budget]
            hit_rate = sum(hit_list) / len(hit_list) if hit_list else 0.0
            hits_for_algo.append(hit_rate)

        avg_rank_by_budget.append(ranks_for_algo)
        hit_rate_by_budget.append(hits_for_algo)

    plot_grouped_bars(
        [str(b) for b in budgets],
        avg_rank_by_budget,
        algorithms,
        "Average Rank by Budget",
        "Avg Rank",
        os.path.join(PLOTS_DIR, "avg_rank_by_budget.png"),
    )

    plot_grouped_bars(
        [str(b) for b in budgets],
        hit_rate_by_budget,
        algorithms,
        "Hit Rate by Budget",
        "Hit Rate",
        os.path.join(PLOTS_DIR, "hit_rate_by_budget.png"),
    )

    trace_rows = read_trace_index()
    if trace_rows:
        plot_traces(trace_rows)

    print("Plots saved to", repo_relpath(PLOTS_DIR))
    print("Metrics saved to", repo_relpath(metrics_path))


if __name__ == "__main__":
    main()
