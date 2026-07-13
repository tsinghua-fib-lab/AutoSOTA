import argparse
import os
import random
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from env import Bandit
from VarDE import VarDE_lse, lse_decision_variance_gap_history


METRIC_KEYS = ("V_1st", "V_full", "abs_gap", "rel_gap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the higher-order gap in the LSE decision-variance approximation "
            "under the empirical Gaussian sampling distribution of the arm means."
        )
    )
    parser.add_argument(
        "--taus",
        type=str,
        default="0.05,0.1,0.15,0.2,0.5",
        help="Comma-separated list of tau values.",
    )
    parser.add_argument("--runs", type=int, default=10000, help="Number of Monte Carlo runs.")
    parser.add_argument("--T", type=int, default=1000, help="Total budget per run.")
    parser.add_argument(
        "--warm-start",
        type=int,
        default=2,
        help="Initial pulls per arm. Use at least 2 so empirical variances are finite.",
    )
    parser.add_argument(
        "--var-floor",
        type=float,
        default=0.1,
        help="Variance floor used by the agent's empirical trackers.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=2048,
        help="Samples used to estimate V_full at each measured checkpoint.",
    )
    parser.add_argument(
        "--measure-every",
        type=int,
        default=1,
        help="Measure the gap every N decision steps and always at the final step.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Base seed used for the environment, warm-start shuffling, and MC draws.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("results", "lse_variance"),
        help="Directory where plots and summaries are written.",
    )
    return parser.parse_args()


def parse_tau_list(raw: str) -> list[float]:
    taus = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(taus) == 0:
        raise ValueError("At least one tau value must be provided.")
    return taus


def default_problem_instance() -> tuple[np.ndarray, np.ndarray]:
    means = [0.3] + [0.22] * 3 + [0.2] * 3 + [0.15] * 3
    stds = [0.3] + [0.22] * 3 + [0.2] * 3 + [0.15] * 3
    random.Random(2).shuffle(stds)
    return np.asarray(means, dtype=float), np.asarray(stds, dtype=float)


def build_checkpoints(num_steps: int, measure_every: int) -> np.ndarray:
    if num_steps <= 0:
        raise ValueError("The experiment must have at least one decision step.")
    if measure_every <= 0:
        raise ValueError("measure_every must be positive.")
    checkpoints = np.arange(0, num_steps, measure_every, dtype=int)
    if checkpoints.size == 0 or checkpoints[-1] != num_steps - 1:
        checkpoints = np.append(checkpoints, num_steps - 1)
    return np.unique(checkpoints)


def run_single_trial(
    means: np.ndarray,
    stds: np.ndarray,
    tau: float,
    T: int,
    warm_start: int,
    var_floor: float,
    mc_samples: int,
    run_seed: int,
    checkpoints: np.ndarray,
) -> Dict[str, np.ndarray]:
    # Warm-start pulls happen during agent construction, so seed before instantiation.
    random.seed(run_seed)
    env = Bandit(distribution="gaussian", means=means, stds=stds, seed=run_seed)
    agent = VarDE_lse(
        env,
        T=T,
        warm_start=warm_start,
        tau=tau,
        var_floor=var_floor,
    )
    agent.run()

    means_hist = np.asarray(agent.means_history, dtype=float)[checkpoints]
    vars_hist = np.asarray(agent.vars_history, dtype=float)[checkpoints]
    counts_hist = np.asarray(agent.decision_n_history, dtype=float)[checkpoints]

    return lse_decision_variance_gap_history(
        means_history=means_hist,
        variances_history=vars_hist,
        counts_history=counts_hist,
        tau=tau,
        num_samples=mc_samples,
        seed=run_seed + 1_000_003,
    )


def aggregate_trials(
    means: np.ndarray,
    stds: np.ndarray,
    tau: float,
    args: argparse.Namespace,
    checkpoints: np.ndarray,
    tau_index: int,
) -> Dict[str, np.ndarray]:
    per_metric = {key: [] for key in METRIC_KEYS}
    for run_idx in tqdm(range(args.runs), desc=f"tau={tau:g}"):
        run_seed = args.seed + 10_000 * tau_index + run_idx
        metrics = run_single_trial(
            means=means,
            stds=stds,
            tau=tau,
            T=args.T,
            warm_start=args.warm_start,
            var_floor=args.var_floor,
            mc_samples=args.mc_samples,
            run_seed=run_seed,
            checkpoints=checkpoints,
        )
        for key in METRIC_KEYS:
            per_metric[key].append(metrics[key])

    summary = {}
    for key, runs in per_metric.items():
        stack = np.stack(runs, axis=0)
        summary[key] = np.nanmean(stack, axis=0)
        summary[f"{key}_std"] = np.nanstd(stack, axis=0)
    return summary


def plot_results(
    checkpoints: np.ndarray,
    results: Dict[float, Dict[str, np.ndarray]],
    output_path: str,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))

    for color, tau in zip(colors, sorted(results)):
        series = results[tau]
        axes[0].plot(checkpoints, series["V_full"], color=color, label=f"tau={tau:g} full")
        axes[0].plot(
            checkpoints,
            series["V_1st"],
            color=color,
            linestyle="--",
            label=f"tau={tau:g} 1st",
        )
        axes[1].plot(checkpoints, series["abs_gap"], color=color, label=f"tau={tau:g}")
        axes[2].plot(checkpoints, series["rel_gap"], color=color, label=f"tau={tau:g}")

    axes[0].set_title("Decision Variance: First-Order vs Full LSE")
    axes[0].set_ylabel("Variance")
    axes[1].set_title("Higher-Order Contribution")
    axes[1].set_ylabel("abs_gap")
    axes[2].set_title("Relative Higher-Order Contribution")
    axes[2].set_ylabel("rel_gap")
    axes[2].set_xlabel("Decision step after warm start")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_summary(
    checkpoints: np.ndarray,
    results: Dict[float, Dict[str, np.ndarray]],
    output_path: str,
) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for tau in sorted(results):
            series = results[tau]
            final_idx = -1
            max_abs_idx = int(np.nanargmax(series["abs_gap"]))
            max_rel_idx = int(np.nanargmax(series["rel_gap"]))
            handle.write(f"tau={tau:g}\n")
            handle.write(f"  final_step={int(checkpoints[final_idx])}\n")
            handle.write(f"  final_V_1st={series['V_1st'][final_idx]:.8e}\n")
            handle.write(f"  final_V_full={series['V_full'][final_idx]:.8e}\n")
            handle.write(f"  final_abs_gap={series['abs_gap'][final_idx]:.8e}\n")
            handle.write(f"  final_rel_gap={series['rel_gap'][final_idx]:.8e}\n")
            handle.write(
                f"  max_abs_gap={series['abs_gap'][max_abs_idx]:.8e} at_step={int(checkpoints[max_abs_idx])}\n"
            )
            handle.write(
                f"  max_rel_gap={series['rel_gap'][max_rel_idx]:.8e} at_step={int(checkpoints[max_rel_idx])}\n"
            )
            handle.write("\n")


def save_arrays(
    checkpoints: np.ndarray,
    results: Dict[float, Dict[str, np.ndarray]],
    output_path: str,
) -> None:
    arrays = {"checkpoints": checkpoints}
    for tau in sorted(results):
        tag = str(tau).replace(".", "p")
        for key, value in results[tau].items():
            arrays[f"tau_{tag}_{key}"] = value
    np.savez(output_path, **arrays)


def main() -> None:
    args = parse_args()
    taus = parse_tau_list(args.taus)
    means, stds = default_problem_instance()

    if args.warm_start < 2:
        raise ValueError("warm_start must be at least 2 to obtain finite empirical variances.")
    if args.warm_start * means.size >= args.T:
        raise ValueError("T must be larger than warm_start * K.")

    checkpoints = build_checkpoints(
        num_steps=args.T - args.warm_start * means.size,
        measure_every=args.measure_every,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    for tau_index, tau in enumerate(taus):
        results[tau] = aggregate_trials(
            means=means,
            stds=stds,
            tau=tau,
            args=args,
            checkpoints=checkpoints,
            tau_index=tau_index,
        )

    plot_results(
        checkpoints=checkpoints,
        results=results,
        output_path=os.path.join(args.output_dir, "lse_variance_overview.png"),
    )
    save_summary(
        checkpoints=checkpoints,
        results=results,
        output_path=os.path.join(args.output_dir, "lse_variance_summary.txt"),
    )
    save_arrays(
        checkpoints=checkpoints,
        results=results,
        output_path=os.path.join(args.output_dir, "lse_variance_metrics.npz"),
    )


if __name__ == "__main__":
    main()
