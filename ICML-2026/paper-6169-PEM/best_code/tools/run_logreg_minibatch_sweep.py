#!/usr/bin/env python3
"""
External validity benchmark (non-COCO):

Mini-batch logistic regression as a stochastic black-box objective.

Why this exists (external validity):
- COCO bbob-noisy is a valuable benchmark, but it is important to check whether the
  proposed selection-stage robustness component transfers beyond synthetic test suites.
- Empirical risk minimization with mini-batch evaluation is a simple, controlled
  instance of stochastic objectives where *misranking* is directly tunable via
  batch size and tail-heaviness.

Protocol:
- Generate a fixed synthetic dataset (X, y) per seed.
- Optimize w in a box constraint to minimize empirical logistic loss.
- Noisy evaluation uses random mini-batches (and optional lognormal reweighting).
- Report the returned best_x evaluated on the full dataset (noise-free metric).

Outputs:
  Results/application_logreg_minibatch_sweep/
    batch_8/
      runs.csv
      summary.csv
      pairwise_sign_test_post_true.csv
      final_boxplot.png
    ...
    sweep_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from _project import BASE_DIR, repo_relpath

from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero as berw_hetero_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t012 as probeswitch_mr_t012_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_warmstart_t012 as probeswitch_mr_warmstart_t012_optimizer,
)
from baselines.cmaes_full import my_optimizer as cmaes_full_optimizer
from baselines.cmaes_sep import my_optimizer as cmaes_sep_optimizer


def parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i = int(a.strip())
            b_i = int(b.strip())
            if b_i < a_i:
                a_i, b_i = b_i, a_i
            out.extend(range(a_i, b_i + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def safe_dir_token(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip())
    s = s.strip("_")
    return s or "unnamed"


def logistic_loss(z: np.ndarray, y01: np.ndarray) -> np.ndarray:
    # stable: log(1+exp(z)) - y*z
    return np.logaddexp(0.0, z) - y01 * z


def _load_base_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = str(dataset).strip().lower()
    if dataset in {"breast_cancer", "breast-cancer", "cancer"}:
        from sklearn.datasets import load_breast_cancer  # type: ignore

        data = load_breast_cancer()
        X = np.asarray(data.data, dtype=float)
        y = np.asarray(data.target, dtype=float)
        return X, y
    if dataset in {"digits0", "digits_0", "digits-zero"}:
        from sklearn.datasets import load_digits  # type: ignore

        data = load_digits()
        X = np.asarray(data.data, dtype=float) / 16.0
        y = (np.asarray(data.target, dtype=int) == 0).astype(float)
        return X, y
    raise ValueError(f"Unknown dataset: {dataset!r}")


class NoisyMiniBatchLogRegProblem:
    def __init__(
        self,
        *,
        seed: int,
        dim: int,
        n_samples: int,
        batch_size: int,
        w_max: float,
        weight_sigma: float,
        weight_sigma_stochastic_only: bool = False,
        l2_reg: float,
        label_noise: float,
        eval_independent_noise: bool,
        dataset: str = "synthetic",
        add_bias: bool = True,
        standardize: bool = True,
        id_function: int = 9101,
    ):
        self.id_function = int(id_function)
        self.id_instance = int(seed)

        dataset = str(dataset).strip().lower()

        self._base_seed = int(seed) & 0xFFFFFFFF
        self._eval_independent_noise = bool(eval_independent_noise)
        self._rng = np.random.RandomState(int(self._base_seed) ^ 0xC0FFEE)

        self._batch_size = int(max(1, batch_size))
        self._weight_sigma = float(max(0.0, weight_sigma))
        self._weight_sigma_stochastic_only = bool(weight_sigma_stochastic_only)
        self._l2_reg = float(max(0.0, l2_reg))

        rng_data = np.random.RandomState(int(self._base_seed) ^ 0xA5A5A5A5)

        # dataset (per-seed instance)
        n_samples = int(max(16, n_samples))
        if dataset == "synthetic":
            self.dimension = int(dim)
            X = rng_data.randn(n_samples, self.dimension).astype(float, copy=False)
            w_true = rng_data.randn(self.dimension).astype(float, copy=False)
            w_true = w_true / max(1e-12, float(np.linalg.norm(w_true)))
            w_true = 2.0 * w_true
            p = 1.0 / (1.0 + np.exp(-(X @ w_true)))
            y = (rng_data.rand(n_samples) < p).astype(float, copy=False)
        else:
            X0, y0 = _load_base_dataset(dataset)
            idx = rng_data.randint(0, int(X0.shape[0]), size=n_samples)
            X = np.asarray(X0[idx], dtype=float)
            y = np.asarray(y0[idx], dtype=float)
            if bool(standardize):
                mu = np.mean(X, axis=0)
                sd = np.std(X, axis=0)
                sd = np.where(sd < 1e-12, 1.0, sd)
                X = (X - mu) / sd
            if bool(add_bias):
                X = np.concatenate([X, np.ones((n_samples, 1), dtype=float)], axis=1)
            self.dimension = int(X.shape[1])

        if float(label_noise) > 0.0:
            flip = rng_data.rand(n_samples) < float(label_noise)
            y = np.where(flip, 1.0 - y, y)

        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)

        self.lower_bounds = -float(w_max) * np.ones(self.dimension, dtype=float)
        self.upper_bounds = float(w_max) * np.ones(self.dimension, dtype=float)
        self.initial_solution = np.zeros(self.dimension, dtype=float)

        self.evaluations = 0
        self.final_target_hit = False
        self.best_observed_fvalue1 = float("inf")
        self.best_x: np.ndarray | None = None

    def _rng_eval(self, eval_id: int) -> np.random.RandomState:
        if not self._eval_independent_noise:
            return self._rng
        seed_eval = (
            int(self._base_seed) * 1000003
            + int(eval_id) * 9176
            + 2026
        ) & 0xFFFFFFFF
        return np.random.RandomState(int(seed_eval))

    def _loss_full(self, w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        z = self.X @ w
        loss = float(np.mean(logistic_loss(z, self.y)))
        if self._l2_reg > 0.0:
            loss += 0.5 * float(self._l2_reg) * float(np.dot(w, w))
        return float(loss)

    def true_loss(self, w: np.ndarray) -> float:
        w = np.clip(np.asarray(w, dtype=float), self.lower_bounds, self.upper_bounds)
        return self._loss_full(w)

    def __call__(self, w: np.ndarray) -> float:
        self.evaluations += 1
        eval_id = int(self.evaluations)

        w = np.clip(np.asarray(w, dtype=float), self.lower_bounds, self.upper_bounds)
        rng = self._rng_eval(eval_id)

        n = int(self.X.shape[0])
        bs = int(min(self._batch_size, n))
        if bs >= n:
            idx = np.arange(n, dtype=int)
        else:
            idx = rng.randint(0, n, size=bs)
        Xb = self.X[idx]
        yb = self.y[idx]
        z = Xb @ w
        per = logistic_loss(z, yb)

        apply_weights = bool(self._weight_sigma > 0.0)
        if apply_weights and self._weight_sigma_stochastic_only and bs >= n:
            apply_weights = False

        if apply_weights:
            s = float(self._weight_sigma)
            # mean-1 lognormal weights -> unbiased but heavy-tailed estimator
            wts = np.exp(s * rng.randn(bs) - 0.5 * s * s)
            val = float(np.mean(wts * per))
        else:
            val = float(np.mean(per))

        if self._l2_reg > 0.0:
            val += 0.5 * float(self._l2_reg) * float(np.dot(w, w))

        if val < self.best_observed_fvalue1:
            self.best_observed_fvalue1 = float(val)
            self.best_x = w.copy()
        return float(val)


@dataclass(frozen=True)
class RunResult:
    algorithm: str
    seed: int
    batch_size: int
    evaluations: int
    best_noisy: float
    post_true: float


def run_one(
    *,
    algorithm: str,
    optimizer,
    seed: int,
    batch_size: int,
    dim: int,
    n_samples: int,
    max_evals: int,
    w_max: float,
    weight_sigma: float,
    weight_sigma_stochastic_only: bool,
    l2_reg: float,
    label_noise: float,
    eval_independent_noise: bool,
    dataset: str,
    add_bias: bool,
    standardize: bool,
) -> RunResult:
    problem = NoisyMiniBatchLogRegProblem(
        seed=int(seed),
        dim=int(dim),
        n_samples=int(n_samples),
        batch_size=int(batch_size),
        w_max=float(w_max),
        weight_sigma=float(weight_sigma),
        weight_sigma_stochastic_only=bool(weight_sigma_stochastic_only),
        l2_reg=float(l2_reg),
        label_noise=float(label_noise),
        eval_independent_noise=bool(eval_independent_noise),
        dataset=str(dataset),
        add_bias=bool(add_bias),
        standardize=bool(standardize),
    )
    optimizer(problem, int(max_evals))
    x_best = problem.best_x if problem.best_x is not None else np.asarray(problem.initial_solution, dtype=float)
    post_true = float(problem.true_loss(x_best))
    return RunResult(
        algorithm=str(algorithm),
        seed=int(seed),
        batch_size=int(batch_size),
        evaluations=int(problem.evaluations),
        best_noisy=float(problem.best_observed_fvalue1),
        post_true=float(post_true),
    )


def median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return float(np.median(np.asarray(xs, dtype=float)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.path.join(BASE_DIR, "Results", "application_logreg_minibatch_sweep"))
    parser.add_argument(
        "--dataset",
        default="synthetic",
        help="Dataset for the per-seed instance: synthetic | breast_cancer | digits0.",
    )
    parser.add_argument("--dim", type=int, default=40)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--batch-sizes", default="8,16,32,64")
    parser.add_argument("--budget-mult", type=int, default=100, help="Max evals = budget_mult * dim")
    parser.add_argument("--seeds", default="1-8")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers across (seed, algorithm).")
    parser.add_argument("--w-max", type=float, default=5.0)
    parser.add_argument("--weight-sigma", type=float, default=0.0, help="Lognormal reweighting sigma (0 disables).")
    parser.add_argument(
        "--weight-sigma-stochastic-only",
        action="store_true",
        help="If set, apply lognormal weights only when batch_size < N (full-batch becomes deterministic).",
    )
    parser.add_argument("--l2-reg", type=float, default=0.0)
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--eval-independent-noise", action="store_true")
    parser.add_argument(
        "--algorithms",
        default="CMA-ES,BERW-Hetero,ProbeSwitch-MR(t=0.12),ProbeSwitch-MR-Warmstart(t=0.12)",
        help="Comma-separated subset of algorithms to run.",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    os.chdir(BASE_DIR)
    os.makedirs(results_dir, exist_ok=True)

    dataset = str(args.dataset).strip().lower()
    if dataset == "synthetic":
        dim = int(args.dim)
    else:
        X0, _y0 = _load_base_dataset(dataset)
        dim = int(X0.shape[1]) + 1  # +1 bias
        if int(args.dim) != dim:
            print(f"[logreg] NOTE: --dim={int(args.dim)} ignored for dataset={dataset!r}; using dim={dim}.")
    n_samples = int(args.n_samples)
    batch_sizes = parse_int_list(args.batch_sizes)
    seeds = parse_int_list(args.seeds)
    max_evals = int(args.budget_mult) * dim

    all_algorithms = {
        "CMA-ES": cmaes_full_optimizer,
        "CMA-ES-sep": cmaes_sep_optimizer,
        "BERW-Hetero": berw_hetero_optimizer,
        "ProbeSwitch-MR(t=0.12)": probeswitch_mr_t012_optimizer,
        "ProbeSwitch-MR-Warmstart(t=0.12)": probeswitch_mr_warmstart_t012_optimizer,
    }
    want = [s.strip() for s in str(args.algorithms).split(",") if s.strip()]
    algorithms = {k: v for k, v in all_algorithms.items() if k in set(want)}
    if not algorithms:
        raise SystemExit("No algorithms selected. Available: " + ", ".join(sorted(all_algorithms.keys())))

    sweep_rows: list[dict[str, object]] = []

    for bs in batch_sizes:
        subdir = os.path.join(results_dir, f"batch_{safe_dir_token(str(bs))}")
        os.makedirs(subdir, exist_ok=True)

        run_rows: list[RunResult] = []
        # Note: multi-threading here parallelizes independent (seed, algorithm) runs.
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            futures = []
            for seed in seeds:
                for algo_name, opt in algorithms.items():
                    futures.append(
                        ex.submit(
                            run_one,
                            algorithm=str(algo_name),
                            optimizer=opt,
                            seed=int(seed),
                            batch_size=int(bs),
                            dim=int(dim),
                            n_samples=int(n_samples),
                            max_evals=int(max_evals),
                            w_max=float(args.w_max),
                            weight_sigma=float(args.weight_sigma),
                            weight_sigma_stochastic_only=bool(args.weight_sigma_stochastic_only),
                            l2_reg=float(args.l2_reg),
                            label_noise=float(args.label_noise),
                            eval_independent_noise=bool(args.eval_independent_noise),
                            dataset=str(args.dataset),
                            add_bias=(str(args.dataset).strip().lower() != "synthetic"),
                            standardize=True,
                        )
                    )
            for fut in as_completed(futures):
                run_rows.append(fut.result())

        runs_csv = os.path.join(subdir, "runs.csv")
        with open(runs_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["algorithm", "seed", "batch_size", "max_evals", "evaluations", "best_noisy", "post_true"])
            for r in sorted(run_rows, key=lambda rr: (rr.algorithm, rr.seed)):
                w.writerow(
                    [
                        r.algorithm,
                        r.seed,
                        r.batch_size,
                        int(max_evals),
                        int(r.evaluations),
                        f"{r.best_noisy:.12g}",
                        f"{r.post_true:.12g}",
                    ]
                )

        # summary.csv (median post_true)
        by_algo: dict[str, list[float]] = defaultdict(list)
        for r in run_rows:
            by_algo[str(r.algorithm)].append(float(r.post_true))
        summary_csv = os.path.join(subdir, "summary.csv")
        with open(summary_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["algorithm", "n", "median_post_true"])
            for algo in sorted(by_algo.keys()):
                vals = by_algo[algo]
                w.writerow([algo, len(vals), f"{median(vals):.12g}"])

        # pairwise sign tests (post_true)
        sign_out = os.path.join(subdir, "pairwise_sign_test_post_true.csv")
        if len(by_algo) >= 2:
            subprocess.check_call(
                [
                    "python3",
                    os.path.join(BASE_DIR, "tools", "pairwise_sign_test_runs.py"),
                    "--runs-csv",
                    runs_csv,
                    "--metric",
                    "post_true",
                    "--group-by",
                    "seed",
                    "--lower-is-better",
                    "--output",
                    sign_out,
                ],
                cwd=BASE_DIR,
            )

        # boxplot
        plt.figure(figsize=(8.5, 4.6))
        names = sorted(by_algo.keys())
        data = [by_algo[n] for n in names]
        try:
            plt.boxplot(data, tick_labels=names, showmeans=True)
        except TypeError:
            plt.boxplot(data, labels=names, showmeans=True)
        plt.ylabel("Post hoc true loss (full dataset; lower is better)")
        plt.title(
            f"Mini-batch logistic regression | dataset={dataset} d={dim} N={n_samples} bs={bs} "
            f"evals={max_evals} | weight_sigma={float(args.weight_sigma):g}"
        )
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(subdir, "final_boxplot.png"), dpi=200)
        plt.close()

        for algo in names:
            sweep_rows.append(
                {
                    "batch_size": int(bs),
                    "algorithm": algo,
                    "median_post_true": float(median(by_algo[algo])),
                }
            )

        print("Wrote:", repo_relpath(runs_csv))
        print("Wrote:", repo_relpath(summary_csv))
        if len(by_algo) >= 2:
            print("Wrote:", repo_relpath(sign_out))

    sweep_csv = os.path.join(results_dir, "sweep_summary.csv")
    with open(sweep_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch_size", "algorithm", "median_post_true"])
        for row in sweep_rows:
            w.writerow([row["batch_size"], row["algorithm"], f"{row['median_post_true']:.12g}"])
    print("Wrote:", repo_relpath(sweep_csv))


if __name__ == "__main__":
    main()
