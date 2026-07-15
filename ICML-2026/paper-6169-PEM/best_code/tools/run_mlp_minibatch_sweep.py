#!/usr/bin/env python3
"""
External validity benchmark (non-COCO):

Mini-batch 1-hidden-layer MLP classification as a stochastic black-box objective.

Why this exists:
- The strongest external evidence is convex logreg. A natural next question is whether
  misranking-aware selection and ProbeSwitch generalize beyond convex ERM.
- A small MLP makes the objective nonconvex while keeping a controlled, tunable
  misranking axis via batch size (and optional heavy-tailed reweighting).

This script supports both:
- `--dataset synthetic` (teacher-generated synthetic data; legacy default), and
- real ML datasets (e.g., `--dataset breast_cancer|digits0`) for stronger external validity.

Protocol:
- Generate a fixed synthetic dataset (X, y) per seed from a teacher MLP.
- Optimize theta (all network parameters) in a box constraint to minimize
  cross-entropy on random mini-batches.
- Report the returned best_x evaluated on the full dataset (noise-free metric).

Outputs:
  Results/application_mlp_minibatch_sweep/
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
from functools import lru_cache
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from _project import BASE_DIR, repo_relpath

from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero as berw_hetero_optimizer,
)
from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_robust as berw_hetero_robust_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t012 as probeswitch_mr_t012_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_warmstart_t012 as probeswitch_mr_warmstart_t012_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_noise_probe_switch as probeswitch_noise_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_noise_probe_switch_warmstart as probeswitch_noise_warmstart_optimizer,
)
from baselines.cmaes_sep import my_optimizer as cmaes_sep_optimizer
from baselines.cmaes_sep_bootstrap_rank import (
    my_optimizer as cmaes_sep_bootstrap_rank_optimizer,
)


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


def theta_dim_from_in_dim(*, in_dim: int, hidden_dim: int, out_dim: int = 1) -> int:
    in_dim = int(in_dim)
    hidden_dim = int(hidden_dim)
    out_dim = int(out_dim)
    if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
        raise ValueError("in_dim/hidden_dim/out_dim must be positive.")
    return int(in_dim * hidden_dim + hidden_dim + hidden_dim * out_dim + out_dim)


def infer_in_dim_from_theta_dim(*, theta_dim: int, hidden_dim: int, out_dim: int = 1) -> int:
    """
    For a 1-hidden-layer MLP with parameters:
      W1: (in_dim, hidden_dim)
      b1: (hidden_dim,)
      W2: (hidden_dim, out_dim)
      b2: (out_dim,)

    total = in_dim*hidden + hidden + hidden*out + out
          = hidden*(in_dim + 1 + out) + out
    """

    theta_dim = int(theta_dim)
    hidden_dim = int(hidden_dim)
    out_dim = int(out_dim)
    if theta_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
        raise ValueError("theta_dim/hidden_dim/out_dim must be positive.")

    rem = theta_dim - out_dim
    denom = hidden_dim
    if rem <= 0 or rem % denom != 0:
        raise ValueError(
            f"Incompatible dims: theta_dim={theta_dim}, hidden_dim={hidden_dim}, out_dim={out_dim}. "
            f"Need (theta_dim - out_dim) % hidden_dim == 0."
        )
    in_plus = rem // denom  # = in_dim + 1 + out_dim
    in_dim = int(in_plus - 1 - out_dim)
    if in_dim <= 0:
        raise ValueError(
            f"Inferred in_dim={in_dim} is invalid for theta_dim={theta_dim}, hidden_dim={hidden_dim}, out_dim={out_dim}."
        )
    return int(in_dim)


@lru_cache(maxsize=None)
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


class NoisyMiniBatchMLPProblem:
    def __init__(
        self,
        *,
        seed: int,
        dim: int,
        hidden_dim: int,
        n_samples: int,
        batch_size: int,
        w_max: float,
        weight_sigma: float,
        weight_sigma_stochastic_only: bool = False,
        l2_reg: float,
        label_noise: float,
        teacher_scale: float,
        eval_independent_noise: bool,
        dataset: str = "synthetic",
        standardize: bool = True,
        id_function: int = 9102,
    ):
        self.dimension = int(dim)
        self.lower_bounds = -float(w_max) * np.ones(self.dimension, dtype=float)
        self.upper_bounds = float(w_max) * np.ones(self.dimension, dtype=float)
        self.initial_solution = np.zeros(self.dimension, dtype=float)

        self.id_function = int(id_function)
        self.id_instance = int(seed)

        dataset = str(dataset).strip().lower()

        self._base_seed = int(seed) & 0xFFFFFFFF
        self._eval_independent_noise = bool(eval_independent_noise)
        self._rng = np.random.RandomState(int(self._base_seed) ^ 0xC0FFEE)

        self._hidden_dim = int(max(1, hidden_dim))
        self._out_dim = 1
        self._in_dim = infer_in_dim_from_theta_dim(
            theta_dim=int(self.dimension),
            hidden_dim=int(self._hidden_dim),
            out_dim=int(self._out_dim),
        )

        self._batch_size = int(max(1, batch_size))
        self._weight_sigma = float(max(0.0, weight_sigma))
        self._weight_sigma_stochastic_only = bool(weight_sigma_stochastic_only)
        self._l2_reg = float(max(0.0, l2_reg))

        # dataset
        n_samples = int(max(32, n_samples))
        rng_data = np.random.RandomState(int(self._base_seed) ^ 0xA5A5A5A5)
        if dataset == "synthetic":
            X = rng_data.randn(n_samples, self._in_dim).astype(float, copy=False)

            # Teacher MLP -> Bernoulli labels.
            theta_true = rng_data.randn(self.dimension).astype(float, copy=False)
            theta_true = float(teacher_scale) * theta_true / max(1e-12, float(np.linalg.norm(theta_true)))
            logits = self._forward_logits(theta_true, X)
            p = 1.0 / (1.0 + np.exp(-logits))
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

            if int(X.shape[1]) != int(self._in_dim):
                expect_dim = theta_dim_from_in_dim(in_dim=int(X.shape[1]), hidden_dim=int(self._hidden_dim), out_dim=int(self._out_dim))
                raise ValueError(
                    f"Input-dimension mismatch for dataset={dataset!r}: got X.shape[1]={int(X.shape[1])}, "
                    f"but inferred in_dim={int(self._in_dim)} from theta_dim={int(self.dimension)}, hidden_dim={int(self._hidden_dim)}. "
                    f"(If you intended to use in_dim={int(X.shape[1])}, set --dim to {expect_dim} or adjust --hidden-dim.)"
                )
        if float(label_noise) > 0.0:
            flip = rng_data.rand(n_samples) < float(label_noise)
            y = np.where(flip, 1.0 - y, y)

        self.X = X
        self.y = y

        self.evaluations = 0
        self.final_target_hit = False
        self.best_observed_fvalue1 = float("inf")
        self.best_x: np.ndarray | None = None

    def _unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta = np.asarray(theta, dtype=float)
        idx = 0
        in_dim = int(self._in_dim)
        h = int(self._hidden_dim)

        W1 = theta[idx : idx + in_dim * h].reshape((in_dim, h))
        idx += in_dim * h
        b1 = theta[idx : idx + h]
        idx += h
        W2 = theta[idx : idx + h].reshape((h, 1))
        idx += h
        b2 = theta[idx : idx + 1]
        idx += 1

        if idx != int(self.dimension):
            raise RuntimeError("Internal shape mismatch while unpacking theta.")
        return W1, b1, W2, b2

    def _forward_logits(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        W1, b1, W2, b2 = self._unpack(theta)
        h = np.tanh(X @ W1 + b1[None, :])
        logits = (h @ W2 + b2[None, :]).reshape((-1,))
        return np.asarray(logits, dtype=float)

    def _rng_eval(self, eval_id: int) -> np.random.RandomState:
        if not self._eval_independent_noise:
            return self._rng
        seed_eval = (int(self._base_seed) * 1000003 + int(eval_id) * 9176 + 2026) & 0xFFFFFFFF
        return np.random.RandomState(int(seed_eval))

    def _loss_full(self, theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        logits = self._forward_logits(theta, self.X)
        loss = float(np.mean(logistic_loss(logits, self.y)))
        if self._l2_reg > 0.0:
            loss += 0.5 * float(self._l2_reg) * float(np.dot(theta, theta))
        return float(loss)

    def true_loss(self, theta: np.ndarray) -> float:
        theta = np.clip(np.asarray(theta, dtype=float), self.lower_bounds, self.upper_bounds)
        return self._loss_full(theta)

    def __call__(self, theta: np.ndarray) -> float:
        self.evaluations += 1
        eval_id = int(self.evaluations)

        theta = np.clip(np.asarray(theta, dtype=float), self.lower_bounds, self.upper_bounds)
        rng = self._rng_eval(eval_id)

        n = int(self.X.shape[0])
        bs = int(min(self._batch_size, n))
        if bs >= n:
            idx = np.arange(n, dtype=int)
        else:
            idx = rng.randint(0, n, size=bs)

        Xb = self.X[idx]
        yb = self.y[idx]
        logits = self._forward_logits(theta, Xb)
        per = logistic_loss(logits, yb)

        apply_weights = bool(self._weight_sigma > 0.0)
        if apply_weights and self._weight_sigma_stochastic_only and bs >= n:
            apply_weights = False

        if apply_weights:
            s = float(self._weight_sigma)
            wts = np.exp(s * rng.randn(bs) - 0.5 * s * s)  # mean-1 lognormal weights
            val = float(np.mean(wts * per))
        else:
            val = float(np.mean(per))

        if self._l2_reg > 0.0:
            val += 0.5 * float(self._l2_reg) * float(np.dot(theta, theta))

        if val < self.best_observed_fvalue1:
            self.best_observed_fvalue1 = float(val)
            self.best_x = theta.copy()
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
    hidden_dim: int,
    n_samples: int,
    max_evals: int,
    w_max: float,
    weight_sigma: float,
    weight_sigma_stochastic_only: bool,
    l2_reg: float,
    label_noise: float,
    teacher_scale: float,
    eval_independent_noise: bool,
    dataset: str,
    standardize: bool,
) -> RunResult:
    problem = NoisyMiniBatchMLPProblem(
        seed=int(seed),
        dim=int(dim),
        hidden_dim=int(hidden_dim),
        n_samples=int(n_samples),
        batch_size=int(batch_size),
        w_max=float(w_max),
        weight_sigma=float(weight_sigma),
        weight_sigma_stochastic_only=bool(weight_sigma_stochastic_only),
        l2_reg=float(l2_reg),
        label_noise=float(label_noise),
        teacher_scale=float(teacher_scale),
        eval_independent_noise=bool(eval_independent_noise),
        dataset=str(dataset),
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
    parser.add_argument("--results-dir", default=os.path.join(BASE_DIR, "Results", "application_mlp_minibatch_sweep"))
    parser.add_argument("--dim", type=int, default=40, help="Parameter dimension (theta). Must match 1-hidden-layer MLP.")
    parser.add_argument("--hidden-dim", type=int, default=3, help="Hidden width (controls inferred input dimension).")
    parser.add_argument("--dataset", default="synthetic", help="Dataset: synthetic|breast_cancer|digits0")
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--batch-sizes", default="8,16,32,64")
    parser.add_argument("--budget-mult", type=int, default=100, help="Max evals = budget_mult * dim")
    parser.add_argument("--seeds", default="1-8")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers across (seed, algorithm).")
    parser.add_argument("--w-max", type=float, default=5.0)
    parser.add_argument("--teacher-scale", type=float, default=2.0)
    parser.add_argument("--weight-sigma", type=float, default=0.0, help="Lognormal reweighting sigma (0 disables).")
    parser.add_argument(
        "--weight-sigma-stochastic-only",
        action="store_true",
        help="If set, apply lognormal weights only when batch_size < N (full-batch becomes deterministic).",
    )
    parser.add_argument("--l2-reg", type=float, default=0.0)
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--no-standardize", action="store_true", help="Disable feature standardization for real datasets.")
    parser.add_argument("--eval-independent-noise", action="store_true")
    parser.add_argument(
        "--algorithms",
        default="CMA-ES-sep,BERW-Hetero,ProbeSwitch-MR(t=0.12)",
        help="Comma-separated subset of algorithms to run.",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    os.chdir(BASE_DIR)
    os.makedirs(results_dir, exist_ok=True)

    dataset = str(args.dataset).strip().lower()
    standardize = not bool(args.no_standardize)
    dim = int(args.dim)
    hidden_dim = int(args.hidden_dim)
    if dataset == "synthetic":
        _ = infer_in_dim_from_theta_dim(theta_dim=dim, hidden_dim=hidden_dim, out_dim=1)  # validate
    else:
        X0, _y0 = _load_base_dataset(dataset)
        dim = theta_dim_from_in_dim(in_dim=int(X0.shape[1]), hidden_dim=int(hidden_dim), out_dim=1)

    n_samples = int(args.n_samples)
    batch_sizes = parse_int_list(args.batch_sizes)
    seeds = parse_int_list(args.seeds)
    max_evals = int(args.budget_mult) * dim

    all_algorithms = {
        "CMA-ES-sep": cmaes_sep_optimizer,
        "CMA-ES-sep-BootstrapRank": cmaes_sep_bootstrap_rank_optimizer,
        "BERW-Hetero": berw_hetero_optimizer,
        "BERW-HeteroRobust": berw_hetero_robust_optimizer,
        "ProbeSwitch-MR(t=0.12)": probeswitch_mr_t012_optimizer,
        "ProbeSwitch-MR-Warmstart(t=0.12)": probeswitch_mr_warmstart_t012_optimizer,
        "ProbeSwitch-Noise": probeswitch_noise_optimizer,
        "ProbeSwitch-Noise-Warmstart": probeswitch_noise_warmstart_optimizer,
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
        # For strict determinism, set `--workers 1`.
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
                            hidden_dim=int(hidden_dim),
                            n_samples=int(n_samples),
                            max_evals=int(max_evals),
                            w_max=float(args.w_max),
                            weight_sigma=float(args.weight_sigma),
                            weight_sigma_stochastic_only=bool(args.weight_sigma_stochastic_only),
                            l2_reg=float(args.l2_reg),
                            label_noise=float(args.label_noise),
                            teacher_scale=float(args.teacher_scale),
                            eval_independent_noise=bool(args.eval_independent_noise),
                            dataset=str(dataset),
                            standardize=bool(standardize),
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

        plt.figure(figsize=(8.5, 4.6))
        names = sorted(by_algo.keys())
        data = [by_algo[n] for n in names]
        try:
            plt.boxplot(data, tick_labels=names, showmeans=True)
        except TypeError:
            plt.boxplot(data, labels=names, showmeans=True)
        plt.ylabel("Post hoc true loss (full dataset; lower is better)")
        if dataset == "synthetic":
            in_dim = infer_in_dim_from_theta_dim(theta_dim=int(dim), hidden_dim=int(hidden_dim), out_dim=1)
        else:
            X0, _y0 = _load_base_dataset(dataset)
            in_dim = int(X0.shape[1])
        plt.title(
            f"Mini-batch MLP | dataset={dataset} | theta_dim={dim} in={in_dim} h={hidden_dim} N={n_samples} bs={bs} "
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
