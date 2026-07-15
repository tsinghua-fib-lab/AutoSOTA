#!/usr/bin/env python3
"""
External validity benchmark (standard ML scenario): noisy hyperparameter optimization (HPO).

We optimize *training hyperparameters* of a mini-batch SGD logistic regression learner.
Each objective call is a noisy training run (random mini-batches + optional heavy-tail reweighting),
and returns a validation loss.

Why this exists:
- HPO is a canonical fixed-budget setting where evaluation is expensive and noisy.
- Resampling (repeat training with multiple seeds) is a common uncertainty-handling baseline,
  but consumes budget and can slow progress under fixed budgets.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from _project import BASE_DIR, repo_relpath

from algorithms import probe_switch as ms
from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_robust as berw_hetero_robust,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_robust_t012 as probeswitch_mr_robust_t012,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_robust_t022 as probeswitch_mr_robust_t022,
)
from baselines.cmaes_noise import my_optimizer_uh_maxevals30 as uh_cmaes_maxevals30
from baselines.cmaes_sep import my_optimizer as cmaes_sep
from baselines.cmaes_sep_resample import my_optimizer_resample5, my_optimizer_resample10


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


def _load_base_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    name = str(name).strip().lower()
    if name == "digits0":
        from sklearn.datasets import load_digits  # type: ignore

        X, y = load_digits(return_X_y=True)
        y01 = (np.asarray(y, dtype=int) == 0).astype(float)
        return np.asarray(X, dtype=float), y01

    if name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer  # type: ignore

        X, y = load_breast_cancer(return_X_y=True)
        y01 = np.asarray(y, dtype=float)
        return np.asarray(X, dtype=float), y01

    raise ValueError(f"Unknown dataset: {name}")


def logistic_loss(z: np.ndarray, y01: np.ndarray) -> np.ndarray:
    # stable: log(1+exp(z)) - y*z
    return np.logaddexp(0.0, z) - y01 * z


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def cvar(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= 0:
        return float("nan")
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    k = int(max(1, math.ceil(alpha * values.size)))
    worst = np.sort(values)[-k:]
    return float(np.mean(worst))


def decode_hparams(x: np.ndarray) -> dict[str, float | int]:
    """
    Decode a bounded vector into HPO hyperparameters.

    x = [log10_lr, log10_wd, mom_raw, batch_log2, log10_init]
    """
    x = np.asarray(x, dtype=float).reshape((-1,))
    if x.size != 5:
        raise ValueError("Expected dim=5")

    log10_lr = float(x[0])
    log10_wd = float(x[1])
    mom_raw = float(x[2])
    batch_log2 = float(x[3])
    log10_init = float(x[4])

    lr = 10.0 ** log10_lr
    wd = 10.0 ** log10_wd
    momentum = 0.99 * float(sigmoid(np.asarray([mom_raw]))[0])

    b = int(round(batch_log2))
    b = int(max(2, min(8, b)))
    batch_size = int(2**b)

    init_scale = 10.0 ** log10_init
    return {
        "lr": float(lr),
        "weight_decay": float(wd),
        "momentum": float(momentum),
        "batch_size": int(batch_size),
        "init_scale": float(init_scale),
    }


class NoisyHPOLogRegProblem:
    """
    Black-box interface for optimizing SGD hyperparameters on a fixed dataset instance.

    Each objective call runs *one* noisy training run and returns validation loss.
    """

    def __init__(
        self,
        *,
        seed: int,
        dataset: str,
        n_samples: int,
        standardize: bool,
        train_frac: float,
        train_steps: int,
        weight_sigma: float,
        eval_independent_noise: bool,
        id_function: int = 9400,
    ):
        self.id_function = int(id_function)
        self.id_instance = int(seed)

        self._base_seed = int(seed) & 0xFFFFFFFF
        self._eval_independent_noise = bool(eval_independent_noise)
        self._rng = np.random.RandomState(int(self._base_seed) ^ 0xC0FFEE)

        dataset = str(dataset).strip().lower()
        X0, y0 = _load_base_dataset(dataset)
        rng_data = np.random.RandomState(int(self._base_seed) ^ 0xA5A5A5A5)
        n_samples = int(max(64, n_samples))
        idx = rng_data.randint(0, int(X0.shape[0]), size=n_samples)
        X = np.asarray(X0[idx], dtype=float)
        y = np.asarray(y0[idx], dtype=float)
        if bool(standardize):
            mu = np.mean(X, axis=0)
            sd = np.std(X, axis=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            X = (X - mu) / sd
        X = np.concatenate([X, np.ones((n_samples, 1), dtype=float)], axis=1)  # bias

        # Fixed train/val split per instance.
        n_train = int(max(16, round(float(train_frac) * n_samples)))
        perm = rng_data.permutation(n_samples)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]
        if val_idx.size <= 0:
            val_idx = perm[: int(max(16, n_samples // 4))]

        self.X_train = np.asarray(X[train_idx], dtype=float)
        self.y_train = np.asarray(y[train_idx], dtype=float)
        self.X_val = np.asarray(X[val_idx], dtype=float)
        self.y_val = np.asarray(y[val_idx], dtype=float)

        self._train_steps = int(max(10, train_steps))
        self._weight_sigma = float(max(0.0, weight_sigma))

        # HPO dimension: 5 hyperparameters.
        self.dimension = 5
        # Bounds in the encoded space.
        # log10_lr, log10_wd, mom_raw, batch_log2, log10_init
        self.lower_bounds = np.asarray([-4.0, -6.0, -3.0, 2.0, -3.0], dtype=float)
        self.upper_bounds = np.asarray([-0.5, -2.0, 3.0, 8.0, 0.0], dtype=float)
        self.initial_solution = np.asarray([-2.0, -4.0, 0.0, 5.0, -1.0], dtype=float)  # lr=1e-2, wd=1e-4, mom~0.5, bs=32, init=0.1

        self.evaluations = 0
        self.final_target_hit = False
        self.best_observed_fvalue1 = float("inf")
        self.best_x: np.ndarray | None = None

    def _rng_eval(self, eval_id: int) -> np.random.RandomState:
        if not self._eval_independent_noise:
            return self._rng
        seed_eval = (int(self._base_seed) * 1000003 + int(eval_id) * 9176 + 6060) & 0xFFFFFFFF
        return np.random.RandomState(int(seed_eval))

    def _train_and_eval_once(self, x: np.ndarray, *, rng: np.random.RandomState) -> float:
        hp = decode_hparams(x)
        lr = float(hp["lr"])
        wd = float(hp["weight_decay"])
        mom = float(hp["momentum"])
        bs = int(hp["batch_size"])
        init_scale = float(hp["init_scale"])

        d = int(self.X_train.shape[1])
        w = init_scale * rng.randn(d).astype(float)
        v = np.zeros(d, dtype=float)

        n_train = int(self.X_train.shape[0])
        for _t in range(int(self._train_steps)):
            idx = rng.randint(0, n_train, size=int(bs))
            Xb = self.X_train[idx]
            yb = self.y_train[idx]

            logits = Xb @ w
            p = sigmoid(logits)
            # Gradient of mean logistic loss: X^T (p - y) / bs
            g = (Xb.T @ (p - yb)) / float(bs)

            if self._weight_sigma > 0.0:
                # Mean-1 lognormal reweighting (heavy-tailed), applied as a per-step gradient multiplier.
                # This is not meant to be a faithful optimizer, but a controllable noisy-eval axis for HPO.
                z = rng.randn()
                mult = math.exp(float(self._weight_sigma) * float(z) - 0.5 * float(self._weight_sigma) ** 2)
                g = float(mult) * g

            # L2 regularization.
            g = g + float(wd) * w

            v = float(mom) * v + g
            w = w - float(lr) * v

        # Validation loss (full batch, deterministic given w).
        z = self.X_val @ w
        loss = float(np.mean(logistic_loss(z, self.y_val))) + 0.5 * float(wd) * float(np.dot(w, w))
        return float(loss)

    def post_eval(self, x: np.ndarray, *, rng: np.random.RandomState, runs: int) -> dict[str, float]:
        vals = np.empty(int(runs), dtype=float)
        for i in range(int(runs)):
            vals[i] = self._train_and_eval_once(x, rng=rng)
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "cvar20": float(cvar(vals, 0.2)),
        }

    def __call__(self, x: np.ndarray) -> float:
        self.evaluations += 1
        eval_id = int(self.evaluations)
        x = np.clip(np.asarray(x, dtype=float), self.lower_bounds, self.upper_bounds)
        rng = self._rng_eval(eval_id)
        val = self._train_and_eval_once(x, rng=rng)
        if val < self.best_observed_fvalue1:
            self.best_observed_fvalue1 = float(val)
            self.best_x = x.copy()
        return float(val)


@dataclass(frozen=True)
class RunResult:
    algorithm: str
    seed: int
    max_evals: int
    evaluations: int
    best_noisy: float
    post_true: float
    post_median: float
    post_cvar20: float


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            for r in rows:
                w.writerow(r)


def boxplot(*, out_path: str, values_by_algo: dict[str, list[float]], title: str, ylabel: str) -> None:
    algos = sorted(values_by_algo.keys())
    data = [values_by_algo[a] for a in algos]
    plt.figure(figsize=(9.2, 4.8))
    plt.boxplot(data, tick_labels=algos, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def run_one(
    *,
    algorithm: str,
    optimizer,
    seed: int,
    max_evals: int,
    dataset: str,
    n_samples: int,
    standardize: bool,
    train_frac: float,
    train_steps: int,
    weight_sigma: float,
    eval_independent_noise: bool,
    post_runs: int,
    postselect_k: int,
) -> RunResult:
    base_problem = NoisyHPOLogRegProblem(
        seed=int(seed),
        dataset=str(dataset),
        n_samples=int(n_samples),
        standardize=bool(standardize),
        train_frac=float(train_frac),
        train_steps=int(train_steps),
        weight_sigma=float(weight_sigma),
        eval_independent_noise=bool(eval_independent_noise),
    )

    class ArchiveProblem:
        def __init__(self, problem, *, k: int):
            self._problem = problem
            self._k = int(max(1, k))
            self._seen: set[bytes] = set()
            self._items: list[tuple[float, np.ndarray]] = []

        def __call__(self, x):
            val = float(self._problem(x))
            x_arr = np.asarray(x, dtype=float)
            key = np.round(x_arr, 6).astype(np.float32, copy=False).tobytes()
            if key not in self._seen:
                self._seen.add(key)
                self._items.append((val, x_arr.copy()))
                self._items.sort(key=lambda t: t[0])
                if len(self._items) > self._k:
                    self._items = self._items[: self._k]
            return val

        def candidates(self) -> list[np.ndarray]:
            return [x for _v, x in sorted(self._items, key=lambda t: t[0])]

        def __getattr__(self, name):
            return getattr(self._problem, name)

    wrapped = ArchiveProblem(base_problem, k=int(postselect_k))
    optimizer(wrapped, int(max_evals))

    cand = wrapped.candidates()
    if base_problem.best_x is not None:
        cand.append(np.asarray(base_problem.best_x, dtype=float).copy())
    cand.append(np.asarray(base_problem.initial_solution, dtype=float).copy())

    uniq: list[np.ndarray] = []
    seen: set[bytes] = set()
    for x in cand:
        key = np.round(x, 6).astype(np.float32, copy=False).tobytes()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(x)

    best_post = None
    for x in uniq:
        # Common RNG across candidates for stable post-selection.
        rng = np.random.RandomState((int(seed) * 1000003 + 777) & 0xFFFFFFFF)
        stats = base_problem.post_eval(x, rng=rng, runs=int(post_runs))
        if best_post is None or float(stats["mean"]) < float(best_post["mean"]):
            best_post = stats
    if best_post is None:
        best_post = base_problem.post_eval(base_problem.initial_solution, rng=np.random.RandomState(int(seed) ^ 0xABC), runs=int(post_runs))

    return RunResult(
        algorithm=str(algorithm),
        seed=int(seed),
        max_evals=int(max_evals),
        evaluations=int(base_problem.evaluations),
        best_noisy=float(base_problem.best_observed_fvalue1),
        post_true=float(best_post["mean"]),
        post_median=float(best_post["median"]),
        post_cvar20=float(best_post["cvar20"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--dataset", default="digits0", choices=["digits0", "breast_cancer"])
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--weight-sigma", type=float, default=1.0, help="Heavy-tail strength for gradient noise.")
    parser.add_argument("--eval-independent-noise", action="store_true")

    parser.add_argument("--seeds", default="1-20")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget-mult", type=int, default=40, help="Budget multiplier × hpo_dim (dim=5).")
    parser.add_argument("--post-runs", type=int, default=16, help="Post-hoc evaluation runs per candidate.")
    parser.add_argument("--postselect-k", type=int, default=8)

    parser.add_argument(
        "--algorithms",
        default="CMA-ES-sep,CMA-ES-Resample(k=5),CMA-ES-Resample(k=10),BERW-HeteroRobust",
    )
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    seeds = parse_int_list(str(args.seeds))
    hpo_dim = 5
    max_evals = int(args.budget_mult) * int(hpo_dim)

    algo_map = {
        "CMA-ES-sep": cmaes_sep,
        "CMA-ES-Resample(k=5)": my_optimizer_resample5,
        "CMA-ES-Resample(k=10)": my_optimizer_resample10,
        "BERW-HeteroRobust": berw_hetero_robust,
        "UH-CMA-ES(maxevals=30)": uh_cmaes_maxevals30,
        "ProbeSwitch-MR-Robust(t=0.12)": probeswitch_mr_robust_t012,
        "ProbeSwitch-MR-Robust(t=0.22)": probeswitch_mr_robust_t022,
    }
    algos = [a.strip() for a in str(args.algorithms).split(",") if a.strip()]
    for a in algos:
        if a not in algo_map:
            raise SystemExit(f"Unknown algorithm: {a} (known: {sorted(algo_map.keys())})")

    out_dir = os.path.abspath(str(args.results_dir))
    os.makedirs(out_dir, exist_ok=True)

    # Probe values at x0 (per seed) for transfer studies.
    probe_rows: list[dict[str, object]] = []
    for seed in seeds:
        p = NoisyHPOLogRegProblem(
            seed=int(seed),
            dataset=str(args.dataset),
            n_samples=int(args.n_samples),
            standardize=not bool(args.no_standardize),
            train_frac=float(args.train_frac),
            train_steps=int(args.train_steps),
            weight_sigma=float(args.weight_sigma),
            eval_independent_noise=bool(args.eval_independent_noise),
        )
        rd = ms._misranking_probe(p, max_evals=10**9)
        rd2, tail_ratio = ms._tail_ratio_probe(p, max_evals=10**9, reps=2)
        rel_sd = ms._variance_probe(p, max_evals=10**9, reps=10)
        probe_rows.append(
            {
                "seed": int(seed),
                "dataset": str(args.dataset),
                "n_samples": int(args.n_samples),
                "train_steps": int(args.train_steps),
                "weight_sigma": float(args.weight_sigma),
                "misranking_rd": "" if rd is None else float(rd),
                "tail_probe_rd": "" if rd2 is None else float(rd2),
                "tail_ratio": "" if tail_ratio is None else float(tail_ratio),
                "variance_rel_sd": "" if rel_sd is None else float(rel_sd),
            }
        )
    write_csv(os.path.join(out_dir, "probe_values.csv"), probe_rows)

    runs: list[RunResult] = []
    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = []
        for seed in seeds:
            for algo in algos:
                futs.append(
                    ex.submit(
                        run_one,
                        algorithm=str(algo),
                        optimizer=algo_map[str(algo)],
                        seed=int(seed),
                        max_evals=int(max_evals),
                        dataset=str(args.dataset),
                        n_samples=int(args.n_samples),
                        standardize=not bool(args.no_standardize),
                        train_frac=float(args.train_frac),
                        train_steps=int(args.train_steps),
                        weight_sigma=float(args.weight_sigma),
                        eval_independent_noise=bool(args.eval_independent_noise),
                        post_runs=int(args.post_runs),
                        postselect_k=int(args.postselect_k),
                    )
                )
        for fut in as_completed(futs):
            runs.append(fut.result())

    runs_sorted = sorted(runs, key=lambda r: (r.algorithm, r.seed))
    write_csv(os.path.join(out_dir, "runs.csv"), [r.__dict__ for r in runs_sorted])

    by_algo: dict[str, list[RunResult]] = {}
    for r in runs_sorted:
        by_algo.setdefault(r.algorithm, []).append(r)

    summary_rows: list[dict[str, object]] = []
    for algo, rs in sorted(by_algo.items(), key=lambda t: t[0]):
        post_true = np.asarray([x.post_true for x in rs], dtype=float)
        summary_rows.append(
            {
                "algorithm": str(algo),
                "n_runs": int(len(rs)),
                "median_post_true": float(np.median(post_true)),
                "median_post_cvar20": float(np.median(np.asarray([x.post_cvar20 for x in rs], dtype=float))),
            }
        )
    write_csv(os.path.join(out_dir, "summary.csv"), summary_rows)

    boxplot(
        out_path=os.path.join(out_dir, "final_boxplot.png"),
        values_by_algo={a: [x.post_true for x in rs] for a, rs in by_algo.items()},
        title=f"HPO noisy logreg ({args.dataset}) | fixed budget={max_evals} evals | weight_sigma={args.weight_sigma}",
        ylabel="post_true (mean val loss across post runs) [lower is better]",
    )

    print("Wrote:", repo_relpath(os.path.join(out_dir, "runs.csv")))
    print("Wrote:", repo_relpath(os.path.join(out_dir, "summary.csv")))
    print("Wrote:", repo_relpath(os.path.join(out_dir, "probe_values.csv")))
    print("Wrote:", repo_relpath(os.path.join(out_dir, "final_boxplot.png")))


if __name__ == "__main__":
    main()
