#!/usr/bin/env python3
"""
External validity benchmark (non-COCO):

Risk-sensitive-ish control objective under *state-dependent* and *heavy-tailed* noise.

We optimize a linear feedback controller for a noisy linear dynamical system:
  x_{t+1} = A x_t + B u_t + w_t
  u_t = sat(-K x_t)

Key properties:
- Noise is *state-dependent* in effect: the same disturbance distribution can lead to
  very different cost variance depending on the closed-loop dynamics induced by K.
- Disturbances are heavy-tailed (Student-t), producing occasional outliers.
- Evaluation uses a small number of rollouts per call (noisy objective), while
  post-hoc evaluation uses many fresh rollouts (cleaner metric).

Outputs:
  Results/application_lqr_heavytail_control_*/
    runs.csv
    summary.csv
    probe_values.csv
    final_boxplot.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from _project import BASE_DIR, repo_relpath

from algorithms import probe_switch as ms
from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero as berw_hetero_optimizer,
)
from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_robust as berw_hetero_robust_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t008 as probeswitch_mr_t008_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t010 as probeswitch_mr_t010_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t012 as probeswitch_mr_t012_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t022 as probeswitch_mr_t022_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_robust_t012 as probeswitch_mr_robust_t012_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_robust_t022 as probeswitch_mr_robust_t022_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_robust_t019 as probeswitch_mr_robust_t019_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_robust_t021 as probeswitch_mr_robust_t021_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t019 as probeswitch_mr_t019_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t021 as probeswitch_mr_t021_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_noise_probe_switch as probeswitch_noise_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_noise_probe_switch_warmstart as probeswitch_noise_warmstart_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_variance_probe_switch as probeswitch_var_optimizer,
)
from baselines.cmaes_noise import my_optimizer_uh_maxevals30 as uh_cmaes_maxevals30_optimizer
from baselines.cmaes_sep import my_optimizer as cmaes_sep_optimizer
from baselines.cmaes_sep_resample import my_optimizer_resample5 as cmaes_resample5_optimizer
from baselines.cmaes_sep_resample import my_optimizer_resample10 as cmaes_resample10_optimizer


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


def spectral_radius(a: np.ndarray) -> float:
    vals = np.linalg.eigvals(np.asarray(a, dtype=float))
    return float(np.max(np.abs(vals)))


def solve_dare_iter(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    *,
    max_iter: int = 5000,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the discrete-time algebraic Riccati equation (DARE) by fixed-point iteration:
        P = Q + A^T P A - A^T P B (R + B^T P B)^(-1) B^T P A
    Returns (P, K) where K = (R + B^T P B)^(-1) B^T P A.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)

    p = q.copy()
    for _ in range(int(max_iter)):
        s = r + b.T @ p @ b
        # Jitter for numerical stability.
        s = s + 1e-12 * np.eye(s.shape[0], dtype=float)
        k = np.linalg.solve(s, b.T @ p @ a)
        p_new = q + a.T @ p @ a - a.T @ p @ b @ k
        if float(np.linalg.norm(p_new - p, ord="fro")) <= float(tol) * (1.0 + float(np.linalg.norm(p, ord="fro"))):
            p = p_new
            return p, k
        p = p_new
    # Best-effort return.
    s = r + b.T @ p @ b + 1e-12 * np.eye(r.shape[0], dtype=float)
    k = np.linalg.solve(s, b.T @ p @ a)
    return p, k


def sat_tanh(u: np.ndarray, u_max: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    m = float(max(1e-12, u_max))
    return m * np.tanh(u / m)


def cvar(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= 0:
        return float("nan")
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    k = int(max(1, math.ceil(alpha * values.size)))
    worst = np.sort(values)[-k:]
    return float(np.mean(worst))


class LQRHeavyTailControlProblem:
    """
    Black-box interface compatible with our ES optimizers.

    Decision variable: flattened K (shape: [m, n]) in [-bound, bound]^{m*n}.
    Objective (per evaluation): Monte Carlo estimate of expected cumulative cost over `T` steps,
    using `eval_rollouts` rollouts (typically small, e.g., 1–4).
    """

    def __init__(
        self,
        *,
        seed: int,
        state_dim: int,
        action_dim: int,
        horizon: int,
        bound: float,
        rho_target: float,
        u_max: float,
        init_std: float,
        noise_std: float,
        noise_df: float,
        noise_state_beta: float,
        noise_state_clip: float,
        eval_rollouts: int,
        eval_independent_noise: bool,
        q_scale: float,
        r_scale: float,
        terminal_scale: float,
        init_mode: str = "lqr",
        init_scale: float = 1.0,
        id_function: int = 9200,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.dimension = int(self.state_dim * self.action_dim)

        self.lower_bounds = -float(bound) * np.ones(self.dimension, dtype=float)
        self.upper_bounds = float(bound) * np.ones(self.dimension, dtype=float)

        self.id_function = int(id_function)
        self.id_instance = int(seed)

        self._base_seed = int(seed) & 0xFFFFFFFF
        self._eval_independent_noise = bool(eval_independent_noise)
        self._rng = np.random.RandomState(int(self._base_seed) ^ 0xC0FFEE)

        self._T = int(max(5, horizon))
        self._u_max = float(u_max)
        self._init_std = float(max(0.0, init_std))
        self._noise_std = float(max(0.0, noise_std))
        self._noise_df = float(max(2.05, noise_df))
        self._noise_state_beta = float(max(0.0, noise_state_beta))
        self._noise_state_clip = float(max(1.0, noise_state_clip))
        self._eval_rollouts = int(max(1, eval_rollouts))

        # Fix (A, B, Q, R) per instance.
        rng_sys = np.random.RandomState(int(self._base_seed) ^ 0xA5A5A5A5)
        a0 = rng_sys.randn(self.state_dim, self.state_dim).astype(float)
        sr = spectral_radius(a0)
        sr = max(1e-12, sr)
        a = (float(rho_target) / sr) * a0
        # Mildly encourage sparsity / stable directions.
        a = 0.90 * a + 0.10 * np.eye(self.state_dim, dtype=float)
        b = rng_sys.randn(self.state_dim, self.action_dim).astype(float)
        b = 0.6 * b / max(1e-12, float(np.linalg.norm(b, ord="fro")) / math.sqrt(float(self.state_dim * self.action_dim)))
        self._A = a
        self._B = b

        q = float(q_scale) * np.eye(self.state_dim, dtype=float)
        r = float(r_scale) * np.eye(self.action_dim, dtype=float)
        self._Q = q
        self._R = r
        self._Qf = float(terminal_scale) * q

        # Compute a stabilizing-ish LQR controller for initialization.
        _p, k_lqr = solve_dare_iter(self._A, self._B, self._Q, self._R)
        # Our control uses u = sat(-K x). LQR uses u = -K x, so keep K as-is.
        k0 = k_lqr.astype(float, copy=False)
        init_mode = str(init_mode).strip().lower()
        init_scale = float(init_scale)
        if init_mode == "zero":
            x0 = np.zeros(self.dimension, dtype=float)
        elif init_mode == "scaled_lqr":
            x0 = (init_scale * k0).reshape((self.action_dim * self.state_dim,))
        else:  # "lqr"
            x0 = k0.reshape((self.action_dim * self.state_dim,))
        x0 = np.clip(np.asarray(x0, dtype=float), self.lower_bounds, self.upper_bounds)
        self.initial_solution = x0

        self.evaluations = 0
        self.final_target_hit = False
        self.best_observed_fvalue1 = float("inf")
        self.best_x: np.ndarray | None = None

    def _rng_eval(self, eval_id: int) -> np.random.RandomState:
        if not self._eval_independent_noise:
            return self._rng
        seed_eval = (int(self._base_seed) * 1000003 + int(eval_id) * 9176 + 2026) & 0xFFFFFFFF
        return np.random.RandomState(int(seed_eval))

    def _rollout_cost(self, k_vec: np.ndarray, *, rng: np.random.RandomState) -> float:
        k = np.asarray(k_vec, dtype=float).reshape((self.action_dim, self.state_dim))
        x = self._init_std * rng.randn(self.state_dim).astype(float)
        cost = 0.0
        for _t in range(self._T):
            u = -k @ x
            u = sat_tanh(u, self._u_max)
            cost += float(x.T @ self._Q @ x) + float(u.T @ self._R @ u)
            scale = float(self._noise_std)
            if float(self._noise_state_beta) > 0.0:
                state_scale = float(np.linalg.norm(x)) / math.sqrt(float(self.state_dim))
                mult = 1.0 + float(self._noise_state_beta) * float(state_scale)
                mult = min(float(mult), float(self._noise_state_clip))
                scale = float(scale) * float(mult)
            w = float(scale) * rng.standard_t(self._noise_df, size=self.state_dim).astype(float)
            x = self._A @ x + self._B @ u + w
            if not np.all(np.isfinite(x)):
                return float("inf")
            if float(np.linalg.norm(x)) > 1e6:
                return 1e30
        cost += float(x.T @ self._Qf @ x)
        return float(cost)

    def post_eval(self, k_vec: np.ndarray, *, rng: np.random.RandomState, rollouts: int, alpha: float = 0.2) -> dict[str, float]:
        vals = np.empty(int(rollouts), dtype=float)
        for i in range(int(rollouts)):
            vals[i] = self._rollout_cost(k_vec, rng=rng)
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "cvar20": float(cvar(vals, float(alpha))),
        }

    def __call__(self, k_vec: np.ndarray) -> float:
        self.evaluations += 1
        eval_id = int(self.evaluations)
        k_vec = np.clip(np.asarray(k_vec, dtype=float), self.lower_bounds, self.upper_bounds)
        rng = self._rng_eval(eval_id)

        vals = np.empty(self._eval_rollouts, dtype=float)
        for i in range(self._eval_rollouts):
            vals[i] = self._rollout_cost(k_vec, rng=rng)
        val = float(np.mean(vals))

        if val < self.best_observed_fvalue1:
            self.best_observed_fvalue1 = float(val)
            self.best_x = k_vec.copy()
        return float(val)


@dataclass(frozen=True)
class RunResult:
    algorithm: str
    seed: int
    budget: int
    evaluations: int
    best_observed: float
    post_mean_best_observed: float
    post_median_best_observed: float
    post_cvar20_best_observed: float
    post_mean: float
    post_median: float
    post_cvar20: float
    postselect_k: int


def run_one(
    *,
    algorithm: str,
    optimizer,
    seed: int,
    budget: int,
    state_dim: int,
    action_dim: int,
    horizon: int,
    bound: float,
    rho_target: float,
    u_max: float,
    init_std: float,
    noise_std: float,
    noise_df: float,
    noise_state_beta: float,
    noise_state_clip: float,
    eval_rollouts: int,
    post_rollouts: int,
    eval_independent_noise: bool,
    q_scale: float,
    r_scale: float,
    terminal_scale: float,
    init_mode: str,
    init_scale: float,
    postselect_k: int,
) -> RunResult:
    base_problem = LQRHeavyTailControlProblem(
        seed=int(seed),
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        horizon=int(horizon),
        bound=float(bound),
        rho_target=float(rho_target),
        u_max=float(u_max),
        init_std=float(init_std),
        noise_std=float(noise_std),
        noise_df=float(noise_df),
        noise_state_beta=float(noise_state_beta),
        noise_state_clip=float(noise_state_clip),
        eval_rollouts=int(eval_rollouts),
        eval_independent_noise=bool(eval_independent_noise),
        q_scale=float(q_scale),
        r_scale=float(r_scale),
        terminal_scale=float(terminal_scale),
        init_mode=str(init_mode),
        init_scale=float(init_scale),
    )

    class ArchiveProblem:
        """
        Wrap a noisy problem and keep a small top-k archive of unique candidates
        by *noisy* value (for post-hoc, noise-reduced selection).
        """

        def __init__(self, problem, *, k: int):
            self._problem = problem
            self._k = int(max(1, k))
            self._seen: set[bytes] = set()
            self._items: list[tuple[float, np.ndarray]] = []  # (noisy_value, x)

        def __call__(self, x):
            val = float(self._problem(x))
            x_arr = np.asarray(x, dtype=float)
            key = np.round(x_arr, 6).astype(np.float32, copy=False).tobytes()
            if key not in self._seen:
                self._seen.add(key)
                self._items.append((val, x_arr.copy()))
                # keep only best k by noisy value
                self._items.sort(key=lambda t: t[0])
                if len(self._items) > self._k:
                    self._items = self._items[: self._k]
            return val

        def candidates(self) -> list[np.ndarray]:
            return [x for _v, x in sorted(self._items, key=lambda t: t[0])]

        def __getattr__(self, name):
            return getattr(self._problem, name)

    wrapped = ArchiveProblem(base_problem, k=int(postselect_k))
    optimizer(wrapped, int(budget))

    best_obs_x = base_problem.best_x if base_problem.best_x is not None else base_problem.initial_solution
    rng_best = np.random.RandomState((int(seed) * 1000003 + 424242) & 0xFFFFFFFF)
    post_best = base_problem.post_eval(best_obs_x, rng=rng_best, rollouts=int(post_rollouts), alpha=0.2)

    cand = wrapped.candidates()
    cand.append(np.asarray(base_problem.initial_solution, dtype=float).copy())
    # Deduplicate candidates (roughly) while preserving order.
    uniq: list[np.ndarray] = []
    seen: set[bytes] = set()
    for x in cand:
        key = np.round(x, 6).astype(np.float32, copy=False).tobytes()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(x)

    best_post = None
    best_post_x = None
    for x in uniq:
        # Common random numbers across candidates (for stable post-selection).
        rng = np.random.RandomState((int(seed) * 1000003 + 777) & 0xFFFFFFFF)
        stats = base_problem.post_eval(x, rng=rng, rollouts=int(post_rollouts), alpha=0.2)
        if best_post is None or float(stats["mean"]) < float(best_post["mean"]):
            best_post = stats
            best_post_x = x
    if best_post is None or best_post_x is None:
        best_post = post_best

    post = best_post

    return RunResult(
        algorithm=str(algorithm),
        seed=int(seed),
        budget=int(budget),
        evaluations=int(base_problem.evaluations),
        best_observed=float(base_problem.best_observed_fvalue1),
        post_mean_best_observed=float(post_best["mean"]),
        post_median_best_observed=float(post_best["median"]),
        post_cvar20_best_observed=float(post_best["cvar20"]),
        post_mean=float(post["mean"]),
        post_median=float(post["median"]),
        post_cvar20=float(post["cvar20"]),
        postselect_k=int(postselect_k),
    )


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
    plt.boxplot(data, labels=algos, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def _initial_reeval_then_run(problem, max_evals: int, *, reps: int, optimizer) -> None:
    reps = int(max(0, reps))
    x0 = np.asarray(problem.initial_solution, dtype=float)
    for _ in range(int(reps)):
        if bool(getattr(problem, "final_target_hit", False)) or int(getattr(problem, "evaluations", 0)) >= int(max_evals):
            break
        problem(x0)
    optimizer(problem, int(max_evals))


def cmaes_sep_init_reeval10(problem, max_evals):
    """Baseline: spend 10 evals re-evaluating x0, then run CMA-ES-sep (controls for probe cost/effect)."""
    _initial_reeval_then_run(problem, int(max_evals), reps=10, optimizer=cmaes_sep_optimizer)


def berw_hetero_init_reeval10(problem, max_evals):
    """Baseline: spend 10 evals re-evaluating x0, then run BERW-Hetero (controls for probe cost/effect)."""
    _initial_reeval_then_run(problem, int(max_evals), reps=10, optimizer=berw_hetero_optimizer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--budget-mult", type=int, default=80, help="Budget multiplier (×(state_dim*action_dim)).")
    parser.add_argument("--bound", type=float, default=2.5)
    parser.add_argument("--rho-target", type=float, default=1.02, help="Target spectral radius for A before adding identity mix.")
    parser.add_argument("--u-max", type=float, default=2.0)
    parser.add_argument("--init-std", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--noise-df", type=float, default=3.0)
    parser.add_argument(
        "--noise-state-beta",
        type=float,
        default=0.0,
        help="State-dependent noise: scale(t) = noise_std * min(noise_state_clip, 1 + beta * ||x||/sqrt(n)).",
    )
    parser.add_argument("--noise-state-clip", type=float, default=5.0, help="Max multiplicative factor for state-dependent noise.")
    parser.add_argument("--eval-rollouts", type=int, default=1)
    parser.add_argument("--post-rollouts", type=int, default=1024)
    parser.add_argument("--q-scale", type=float, default=1.0)
    parser.add_argument("--r-scale", type=float, default=0.05)
    parser.add_argument("--terminal-scale", type=float, default=5.0)
    parser.add_argument(
        "--init-mode",
        default="lqr",
        choices=["lqr", "zero", "scaled_lqr"],
        help="Initial solution used by all optimizers (lqr is a stabilizing warm-start; zero is a neutral start).",
    )
    parser.add_argument("--init-scale", type=float, default=1.0, help="Only used when --init-mode scaled_lqr.")
    parser.add_argument(
        "--postselect-k",
        type=int,
        default=20,
        help="Post-hoc selection: re-evaluate top-k unique candidates (by noisy value) and report the best by post_mean.",
    )
    parser.add_argument("--seeds", default="1-8")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-independent-noise", action="store_true")
    parser.add_argument(
        "--algorithms",
        default="CMA-ES-sep,BERW-Hetero,BERW-HeteroRobust,ProbeSwitch-MR(t=0.12),ProbeSwitch-Noise-Warmstart,ProbeSwitch-Var",
        help="Comma-separated list of algorithms (stable names used throughout this repository).",
    )
    args = parser.parse_args()

    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)
    dim = int(state_dim * action_dim)
    budget = int(args.budget_mult) * dim

    algo_map = {
        "CMA-ES-sep": cmaes_sep_optimizer,
        "CMA-ES-Resample(k=5)": cmaes_resample5_optimizer,
        "CMA-ES-Resample(k=10)": cmaes_resample10_optimizer,
        "CMA-ES-sep-InitReeval10": cmaes_sep_init_reeval10,
        "BERW-Hetero": berw_hetero_optimizer,
        "BERW-Hetero-InitReeval10": berw_hetero_init_reeval10,
        "BERW-HeteroRobust": berw_hetero_robust_optimizer,
        "ProbeSwitch-MR(t=0.08)": probeswitch_mr_t008_optimizer,
        "ProbeSwitch-MR(t=0.10)": probeswitch_mr_t010_optimizer,
        "ProbeSwitch-MR(t=0.12)": probeswitch_mr_t012_optimizer,
        "ProbeSwitch-MR(t=0.19)": probeswitch_mr_t019_optimizer,
        "ProbeSwitch-MR(t=0.21)": probeswitch_mr_t021_optimizer,
        "ProbeSwitch-MR(t=0.22)": probeswitch_mr_t022_optimizer,
        "ProbeSwitch-MR-Robust(t=0.12)": probeswitch_mr_robust_t012_optimizer,
        "ProbeSwitch-MR-Robust(t=0.19)": probeswitch_mr_robust_t019_optimizer,
        "ProbeSwitch-MR-Robust(t=0.21)": probeswitch_mr_robust_t021_optimizer,
        "ProbeSwitch-MR-Robust(t=0.22)": probeswitch_mr_robust_t022_optimizer,
        "ProbeSwitch-Noise": probeswitch_noise_optimizer,
        "ProbeSwitch-Noise-Warmstart": probeswitch_noise_warmstart_optimizer,
        "ProbeSwitch-Var": probeswitch_var_optimizer,
        "UH-CMA-ES(maxevals=30)": uh_cmaes_maxevals30_optimizer,
    }

    algos = [a.strip() for a in str(args.algorithms).split(",") if a.strip()]
    for a in algos:
        if a not in algo_map:
            raise SystemExit(f"Unknown algorithm: {a}. Available: {sorted(algo_map.keys())}")

    seeds = parse_int_list(args.seeds)
    out_dir = os.path.join(os.path.abspath(BASE_DIR), str(args.results_dir))
    os.makedirs(out_dir, exist_ok=True)

    # Probe values at x0 (per seed): misranking vs variance (and tail ratio for completeness).
    probe_rows: list[dict] = []
    for seed in seeds:
        p = LQRHeavyTailControlProblem(
            seed=int(seed),
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            horizon=int(args.horizon),
            bound=float(args.bound),
            rho_target=float(args.rho_target),
            u_max=float(args.u_max),
            init_std=float(args.init_std),
            noise_std=float(args.noise_std),
            noise_df=float(args.noise_df),
            noise_state_beta=float(args.noise_state_beta),
            noise_state_clip=float(args.noise_state_clip),
            eval_rollouts=int(args.eval_rollouts),
            eval_independent_noise=bool(args.eval_independent_noise),
            q_scale=float(args.q_scale),
            r_scale=float(args.r_scale),
            terminal_scale=float(args.terminal_scale),
            init_mode=str(args.init_mode),
            init_scale=float(args.init_scale),
        )
        # Use a generous max_evals so probes don't early-return None.
        rd = ms._misranking_probe(p, int(10**9))
        rel_sd = ms._variance_probe(p, int(10**9), reps=10)
        rd2, tail_ratio = ms._tail_ratio_probe(p, int(10**9), reps=2)
        probe_rows.append(
            {
                "seed": int(seed),
                "state_dim": int(state_dim),
                "action_dim": int(action_dim),
                "dimension": int(dim),
                "horizon": int(args.horizon),
                "budget": int(budget),
                "eval_rollouts": int(args.eval_rollouts),
                "noise_df": float(args.noise_df),
                "noise_std": float(args.noise_std),
                "noise_state_beta": float(args.noise_state_beta),
                "noise_state_clip": float(args.noise_state_clip),
                "misranking_rd": "" if rd is None else float(rd),
                "variance_rel_sd": "" if rel_sd is None else float(rel_sd),
                "tail_probe_rd": "" if rd2 is None else float(rd2),
                "tail_ratio": "" if tail_ratio is None else float(tail_ratio),
            }
        )
    write_csv(os.path.join(out_dir, "probe_values.csv"), probe_rows)

    # Run optimizers.
    futures = []
    results: list[RunResult] = []
    with ThreadPoolExecutor(max_workers=int(max(1, args.workers))) as ex:
        for algo in algos:
            for seed in seeds:
                futures.append(
                    ex.submit(
                        run_one,
                        algorithm=str(algo),
                        optimizer=algo_map[str(algo)],
                        seed=int(seed),
                        budget=int(budget),
                        state_dim=int(state_dim),
                        action_dim=int(action_dim),
                        horizon=int(args.horizon),
                        bound=float(args.bound),
                        rho_target=float(args.rho_target),
                        u_max=float(args.u_max),
                        init_std=float(args.init_std),
                        noise_std=float(args.noise_std),
                        noise_df=float(args.noise_df),
                        noise_state_beta=float(args.noise_state_beta),
                        noise_state_clip=float(args.noise_state_clip),
                        eval_rollouts=int(args.eval_rollouts),
                        post_rollouts=int(args.post_rollouts),
                        eval_independent_noise=bool(args.eval_independent_noise),
                        q_scale=float(args.q_scale),
                        r_scale=float(args.r_scale),
                        terminal_scale=float(args.terminal_scale),
                        init_mode=str(args.init_mode),
                        init_scale=float(args.init_scale),
                        postselect_k=int(args.postselect_k),
                    )
                )
        for fut in as_completed(futures):
            results.append(fut.result())

    results_sorted = sorted(results, key=lambda r: (r.algorithm, r.seed))
    runs_rows = [
        {
            "algorithm": r.algorithm,
            "seed": r.seed,
            "budget": r.budget,
            "evaluations": r.evaluations,
            "best_observed": r.best_observed,
            "post_mean_best_observed": r.post_mean_best_observed,
            "post_median_best_observed": r.post_median_best_observed,
            "post_cvar20_best_observed": r.post_cvar20_best_observed,
            "post_mean": r.post_mean,
            "post_median": r.post_median,
            "post_cvar20": r.post_cvar20,
            "postselect_k": r.postselect_k,
        }
        for r in results_sorted
    ]
    write_csv(os.path.join(out_dir, "runs.csv"), runs_rows)

    # Summary (median across seeds).
    by_algo: dict[str, list[RunResult]] = defaultdict(list)
    for r in results_sorted:
        by_algo[str(r.algorithm)].append(r)

    summary_rows = []
    for algo, rs in sorted(by_algo.items()):
        summary_rows.append(
            {
                "algorithm": str(algo),
                "n_runs": int(len(rs)),
                "median_post_mean": float(np.median([r.post_mean for r in rs])),
                "median_post_median": float(np.median([r.post_median for r in rs])),
                "median_post_cvar20": float(np.median([r.post_cvar20 for r in rs])),
            }
        )
    write_csv(os.path.join(out_dir, "summary.csv"), summary_rows)

    # Plot.
    values_by_algo = {algo: [r.post_mean for r in rs] for algo, rs in by_algo.items()}
    boxplot(
        out_path=os.path.join(out_dir, "final_boxplot.png"),
        values_by_algo=values_by_algo,
        title=f"LQR heavy-tail control | d={dim} (K) | T={int(args.horizon)} | budget={budget} | post_mean",
        ylabel="post-hoc mean cost (lower is better)",
    )

    print("Wrote:", repo_relpath(os.path.join(out_dir, "runs.csv")))
    print("Wrote:", repo_relpath(os.path.join(out_dir, "summary.csv")))
    print("Wrote:", repo_relpath(os.path.join(out_dir, "probe_values.csv")))
    print("Wrote:", repo_relpath(os.path.join(out_dir, "final_boxplot.png")))


if __name__ == "__main__":
    main()
