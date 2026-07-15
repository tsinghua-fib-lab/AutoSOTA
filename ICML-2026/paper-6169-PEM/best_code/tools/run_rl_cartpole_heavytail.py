#!/usr/bin/env python3
"""
External validity benchmark (standard RL-style objective):

Noisy policy search on CartPole with heavy-tailed disturbances.

We treat policy optimization as black-box optimization:
  - decision variable: parameters of a small MLP policy,
  - objective call: one (noisy) episode return -> converted to a minimization loss,
  - post-hoc evaluation: many fresh episodes (cleaner metric).

This is designed to stress the same qualitative point as the COCO fixed-budget evidence:
under a fixed episode budget, aggressive resampling burns evaluations and slows progress,
while selection-stage uncertainty integration / noise-aware switching can be more sample-efficient.
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
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero as berw_hetero,
)
from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_robust as berw_hetero_robust,
)
from algorithms.probe_switch import (
    my_optimizer_noise_probe_switch_warmstart as probeswitch_noise_warmstart,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t012 as probeswitch_mr_t012,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t022 as probeswitch_mr_t022,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_robust_t012 as probeswitch_mr_robust_t012,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_robust_t022 as probeswitch_mr_robust_t022,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_robust_t019 as probeswitch_mr_robust_t019,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_robust_t021 as probeswitch_mr_robust_t021,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t019 as probeswitch_mr_t019,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t021 as probeswitch_mr_t021,
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


def cvar(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= 0:
        return float("nan")
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    k = int(max(1, math.ceil(alpha * values.size)))
    worst = np.sort(values)[-k:]
    return float(np.mean(worst))


class CartPoleEnv:
    # Classic CartPole parameters (Gym-style).
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masscart + masspole
    length = 0.5  # actually half the pole's length
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02
    x_threshold_default = 2.4
    theta_threshold_radians_default = 12.0 * math.pi / 180.0

    def __init__(
        self,
        *,
        rng: np.random.RandomState,
        max_steps: int,
        force_noise_std: float,
        noise_df: float,
        x_threshold: float,
        theta_threshold_radians: float,
    ):
        self.rng = rng
        self.max_steps = int(max_steps)
        self.force_noise_std = float(max(0.0, force_noise_std))
        self.noise_df = float(max(2.05, noise_df))
        self.x_threshold = float(x_threshold)
        self.theta_threshold_radians = float(theta_threshold_radians)
        self.state = np.zeros(4, dtype=float)
        self.steps = 0

    def reset(self) -> np.ndarray:
        self.state = self.rng.uniform(low=-0.05, high=0.05, size=(4,)).astype(float)
        self.steps = 0
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        x, x_dot, theta, theta_dot = self.state.tolist()
        force = self.force_mag if int(action) == 1 else -self.force_mag
        if self.force_noise_std > 0.0:
            force += float(self.force_noise_std) * float(self.rng.standard_t(self.noise_df))

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.asarray([x, x_dot, theta, theta_dot], dtype=float)
        self.steps += 1

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.steps >= self.max_steps
        )
        reward = 1.0  # per-step reward (survival)
        return self.state.copy(), float(reward), done


def unpack_policy(theta: np.ndarray, *, hidden_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    theta = np.asarray(theta, dtype=float).reshape((-1,))
    h = int(hidden_dim)
    # params: W1(h,4), b1(h), W2(1,h), b2(1)
    idx = 0
    w1 = theta[idx : idx + h * 4].reshape((h, 4))
    idx += h * 4
    b1 = theta[idx : idx + h].reshape((h,))
    idx += h
    w2 = theta[idx : idx + h].reshape((h,))
    idx += h
    b2 = float(theta[idx]) if idx < theta.size else 0.0
    return w1, b1, w2, b2


def policy_action(state: np.ndarray, theta: np.ndarray, *, hidden_dim: int) -> int:
    w1, b1, w2, b2 = unpack_policy(theta, hidden_dim=int(hidden_dim))
    s = np.asarray(state, dtype=float).reshape((4,))
    h = np.tanh(w1 @ s + b1)
    logit = float(np.dot(w2, h) + b2)
    return 1 if logit > 0.0 else 0


class CartPoleHeavyTailProblem:
    """
    Black-box optimization interface.

    Objective call: one episode => loss = (max_steps - return).
    """

    def __init__(
        self,
        *,
        seed: int,
        hidden_dim: int,
        max_steps: int,
        bound: float,
        force_noise_std: float,
        noise_df: float,
        objective: str,
        x_threshold: float,
        theta_threshold_deg: float,
        terminate_penalty: float,
        eval_independent_noise: bool,
        id_function: int = 9300,
    ):
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        self.bound = float(bound)
        self.force_noise_std = float(force_noise_std)
        self.noise_df = float(noise_df)
        self.objective = str(objective).strip().lower()
        if self.objective not in {"survival", "quadratic_cost"}:
            raise ValueError(f"Unknown objective: {objective}")
        self.x_threshold = float(x_threshold)
        self.theta_threshold_radians = float(theta_threshold_deg) * math.pi / 180.0
        # In quadratic_cost mode, early termination is penalized by:
        #   penalty_per_missing_step * (max_steps - steps_survived)
        self.terminate_penalty = float(max(0.0, terminate_penalty))
        self._eval_independent_noise = bool(eval_independent_noise)

        self.dimension = int(self.hidden_dim * 6 + 1)
        self.lower_bounds = -self.bound * np.ones(self.dimension, dtype=float)
        self.upper_bounds = self.bound * np.ones(self.dimension, dtype=float)
        self.initial_solution = np.zeros(self.dimension, dtype=float)

        self.id_function = int(id_function)
        self.id_instance = int(seed)

        self._base_seed = int(seed) & 0xFFFFFFFF
        self._rng = np.random.RandomState(int(self._base_seed) ^ 0xC0FFEE)

        self.evaluations = 0
        self.final_target_hit = False
        self.best_observed_fvalue1 = float("inf")
        self.best_x: np.ndarray | None = None

    def _rng_eval(self, eval_id: int) -> np.random.RandomState:
        if not self._eval_independent_noise:
            return self._rng
        seed_eval = (int(self._base_seed) * 1000003 + int(eval_id) * 9176 + 4040) & 0xFFFFFFFF
        return np.random.RandomState(int(seed_eval))

    def episode_return(self, theta: np.ndarray, *, rng: np.random.RandomState) -> float:
        env = CartPoleEnv(
            rng=rng,
            max_steps=int(self.max_steps),
            force_noise_std=float(self.force_noise_std),
            noise_df=float(self.noise_df),
            x_threshold=float(self.x_threshold),
            theta_threshold_radians=float(self.theta_threshold_radians),
        )
        state = env.reset()
        total = 0.0
        done = False
        while not done:
            a = policy_action(state, theta, hidden_dim=int(self.hidden_dim))
            state, r, done = env.step(int(a))
            total += float(r)
        return float(total)

    def episode_objective(self, theta: np.ndarray, *, rng: np.random.RandomState) -> tuple[float, float]:
        """
        Returns (objective_value, return_value).
        - survival: objective = max_steps - return (minimize)
        - quadratic_cost: objective = sum_t cost(state, action) + terminal penalty if terminated early
        """

        env = CartPoleEnv(
            rng=rng,
            max_steps=int(self.max_steps),
            force_noise_std=float(self.force_noise_std),
            noise_df=float(self.noise_df),
            x_threshold=float(self.x_threshold),
            theta_threshold_radians=float(self.theta_threshold_radians),
        )
        state = env.reset()

        if self.objective == "survival":
            total = 0.0
            done = False
            while not done:
                a = policy_action(state, theta, hidden_dim=int(self.hidden_dim))
                state, r, done = env.step(int(a))
                total += float(r)
            ret = float(total)
            obj = float(self.max_steps) - ret
            return obj, ret

        # quadratic_cost: continuous control-style objective
        cost = 0.0
        done = False
        steps = 0
        while not done and steps < int(self.max_steps):
            a = policy_action(state, theta, hidden_dim=int(self.hidden_dim))
            next_state, _r, done = env.step(int(a))
            x, x_dot, theta_s, theta_dot = state.tolist()
            u = 1.0 if int(a) == 1 else -1.0
            # A smooth quadratic cost (continuous, no ties).
            cost += float(x * x + 0.1 * x_dot * x_dot + 10.0 * theta_s * theta_s + 0.1 * theta_dot * theta_dot + 0.001 * u * u)
            state = next_state
            steps += 1
        if done and steps < int(self.max_steps):
            # Penalize early termination proportionally to remaining horizon.
            cost += float(self.terminate_penalty) * float(int(self.max_steps) - int(steps))
        ret = -float(cost)
        return float(cost), float(ret)

    def post_eval(self, theta: np.ndarray, *, rng: np.random.RandomState, episodes: int) -> dict[str, float]:
        objs = np.empty(int(episodes), dtype=float)
        rets = np.empty(int(episodes), dtype=float)
        for i in range(int(episodes)):
            obj, ret = self.episode_objective(theta, rng=rng)
            objs[i] = float(obj)
            rets[i] = float(ret)
        return {
            "obj_mean": float(np.mean(objs)),
            "obj_median": float(np.median(objs)),
            "obj_cvar20": float(cvar(objs, 0.2)),
            "return_mean": float(np.mean(rets)),
            "return_median": float(np.median(rets)),
        }

    def __call__(self, theta: np.ndarray) -> float:
        self.evaluations += 1
        eval_id = int(self.evaluations)
        theta = np.clip(np.asarray(theta, dtype=float), self.lower_bounds, self.upper_bounds)
        rng = self._rng_eval(eval_id)
        obj, _ret = self.episode_objective(theta, rng=rng)
        if float(obj) < self.best_observed_fvalue1:
            self.best_observed_fvalue1 = float(obj)
            self.best_x = theta.copy()
        return float(obj)


@dataclass(frozen=True)
class RunResult:
    algorithm: str
    seed: int
    max_evals: int
    evaluations: int
    best_noisy: float
    post_true: float
    post_return_mean: float
    post_loss_median: float
    post_loss_cvar20: float


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


def run_one(
    *,
    algorithm: str,
    optimizer,
    seed: int,
    max_evals: int,
    hidden_dim: int,
    max_steps: int,
    bound: float,
    force_noise_std: float,
    noise_df: float,
    objective: str,
    x_threshold: float,
    theta_threshold_deg: float,
    terminate_penalty: float,
    eval_independent_noise: bool,
    post_episodes: int,
    postselect_k: int,
) -> RunResult:
    base_problem = CartPoleHeavyTailProblem(
        seed=int(seed),
        hidden_dim=int(hidden_dim),
        max_steps=int(max_steps),
        bound=float(bound),
        force_noise_std=float(force_noise_std),
        noise_df=float(noise_df),
        objective=str(objective),
        x_threshold=float(x_threshold),
        theta_threshold_deg=float(theta_threshold_deg),
        terminate_penalty=float(terminate_penalty),
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

    # Deduplicate candidates.
    uniq: list[np.ndarray] = []
    seen: set[bytes] = set()
    for x in cand:
        key = np.round(x, 6).astype(np.float32, copy=False).tobytes()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(x)

    best_stats = None
    for x in uniq:
        # Common RNG across candidates for stable post-selection.
        rng = np.random.RandomState((int(seed) * 1000003 + 777) & 0xFFFFFFFF)
        stats = base_problem.post_eval(x, rng=rng, episodes=int(post_episodes))
        if best_stats is None or float(stats["obj_mean"]) < float(best_stats["obj_mean"]):
            best_stats = stats
    if best_stats is None:
        best_stats = base_problem.post_eval(
            base_problem.initial_solution,
            rng=np.random.RandomState(int(seed) ^ 0xABC),
            episodes=int(post_episodes),
        )

    return RunResult(
        algorithm=str(algorithm),
        seed=int(seed),
        max_evals=int(max_evals),
        evaluations=int(base_problem.evaluations),
        best_noisy=float(base_problem.best_observed_fvalue1),
        post_true=float(best_stats["obj_mean"]),
        post_return_mean=float(best_stats["return_mean"]),
        post_loss_median=float(best_stats["obj_median"]),
        post_loss_cvar20=float(best_stats["obj_cvar20"]),
    )


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            for r in rows:
                w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--seeds", default="1-10")
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--hidden-dim", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--budget-mult", type=int, default=10, help="Budget multiplier × policy_dim.")
    parser.add_argument("--bound", type=float, default=2.0)

    parser.add_argument("--force-noise-std", type=float, default=3.0)
    parser.add_argument("--noise-df", type=float, default=3.0)
    parser.add_argument("--objective", default="survival", choices=["survival", "quadratic_cost"])
    parser.add_argument("--x-threshold", type=float, default=CartPoleEnv.x_threshold_default)
    parser.add_argument("--theta-threshold-deg", type=float, default=12.0)
    parser.add_argument(
        "--terminate-penalty",
        type=float,
        default=5.0,
        help="quadratic_cost only: penalty per missing step when the episode terminates early.",
    )
    parser.add_argument("--eval-independent-noise", action="store_true")

    parser.add_argument("--post-episodes", type=int, default=64)
    parser.add_argument("--postselect-k", type=int, default=10)

    parser.add_argument(
        "--algorithms",
        default="CMA-ES-sep,CMA-ES-Resample(k=5),CMA-ES-Resample(k=10),BERW-HeteroRobust,ProbeSwitch-Noise-Warmstart,ProbeSwitch-MR(t=0.12)",
    )
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    seeds = parse_int_list(str(args.seeds))
    hidden_dim = int(args.hidden_dim)
    dim = int(hidden_dim * 6 + 1)
    max_evals = int(args.budget_mult) * int(dim)

    algo_map = {
        "CMA-ES-sep": cmaes_sep,
        "CMA-ES-Resample(k=5)": my_optimizer_resample5,
        "CMA-ES-Resample(k=10)": my_optimizer_resample10,
        "BERW-Hetero": berw_hetero,
        "BERW-HeteroRobust": berw_hetero_robust,
        "ProbeSwitch-Noise-Warmstart": probeswitch_noise_warmstart,
        "ProbeSwitch-MR(t=0.12)": probeswitch_mr_t012,
        "ProbeSwitch-MR(t=0.19)": probeswitch_mr_t019,
        "ProbeSwitch-MR(t=0.21)": probeswitch_mr_t021,
        "ProbeSwitch-MR(t=0.22)": probeswitch_mr_t022,
        "ProbeSwitch-MR-Robust(t=0.12)": probeswitch_mr_robust_t012,
        "ProbeSwitch-MR-Robust(t=0.19)": probeswitch_mr_robust_t019,
        "ProbeSwitch-MR-Robust(t=0.21)": probeswitch_mr_robust_t021,
        "ProbeSwitch-MR-Robust(t=0.22)": probeswitch_mr_robust_t022,
        "UH-CMA-ES(maxevals=30)": uh_cmaes_maxevals30,
    }

    algos = [a.strip() for a in str(args.algorithms).split(",") if a.strip()]
    for a in algos:
        if a not in algo_map:
            raise SystemExit(f"Unknown algorithm: {a} (known: {sorted(algo_map.keys())})")

    out_dir = os.path.abspath(str(args.results_dir))
    os.makedirs(out_dir, exist_ok=True)

    # Probe values (per seed, at x0) for documentation / threshold-transfer studies.
    probe_rows: list[dict[str, object]] = []
    for seed in seeds:
        p = CartPoleHeavyTailProblem(
            seed=int(seed),
            hidden_dim=int(hidden_dim),
            max_steps=int(args.max_steps),
            bound=float(args.bound),
            force_noise_std=float(args.force_noise_std),
            noise_df=float(args.noise_df),
            objective=str(args.objective),
            x_threshold=float(args.x_threshold),
            theta_threshold_deg=float(args.theta_threshold_deg),
            terminate_penalty=float(args.terminate_penalty),
            eval_independent_noise=bool(args.eval_independent_noise),
        )
        rd = ms._misranking_probe(p, max_evals=10**9)
        rd2, tail_ratio = ms._tail_ratio_probe(p, max_evals=10**9, reps=2)
        rel_sd = ms._variance_probe(p, max_evals=10**9, reps=10)
        probe_rows.append(
            {
                "seed": int(seed),
                "dim": int(dim),
                "hidden_dim": int(hidden_dim),
                "max_steps": int(args.max_steps),
                "force_noise_std": float(args.force_noise_std),
                "noise_df": float(args.noise_df),
                "objective": str(args.objective),
                "x_threshold": float(args.x_threshold),
                "theta_threshold_deg": float(args.theta_threshold_deg),
                "terminate_penalty": float(args.terminate_penalty),
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
                        hidden_dim=int(hidden_dim),
                        max_steps=int(args.max_steps),
                        bound=float(args.bound),
                        force_noise_std=float(args.force_noise_std),
                        noise_df=float(args.noise_df),
                        objective=str(args.objective),
                        x_threshold=float(args.x_threshold),
                        theta_threshold_deg=float(args.theta_threshold_deg),
                        terminate_penalty=float(args.terminate_penalty),
                        eval_independent_noise=bool(args.eval_independent_noise),
                        post_episodes=int(args.post_episodes),
                        postselect_k=int(args.postselect_k),
                    )
                )
        for fut in as_completed(futs):
            runs.append(fut.result())

    runs_sorted = sorted(runs, key=lambda r: (r.algorithm, r.seed))
    runs_csv = os.path.join(out_dir, "runs.csv")
    write_csv(runs_csv, [r.__dict__ for r in runs_sorted])

    # Summary (median across seeds, per algorithm).
    by_algo: dict[str, list[RunResult]] = {}
    for r in runs_sorted:
        by_algo.setdefault(r.algorithm, []).append(r)
    summary_rows: list[dict[str, object]] = []
    for algo, rs in sorted(by_algo.items(), key=lambda t: t[0]):
        post_true = np.asarray([x.post_true for x in rs], dtype=float)
        post_ret = np.asarray([x.post_return_mean for x in rs], dtype=float)
        summary_rows.append(
            {
                "algorithm": str(algo),
                "n_runs": int(len(rs)),
                "median_post_true": float(np.median(post_true)),
                "median_post_return": float(np.median(post_ret)),
            }
        )
    write_csv(os.path.join(out_dir, "summary.csv"), summary_rows)

    # Boxplot of post_true (loss).
    vals = {algo: [x.post_true for x in rs] for algo, rs in by_algo.items()}
    boxplot(
        out_path=os.path.join(out_dir, "final_boxplot.png"),
        values_by_algo=vals,
        title=f"CartPole heavy-tail ({str(args.objective)}) | fixed budget={max_evals} episodes | hidden_dim={hidden_dim} (d={dim})",
        ylabel="post_true (objective mean across post episodes)  [lower is better]",
    )

    print("Wrote:", repo_relpath(runs_csv))
    print("Wrote:", repo_relpath(os.path.join(out_dir, "summary.csv")))
    print("Wrote:", repo_relpath(os.path.join(out_dir, "probe_values.csv")))
    print("Wrote:", repo_relpath(os.path.join(out_dir, "final_boxplot.png")))


if __name__ == "__main__":
    main()
