import argparse
import json
import pathlib
import pickle
import random
import time

import numpy as np
from joblib import Parallel, delayed

from mcts import MCTS
from tree_env import SyntheticTree


ALGORITHMS = ["uct", "ments", "rents", "tents", "dents", "bts", "varde"]

DEFAULT_KDS = [(16, 1), (200, 1), (14, 3), (16, 3), (16, 4), (200, 2)]

DEFAULT_GRID = {
    "uct": {"exploration_coeff": [0.1, 1.0, 2.0, 5.0, 10.0], "tau":[0.1]},
    "ments": {"exploration_coeff": [0.1, 0.5, 1.0], "tau": [0.05, 0.1, 0.2, 0.5, 1.0]},
    "rents": {"exploration_coeff": [0.1, 0.5, 1.0], "tau": [0.05, 0.1, 0.2, 0.5, 1.0]},
    "tents": {"exploration_coeff": [0.1, 0.5, 1.0], "tau": [0.05, 0.1, 0.2, 0.5, 1.0]},
    "dents": {"exploration_coeff": [0.25, 0.5, 0.75, 1.0], "tau": [0.1, 0.2, 0.5, 1.0]},
    "bts": {"exploration_coeff": [0.25, 0.5, 0.75, 1.0], "tau": [0.1, 0.2, 0.5, 1.0]},
    "varde": {
        "tau": [0.05, 0.1, 0.2, 0.5, 1.0],
        "variance_floor": [1e-4, 1e-3, 1e-2, 1e-1],
        "use_dp_value": [True, False],
    },
}

DEFAULT_EXPLORATION_COEFF = 1.0
DEFAULT_ALPHA = 1
DEFAULT_ATOMS = 10
DEFAULT_GAMMA = 1.0
DEFAULT_STEP_SIZE = 0.2


def parse_kds(text):
    if not text:
        return list(DEFAULT_KDS)
    out = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        k_str, d_str = part.split(",")
        out.append((int(k_str), int(d_str)))
    return out


def load_grid(path):
    if not path:
        return DEFAULT_GRID
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def score_series(series, mode):
    if mode == "final":
        return float(series[-1])
    return float(np.mean(series))


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Expected boolean value, got {value!r}.")


def prepare_tree(tree, tau):
    tree._tau = tau
    tree.optimal_v_root, tree.q_root = tree._solver()
    return tree


def run_single(tree_bytes, alg, params, n_simulations, seed):
    np.random.seed(seed)
    random.seed(seed)
    tree = pickle.loads(tree_bytes)
    prepare_tree(tree, params["tau"])

    exploration_coeff = params.get("exploration_coeff", DEFAULT_EXPLORATION_COEFF)
    mcts_kwargs = {
        "exploration_coeff": exploration_coeff,
        "algorithm": alg,
        "tau": params["tau"],
        "alpha": DEFAULT_ALPHA,
        "number_of_atoms": DEFAULT_ATOMS,
        "step_size": DEFAULT_STEP_SIZE,
        "gamma": DEFAULT_GAMMA,
        "update_type": "mean",
    }
    if alg == "varde":
        mcts_kwargs["variance_floor"] = params["variance_floor"]
        mcts_kwargs["use_dp_value"] = params["use_dp_value"]

    mcts = MCTS(**mcts_kwargs)

    v_hat, regret = mcts.run(tree, n_simulations)
    diff = np.abs(v_hat - tree.optimal_v_root)
    diff_uct = np.abs(v_hat - tree.max_mean)
    return diff, diff_uct, regret


def run_tasks(tasks, n_jobs):
    if n_jobs == 1:
        return [run_single(*task) for task in tasks]
    return Parallel(n_jobs=n_jobs)(delayed(run_single)(*task) for task in tasks)


def main():
    parser = argparse.ArgumentParser(description="Tune hyperparameters for selected MCTS algorithms.")
    parser.add_argument("--kds", type=str, default="", help="Semicolon-separated k,d pairs like '16,1;200,1'.")
    parser.add_argument("--algorithms", type=str, default=",".join(ALGORITHMS))
    parser.add_argument("--n-exp", type=int, default=3)
    parser.add_argument("--n-trees", type=int, default=3)
    parser.add_argument("--n-simulations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--score-mode", choices=["mean", "final"], default="mean")
    parser.add_argument("--objective", choices=["diff_uct", "diff", "regret"], default="diff_uct")
    parser.add_argument("--grid", type=str, default="", help="Path to JSON grid definition.")
    parser.add_argument("--output", type=str, default="logs/tuning_results.json")
    args = parser.parse_args()

    kds = parse_kds(args.kds)
    algorithms = [a.strip().lower() for a in args.algorithms.split(",") if a.strip()]
    for alg in algorithms:
        if alg not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {alg}")

    grid = load_grid(args.grid)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_params = {}
    all_scores = {}

    for alg_idx, alg in enumerate(algorithms):
        if alg not in grid:
            raise ValueError(f"Missing grid for algorithm '{alg}'.")

        alg_grid = grid[alg]
        tau_list = list(alg_grid.get("tau", []))
        if not tau_list:
            raise ValueError(f"Grid for '{alg}' must include a tau list.")

        if alg == "varde":
            variance_list = list(alg_grid.get("variance_floor", []))
            use_dp_list = list(alg_grid.get("use_dp_value", []))
            if not variance_list or not use_dp_list:
                raise ValueError(
                    "Grid for 'varde' must include variance_floor and use_dp_value lists."
                )
            total_cfgs = len(tau_list) * len(variance_list) * len(use_dp_list)
        else:
            c_list = list(alg_grid.get("exploration_coeff", []))
            if not c_list:
                raise ValueError(f"Grid for '{alg}' must include exploration_coeff list.")
            total_cfgs = len(c_list) * len(tau_list)

        print(f"Tuning {alg} with {total_cfgs} configurations...")

        tree_bytes_map = {}
        for kd_idx, (k, d) in enumerate(kds):
            trees = []
            for t in range(args.n_trees):
                tree_seed = args.seed + kd_idx * 1000 + t
                np.random.seed(tree_seed)
                random.seed(tree_seed)
                base_tree = SyntheticTree(
                    k=k,
                    d=d,
                    algorithm=alg,
                    tau=tau_list[0],
                    alpha=DEFAULT_ALPHA,
                    number_of_atom=DEFAULT_ATOMS,
                    gamma=DEFAULT_GAMMA,
                    step_size=DEFAULT_STEP_SIZE,
                )
                trees.append(pickle.dumps(base_tree, protocol=pickle.HIGHEST_PROTOCOL))
            tree_bytes_map[(k, d)] = trees

        scores = []
        best_score = None
        best_cfg = None

        if alg == "varde":
            variance_list = list(alg_grid.get("variance_floor", []))
            use_dp_list = list(alg_grid.get("use_dp_value", []))
            for tau in tau_list:
                for variance_floor in variance_list:
                    for use_dp_value in use_dp_list:
                        use_dp_bool = normalize_bool(use_dp_value)
                        params = {
                            "tau": float(tau),
                            "variance_floor": float(variance_floor),
                            "use_dp_value": use_dp_bool,
                        }
                        tasks = []
                        for kd_idx, (k, d) in enumerate(kds):
                            for tree_idx, tree_bytes in enumerate(tree_bytes_map[(k, d)]):
                                for exp_idx in range(args.n_exp):
                                    seed = (
                                        args.seed
                                        + alg_idx * 100000
                                        + kd_idx * 1000
                                        + tree_idx * 10
                                        + exp_idx
                                    )
                                    tasks.append((tree_bytes, alg, params, args.n_simulations, seed))

                        results = run_tasks(tasks, args.n_jobs)
                        metric_values = []
                        for diff, diff_uct, regret in results:
                            if args.objective == "diff_uct":
                                series = diff_uct
                            elif args.objective == "diff":
                                series = diff
                            else:
                                series = regret
                            metric_values.append(score_series(series, args.score_mode))

                        mean_score = float(np.mean(metric_values))
                        scores.append(
                            {
                                "tau": tau,
                                "variance_floor": variance_floor,
                                "use_dp_value": use_dp_bool,
                                "score": mean_score,
                            }
                        )
                        print(
                            f"  {alg}: tau={tau} variance_floor={variance_floor} "
                            f"use_dp_value={use_dp_bool} score={mean_score:.6f}"
                        )

                        if best_score is None or mean_score < best_score:
                            best_score = mean_score
                            best_cfg = {
                                "tau": float(tau),
                                "variance_floor": float(variance_floor),
                                "use_dp_value": use_dp_bool,
                            }
        else:
            c_list = list(alg_grid.get("exploration_coeff", []))
            for c in c_list:
                for tau in tau_list:
                    params = {"exploration_coeff": float(c), "tau": float(tau)}
                    tasks = []
                    for kd_idx, (k, d) in enumerate(kds):
                        for tree_idx, tree_bytes in enumerate(tree_bytes_map[(k, d)]):
                            for exp_idx in range(args.n_exp):
                                seed = (
                                    args.seed
                                    + alg_idx * 100000
                                    + kd_idx * 1000
                                    + tree_idx * 10
                                    + exp_idx
                                )
                                tasks.append((tree_bytes, alg, params, args.n_simulations, seed))

                    results = run_tasks(tasks, args.n_jobs)
                    metric_values = []
                    for diff, diff_uct, regret in results:
                        if args.objective == "diff_uct":
                            series = diff_uct
                        elif args.objective == "diff":
                            series = diff
                        else:
                            series = regret
                        metric_values.append(score_series(series, args.score_mode))

                    mean_score = float(np.mean(metric_values))
                    scores.append({"exploration_coeff": c, "tau": tau, "score": mean_score})
                    print(f"  {alg}: C={c} tau={tau} score={mean_score:.6f}")

                    if best_score is None or mean_score < best_score:
                        best_score = mean_score
                        best_cfg = {"exploration_coeff": float(c), "tau": float(tau)}

        best_params[alg] = best_cfg
        all_scores[alg] = scores

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kds": [{"k": k, "d": d} for k, d in kds],
        "n_exp": args.n_exp,
        "n_trees": args.n_trees,
        "n_simulations": args.n_simulations,
        "objective": args.objective,
        "score_mode": args.score_mode,
        "best_params": best_params,
        "scores": all_scores,
        "grid": grid,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Saved tuning results to {output_path}")


if __name__ == "__main__":
    main()
