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

DEFAULT_PARAMS = {
    "uct": {"exploration_coeff": 0.1, "tau": 0.1},
    "ments": {"exploration_coeff": 0.5, "tau": 0.1},
    "rents": {"exploration_coeff": 0.5, "tau": 0.2},
    "tents": {"exploration_coeff": 0.5, "tau": 0.5},
    "dents": {"exploration_coeff": 0.5, "tau": 0.1},
    "bts": {"exploration_coeff": 0.25, "tau": 0.1},
    "varde": {"exploration_coeff": 1.0, "tau": 0.1, "variance_floor": 0.001, "use_dp_value": False},
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

    mcts_kwargs = {
        "exploration_coeff": params["exploration_coeff"],
        "algorithm": alg,
        "tau": params["tau"],
        "alpha": DEFAULT_ALPHA,
        "number_of_atoms": DEFAULT_ATOMS,
        "step_size": DEFAULT_STEP_SIZE,
        "gamma": DEFAULT_GAMMA,
        "update_type": "mean",
    }
    if alg == "varde":
        mcts_kwargs["variance_floor"] = params.get("variance_floor", 0.0)
        mcts_kwargs["use_dp_value"] = params.get("use_dp_value", True)

    mcts = MCTS(**mcts_kwargs)

    v_hat, regret = mcts.run(tree, n_simulations)
    diff = np.abs(v_hat - tree.optimal_v_root)
    diff_uct = np.abs(v_hat - tree.max_mean)
    return diff, diff_uct, regret


def run_tasks(tasks, n_jobs):
    if n_jobs == 1:
        return [run_single(*task) for task in tasks]
    return Parallel(n_jobs=n_jobs)(delayed(run_single)(*task) for task in tasks)


def load_tuning_params(path, algorithms):
    tuning_path = pathlib.Path(path) if path else None
    if not tuning_path or not tuning_path.exists():
        return {alg: dict(DEFAULT_PARAMS[alg]) for alg in algorithms}

    with open(tuning_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    best = data.get("best_params", {})
    params = {}
    for alg in algorithms:
        if alg not in best:
            raise ValueError(
                f"Missing best params for {alg} in tuning results. "
                f"Either tune it or restrict --algorithms."
            )
        tuned = dict(best.get(alg, {}))
        if alg == "varde":
            if "tau" not in tuned or "variance_floor" not in tuned or "use_dp_value" not in tuned:
                raise ValueError("VarDE tuning results must include tau, variance_floor, use_dp_value.")
            tuned["exploration_coeff"] = float(tuned.get("exploration_coeff", DEFAULT_EXPLORATION_COEFF))
            tuned["variance_floor"] = float(tuned["variance_floor"])
            tuned["use_dp_value"] = normalize_bool(tuned["use_dp_value"])
        else:
            if "exploration_coeff" not in tuned or "tau" not in tuned:
                raise ValueError(f"Missing exploration_coeff or tau for {alg} in tuning results.")
        params[alg] = tuned
    return params


def main():
    parser = argparse.ArgumentParser(description="Run MCTS experiments for selected algorithms.")
    parser.add_argument("--kds", type=str, default="", help="Semicolon-separated k,d pairs like '16,1;200,1'.")
    parser.add_argument("--algorithms", type=str, default=",".join(ALGORITHMS))
    parser.add_argument("--n-exp", type=int, default=5)
    parser.add_argument("--n-trees", type=int, default=5)
    parser.add_argument("--n-simulations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--tuning-results", type=str, default="logs/tuning_results.json")
    parser.add_argument("--run-dir", type=str, default="")
    args = parser.parse_args()

    kds = parse_kds(args.kds)
    algorithms = [a.strip().lower() for a in args.algorithms.split(",") if a.strip()]
    for alg in algorithms:
        if alg not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {alg}")

    params_by_alg = load_tuning_params(args.tuning_results, algorithms)

    if args.run_dir:
        run_dir = pathlib.Path(args.run_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = pathlib.Path("logs") / "runs" / f"run_{timestamp}"

    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory {run_dir} already exists and is not empty.")

    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    for kd_idx, (k, d) in enumerate(kds):
        (run_dir / "results" / f"k_{k}_d_{d}").mkdir(parents=True, exist_ok=True)

    for alg_idx, alg in enumerate(algorithms):
        params = params_by_alg[alg]
        print(f"Running {alg} with params {params}...")

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
                    tau=params["tau"],
                    alpha=DEFAULT_ALPHA,
                    number_of_atom=DEFAULT_ATOMS,
                    gamma=DEFAULT_GAMMA,
                    step_size=DEFAULT_STEP_SIZE,
                )
                trees.append(pickle.dumps(base_tree, protocol=pickle.HIGHEST_PROTOCOL))
            tree_bytes_map[(k, d)] = trees

        for kd_idx, (k, d) in enumerate(kds):
            tasks = []
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
            diff = np.array([r[0] for r in results])
            diff_uct = np.array([r[1] for r in results])
            regret = np.array([r[2] for r in results])

            out_dir = run_dir / "results" / f"k_{k}_d_{d}"
            np.save(out_dir / f"diff_{alg}.npy", diff)
            np.save(out_dir / f"diff_uct_{alg}.npy", diff_uct)
            np.save(out_dir / f"regret_{alg}.npy", regret)

    config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kds": [{"k": k, "d": d} for k, d in kds],
        "n_exp": args.n_exp,
        "n_trees": args.n_trees,
        "n_simulations": args.n_simulations,
        "algorithms": algorithms,
        "params": params_by_alg,
        "gamma": DEFAULT_GAMMA,
        "step_size": DEFAULT_STEP_SIZE,
        "alpha": DEFAULT_ALPHA,
        "number_of_atoms": DEFAULT_ATOMS,
        "seed": args.seed,
    }

    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
