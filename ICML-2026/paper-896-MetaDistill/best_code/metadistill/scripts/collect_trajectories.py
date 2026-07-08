#!/usr/bin/env python3
"""Collect teacher trajectories for MetaDistill training."""

import argparse
import os
import pickle
import random
import sys
from typing import Iterable, Optional

import torch
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from optimizers import BBORunner, CMAES, DE, GA, JADE, LSHADE, PSO, SHADE
from tasks import TaskProblem, cec
from tasks.utils import gen_tf_offsets, set_tf_offset


def _to_distribution_params(pop, runner):
    params = runner.algo.get_distr_param()
    if params:
        return {
            "mean": params["mean"].detach().cpu().numpy(),
            "cov_mat": params["cov_mat"].detach().cpu().numpy(),
        }
    return {
        "mean": pop[..., 1:].mean(dim=1).detach().cpu().numpy(),
        "cov_mat": torch.cov(pop[..., 1:].view(-1, pop.shape[-1] - 1).T)
        .detach()
        .cpu()
        .numpy(),
    }


def collect_trajectories(
    runners: Iterable[BBORunner],
    n_epochs: int,
    n_generations: int,
    batchsize: int,
    popsize: int,
    problemdim: int,
    tf_set,
    trajectory_name: str,
):
    task = TaskProblem(fun=None, repaire=True, dim=problemdim)
    trajectories = []

    with torch.no_grad():
        for epoch in range(int(n_epochs)):
            epoch_trajectory = {}
            for func in tf_set:
                offsets = gen_tf_offsets(dim=problemdim, fun=func, epoch=1)
                if offsets is None or len(offsets) == 0:
                    raise ValueError(f"Function {func['fid']} does not support offsets.")
                offset = offsets[0]
                set_tf_offset(fun=func, offset=offset)

                epoch_trajectory[func["fid"]] = {"offset": offset, "algos": {}}

                task.setfun(func)
                pop = task.genRandomPop([batchsize, popsize, problemdim])
                pop, _ = task.calfitness(pop)
                last_pop = {runner.algo.name: pop.detach() for runner in runners}

                for runner in runners:
                    epoch_trajectory[func["fid"]]["algos"][runner.algo.name] = {
                        "pop": [],
                        "distr_params": [],
                    }
                    runner.reset(problem=task, problemdim=problemdim)

                    gen_bar = tqdm(range(int(n_generations)), ncols=100)
                    for g in gen_bar:
                        gen_bar.set_description(
                            f"Epoch{epoch}, {func['fid']}, {runner.algo.name}, Gen{g}"
                        )
                        pop = last_pop[runner.algo.name].detach()
                        pop, _ = runner.step(pop)
                        last_pop[runner.algo.name] = pop.detach()

                        algo_trajectory = epoch_trajectory[func["fid"]]["algos"][
                            runner.algo.name
                        ]
                        algo_trajectory["pop"].append(pop.detach().cpu().numpy())
                        algo_trajectory["distr_params"].append(
                            _to_distribution_params(pop, runner)
                        )

            trajectories.append(epoch_trajectory)

    os.makedirs("data/teacher", exist_ok=True)
    with open(f"data/teacher/{trajectory_name}.pkl", "wb") as fp:
        pickle.dump(trajectories, fp)


def select_trajectories(
    path: str,
    interval: int,
    trajectory_name: str,
    n_generations: int,
    *,
    mode: str = "best",
    top_k: int = 2,
    p_best: float = 0.5,
    seed: Optional[int] = None,
):
    mode = str(mode).lower()
    if mode not in {"best", "random", "topk", "mix"}:
        raise ValueError(f"Invalid mode: {mode}")

    rng = random.Random(int(seed) if seed is not None else None)

    with open(path, "rb") as fp:
        trajectories = pickle.load(fp)

    selected_trajectories = []
    choice_counts = {}
    for trajectory in trajectories:
        epoch_trajectory = {}
        for fid, func_trajectory in trajectory.items():
            epoch_trajectory[fid] = {
                "offset": func_trajectory["offset"],
                "pop": [],
                "distr_params": [],
            }

            prev_end = 0
            while prev_end < n_generations:
                end = min(prev_end + interval, n_generations)
                scores = {}
                for algo_name, algo_trajectory in func_trajectory["algos"].items():
                    pop = algo_trajectory["pop"][end - 1]
                    scores[algo_name] = torch.mean(
                        torch.min(torch.from_numpy(pop)[..., 0], dim=-1)[0], dim=-1
                    )

                ranked_algos = sorted(scores.keys(), key=lambda name: scores[name])
                if mode == "best":
                    picked = ranked_algos[0]
                elif mode == "random":
                    picked = rng.choice(ranked_algos)
                else:
                    top = ranked_algos[: min(int(top_k), len(ranked_algos))]
                    picked = rng.choice(top) if mode == "topk" else ranked_algos[0]
                    if mode == "mix" and rng.random() >= float(p_best):
                        picked = rng.choice(top)

                choice_counts[picked] = int(choice_counts.get(picked, 0)) + 1
                picked_trajectory = func_trajectory["algos"][picked]
                epoch_trajectory[fid]["pop"] += picked_trajectory["pop"][prev_end:end]
                epoch_trajectory[fid]["distr_params"] += picked_trajectory[
                    "distr_params"
                ][prev_end:end]
                prev_end = end

        selected_trajectories.append(epoch_trajectory)

    os.makedirs("data/teacher", exist_ok=True)
    with open(f"data/teacher/{trajectory_name}.pkl", "wb") as fp:
        pickle.dump(selected_trajectories, fp)

    print(
        f"[select_trajectories] mode={mode} top_k={top_k} p_best={p_best} "
        f"seed={seed} choice_counts={choice_counts}"
    )


def _build_teacher_runners(popsize: int, dim: int, generations: int):
    return [
        BBORunner(
            algo=CMAES(
                popsize=popsize,
                problemdim=dim,
                generations=generations,
                bounds=[0, 1],
            )
        ),
        BBORunner(
            algo=PSO(
                config={
                    "popsize": popsize,
                    "problemdim": dim,
                    "minimize": True,
                    "w": 0.7298,
                    "c1": 1.49618,
                    "c2": 1.49618,
                    "v_max_ratio": 0.2,
                    "v_init_ratio": 0.1,
                    "zero_v_on_clip": True,
                    "assume_fitness": True,
                }
            )
        ),
        BBORunner(
            algo=GA(
                config={
                    "popsize": popsize,
                    "problemdim": dim,
                    "minimize": True,
                    "tournament_k": 3,
                    "pc": 0.9,
                    "eta_c": 15.0,
                    "pm": 1.0 / max(1, dim),
                    "eta_m": 20.0,
                    "assume_fitness": True,
                }
            )
        ),
        BBORunner(
            algo=DE(
                config={
                    "F": 0.5,
                    "CR": 0.5,
                    "popsize": popsize,
                    "problemdim": dim,
                    "minimize": True,
                }
            )
        ),
        BBORunner(
            algo=JADE(
                config={
                    "popsize": popsize,
                    "problemdim": dim,
                    "minimize": True,
                    "c": 0.1,
                    "p": 0.05,
                    "mu_F": 0.5,
                    "mu_CR": 0.5,
                }
            )
        ),
        BBORunner(
            algo=SHADE(
                config={
                    "popsize": popsize,
                    "problemdim": dim,
                    "minimize": True,
                    "p": 0.05,
                    "H": 20,
                    "archive_size": popsize,
                }
            )
        ),
        BBORunner(
            algo=LSHADE(
                config={
                    "popsize": popsize,
                    "problemdim": dim,
                    "minimize": True,
                    "p": 0.05,
                    "H": 20,
                    "archive_factor": 1.0,
                    "max_fe": popsize * generations,
                }
            )
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--popsize", type=int, default=200)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--out-prefix", type=str, default="teacher_cec2-6")
    parser.add_argument(
        "--select-mode",
        type=str,
        default="best",
        choices=["best", "random", "topk", "mix"],
    )
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--p-best", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dim = int(args.dim)
    popsize = int(args.popsize)
    generations = int(args.generations)
    tf_set = [cec.FUNCTIONS[f"cecf{i}"] for i in range(2, 7)]
    runners = _build_teacher_runners(popsize=popsize, dim=dim, generations=generations)

    raw_name = f"{args.out_prefix}_raw_d{dim}_pop{popsize}_gen{generations}_7t_e{args.epochs}"
    selected_name = (
        f"{args.out_prefix}_sel_itvl{args.interval}_d{dim}_pop{popsize}_"
        f"gen{generations}_7t_e{args.epochs}"
    )

    collect_trajectories(
        runners=runners,
        n_epochs=args.epochs,
        n_generations=generations,
        batchsize=args.batchsize,
        popsize=popsize,
        problemdim=dim,
        tf_set=tf_set,
        trajectory_name=raw_name,
    )
    select_trajectories(
        path=f"data/teacher/{raw_name}.pkl",
        interval=args.interval,
        trajectory_name=selected_name,
        n_generations=generations,
        mode=args.select_mode,
        top_k=args.top_k,
        p_best=args.p_best,
        seed=args.seed,
    )

    print(f"[saved] raw: data/teacher/{raw_name}.pkl")
    print(f"[saved] selected: data/teacher/{selected_name}.pkl")


if __name__ == "__main__":
    main()
