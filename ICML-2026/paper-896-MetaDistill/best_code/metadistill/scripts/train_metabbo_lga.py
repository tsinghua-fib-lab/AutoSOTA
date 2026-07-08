#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from optimizers.lga import LGA
from optimizers.cmaes import CMAES
from tasks.problem import MetaTask
from tasks import cec


def _set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_cec_funcs(fids: List[int]) -> List[Dict]:
    funcs: List[Dict] = []
    for fid in fids:
        funcs.append(dict(cec.FUNCTIONS[f"cecf{int(fid)}"]))
    return funcs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--train-fids", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--problemdim", type=int, default=10)
    p.add_argument("--inner-popsize", type=int, default=200)
    p.add_argument("--inner-generations", type=int, default=100)

    p.add_argument("--meta-popsize", type=int, default=16)
    p.add_argument("--meta-epochs", type=int, default=200)
    p.add_argument("--bounds", type=float, nargs=2, default=[-1.0, 1.0])

    p.add_argument("--lga-config", type=str, default="configs/lga_config.json")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    _set_seed(args.seed)

    lga_cfg = json.load(open(os.path.join(REPO_ROOT, args.lga_config), "r", encoding="utf-8"))
    lga_cfg.update(
        {
            "popsize": int(args.inner_popsize),
            "problemdim": int(args.problemdim),
            "minimize": True,
            "seed": int(args.seed),
        }
    )

    inner = LGA(lga_cfg)
    train_funcs = _load_cec_funcs(list(args.train_fids))

    class _Runner:
        def __init__(self, algo):
            self.algo = algo

    meta_task = MetaTask(
        repaire=True,
        dim=int(inner.n_params),
        inner_algo=_Runner(inner),
        inner_generations=int(args.inner_generations),
        inner_funcs=train_funcs,
        inner_popshape=(1, int(args.inner_popsize), int(args.problemdim)),
    )

    meta_opt = CMAES(
        popsize=int(args.meta_popsize),
        problemdim=int(inner.n_params),
        generations=int(args.meta_epochs),
        bounds=(float(args.bounds[0]), float(args.bounds[1])),
    )

    best_f = float("inf")
    best_params = None

    for epoch in range(int(args.meta_epochs)):
        pop, _info = meta_opt.step(pop=None, task=meta_task)
        fit = pop[..., 0].view(-1)
        cur_best, idx = torch.min(fit, dim=0)
        cur_best = float(cur_best.item())
        cur_params = pop.view(-1, pop.shape[-1])[int(idx.item()), 1:].detach().cpu().clone()

        if cur_best < best_f:
            best_f = cur_best
            best_params = cur_params
            ckpt = {
                "best_fitness": float(best_f),
                "best_params": best_params.view(-1),
                "epoch": int(epoch),
                "meta_opt": "CMAES",
                "inner_opt": "LGA",
                "train_problem_dim": int(args.problemdim),
                "train_popsize": int(args.inner_popsize),
                "inner_generations": int(args.inner_generations),
                "train_fids": list(args.train_fids),
                "bounds": [float(args.bounds[0]), float(args.bounds[1])],
            }
            torch.save(ckpt, os.path.join(args.outdir, "metabbo_lga_best.pth"))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch+1}/{int(args.meta_epochs)} cur_best={cur_best:.6f} best={best_f:.6f}", flush=True)

    if best_params is not None:
        ckpt = {
            "best_fitness": float(best_f),
            "best_params": best_params.view(-1),
            "epoch": int(args.meta_epochs - 1),
            "meta_opt": "CMAES",
            "inner_opt": "LGA",
            "train_problem_dim": int(args.problemdim),
            "train_popsize": int(args.inner_popsize),
            "inner_generations": int(args.inner_generations),
            "train_fids": list(args.train_fids),
            "bounds": [float(args.bounds[0]), float(args.bounds[1])],
        }
        torch.save(ckpt, os.path.join(args.outdir, "metabbo_lga_final.pth"))


if __name__ == "__main__":
    main()
