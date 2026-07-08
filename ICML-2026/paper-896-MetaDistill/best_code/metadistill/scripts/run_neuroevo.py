import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from optimizers.base_model import GradFreeBBO, GradTrainedBBO
from optimizers.utils import sort_by_fitness
from tasks.neuroevolution.neuro_evo_tasks import TASKS
from tasks.neuroevolution.neuroevo_net import EvoNet
from tasks.problem import NeuroEvoTask
from torch_basic_settings import DEVICE, DTYPE


def _set_global_seed(seed: int) -> None:
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def _fitness_softmax_weights(
    fitness: torch.Tensor,
    *,
    minimize: bool,
    temperature: float,
) -> torch.Tensor:
    """
    Compute detached per-individual weights from fitness (for SSFT), without backprop through fitness.

    Args:
        fitness: (N,)
        minimize: Whether lower fitness is better.
        temperature: Softmax temperature (>0). Lower -> sharper (more elite-focused).

    Returns:
        weights: (N,), sums to 1, detached.
    """
    if fitness.ndim != 1:
        fitness = fitness.view(-1)
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        temperature = 1.0

    fit = fitness.detach()
    if minimize:
        logits = -fit / temperature
    else:
        logits = fit / temperature
    logits = logits - logits.max()  # stabilize
    return torch.softmax(logits, dim=0).detach()


def _les_diag_gaussian_logprob(
    x: torch.Tensor,
    *,
    mean: torch.Tensor,
    sigma: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Diagonal Gaussian log-probability log N(x | mean, diag(sigma^2)), up to a constant.

    Args:
        x: (N, D)
        mean: (D,)
        sigma: (D,)
        eps: clamp for sigma.

    Returns:
        logprob: (N,)
    """
    mean = mean.view(1, -1)
    sigma = sigma.clamp_min(float(eps)).view(1, -1)
    z = (x - mean) / sigma
    # omit constant -0.5 * D * log(2pi)
    return (-0.5 * (z * z).sum(dim=-1)) - torch.log(sigma).sum(dim=-1)


def prepare_bbo(
    algo_name: str,
    config_path: str,
    ckpt: str,
    *,
    seed: Optional[int] = None,
    problemdim: Optional[int] = None,
    popsize: Optional[int] = None,
) -> Union[GradTrainedBBO, GradFreeBBO]:
    with open(config_path, "r", encoding="utf-8") as fp:
        config = json.load(fp)

    if seed is not None:
        config["seed"] = int(seed)
    if problemdim is not None:
        config["problemdim"] = int(problemdim)
    if popsize is not None:
        config["popsize"] = int(popsize)

    config.setdefault("minimize", True)

    if algo_name == "POM":
        # POM uses state_dict checkpoints.
        from optimizers import POM

        optimizer = POM(config)
        state = torch.load(ckpt, map_location="cpu")
        optimizer.load_state_dict(state, strict=True)
    elif algo_name == "LDE":
        from optimizers import LDE

        optimizer = LDE(config=config)
        optimizer.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    elif algo_name == "GradBasedLES":
        from optimizers import GradBasedLES

        optimizer = GradBasedLES(config=config)
        optimizer.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    elif algo_name == "GradBasedLGA":
        from optimizers import GradBasedLGA

        optimizer = GradBasedLGA(config=config)
        optimizer.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    elif algo_name == "GradFreeLES":
        from optimizers import GradFreeLES

        optimizer = GradFreeLES(config=config)
        optimizer.update_params(torch.load(ckpt, map_location="cpu")["best_params"])
    elif algo_name == "LGA":
        from optimizers import LGA

        optimizer = LGA(config=config)
        optimizer.update_params(torch.load(ckpt, map_location="cpu")["best_params"])

    else:
        raise ValueError(f"Optimizer named {algo_name} not found.")

    if hasattr(optimizer, "to"):
        optimizer = optimizer.to(device=DEVICE, dtype=DTYPE)
    return optimizer


def prepare_task(task_id: int, net_config_path: str, step: int = 128) -> NeuroEvoTask:
    if task_id not in [i for i in range(1, 7)]:
        raise ValueError(f"NeuroEvolution Task with ID {task_id} does not exist.")

    with open(net_config_path, "r", encoding="utf-8") as fp:
        config = json.load(fp)
    net = EvoNet(config=config)

    ne_task = TASKS[task_id]
    task = NeuroEvoTask(fun=ne_task, repaire=True, net=net, step=step)

    return task


def run_bbo(
    algo: Union[GradTrainedBBO, GradFreeBBO],
    task: NeuroEvoTask,
    n_generations: int,
    problemdim: int,
    *,
    adapt_lr: float = 0.0,
    ss_interval: int = 1,
    ssft_mode: str = "mu",
    ssft_fit_temp: float = 1.0,
    ssft_adv_norm: bool = True,
    elites_ratio: float = 0.1,
    bp_interval: int = 1,
    popsize: int = 100,
) -> list[float]:
    if not 0 < elites_ratio < 1:
        raise ValueError("elites ratio should lie in (0, 1)")

    ss_interval = int(ss_interval)
    if ss_interval < 1:
        ss_interval = 1
    
    algo.reset()

    batchsize = 1
    n_elites = int(elites_ratio * popsize)
    pop = task.genRandomPop((batchsize, popsize, problemdim))
    pop, _ = task.calfitness(x=pop)
    pop = sort_by_fitness(pop)
    elites = pop[:, :n_elites, :].detach()  # (1, n_elites, 1+d)

    meta_opt = None
    if algo.needs_backward and adapt_lr > 0:
        meta_opt = torch.optim.Adam(params=algo.parameters(), lr=float(adapt_lr))

    elite_mu_history = deque(maxlen=ss_interval)
    elite_mu_history.append(elites[0, :, 1:].mean(dim=0).detach())  # (d,)

    generation_bar = tqdm(range(n_generations), ncols=100)
    best_curve: list[float] = []
    for t in generation_bar:
        # forward
        if meta_opt is None:
            # No SSFT update => avoid building a computation graph over 500 generations.
            with torch.no_grad():
                pop_raw, _ = algo(pop, task)  # (1, n, 1+d)
            pop_raw = pop_raw.detach()
        else:
            pop_raw, _ = algo(pop, task)  # (1, n, 1+d)
        pop = sort_by_fitness(pop_raw)
        if meta_opt is None:
            pop = pop.detach()
        best_curve.append(float(pop[0, 0, 0].detach().cpu().item()))

        # update elites
        elites = torch.cat([elites, pop[:, :n_elites, :].detach()], dim=1)
        elites = sort_by_fitness(elites)  # (1, <=(t+2)*n_elites, 1+d)
        if elites.shape[1] > popsize:
            elites = elites[:, :popsize, :]  # keep top popsize by fitness

        # backward
        if meta_opt is not None and (t + 1) % bp_interval == 0 and elites.shape[1] >= 0.5 * popsize:
            mu_ref = elite_mu_history[0] if len(elite_mu_history) > 0 else elites[0, :, 1:].mean(dim=0).detach()

            if ssft_mode == "mu":
                mu_x = pop[0, :, 1:].mean(dim=0)  # (d,)
                denom = mu_ref.abs().clamp_min(1e-8)
                loss = ((mu_x - mu_ref) / denom).abs().mean()
            elif ssft_mode == "fitw":
                fitness = pop_raw[0, :, 0].view(-1)
                x = pop_raw[0, :, 1:]
                weights = _fitness_softmax_weights(fitness, minimize=True, temperature=float(ssft_fit_temp))
                mu_x = (weights.view(-1, 1) * x).sum(dim=0)  # (d,)
                denom = mu_ref.abs().clamp_min(1e-8)
                loss = ((mu_x - mu_ref) / denom).abs().mean()
            elif ssft_mode == "reinforce":
                # Score-function estimator for LES' sampling distribution (diag Gaussian).
                core = getattr(algo, "core", None)
                state = getattr(core, "state", None) if core is not None else None
                if state is None or not hasattr(state, "mean") or not hasattr(state, "sigma"):
                    raise RuntimeError("SSFT mode 'reinforce' currently supports LES only (needs algo.core.state.mean/sigma).")

                x = pop_raw[0, :, 1:].detach()  # treat sampled actions as constants
                logprob = _les_diag_gaussian_logprob(x, mean=state.mean, sigma=state.sigma)  # (N,)
                fitness = pop_raw[0, :, 0].detach().view(-1)  # (N,)

                adv = fitness - fitness.mean()
                if ssft_adv_norm:
                    adv = adv / adv.std().clamp_min(1e-8)

                loss = (adv * logprob).mean()
            else:
                raise ValueError(f"Unknown ssft_mode={ssft_mode!r}. Expected one of: mu, fitw, reinforce.")

            if not loss.requires_grad:
                raise RuntimeError(
                    "SSFT loss has no grad. If this is LDE baseline, set action_mode='rsample' (not 'sample')."
                )
            loss.backward()
            meta_opt.step()
            meta_opt.zero_grad()

        elite_mu_history.append(pop[0, :n_elites, 1:].mean(dim=0).detach())

        if meta_opt is not None:
            # Truncated self-supervised adaptation: avoid backprop through the full rollout.
            pop = pop.detach()
            export_state = getattr(algo, "export_state", None)
            import_state = getattr(algo, "import_state", None)
            if callable(export_state) and callable(import_state):
                algo.import_state(export_state())
            

    return best_curve

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, help="POM|LDE|GradBasedLES|GradBasedLGA|GradFreeLES|LGA")
    p.add_argument("--algo-config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tasks", type=int, nargs="+", required=True, help="NeuroEvo task ids, e.g. 3 4 5 6")
    p.add_argument("--net-config-dir", required=True, help="Directory containing *.json net configs")
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--popsize", type=int, default=200)
    p.add_argument("--adapt-lr", type=float, default=0.0)
    p.add_argument("--ss-interval", type=int, default=1, help="SSFT self-supervision interval j (>=1)")
    p.add_argument("--ssft-mode", type=str, default="mu", choices=["mu", "fitw", "reinforce"])
    p.add_argument("--ssft-fit-temp", type=float, default=1.0, help="Softmax temperature for ssft-mode=fitw")
    p.add_argument("--ssft-adv-norm", action="store_true", help="Normalize advantage for ssft-mode=reinforce")
    p.add_argument("--elites-ratio", type=float, default=0.1)
    p.add_argument("--bp-interval", type=int, default=1)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    net_cfg_map = {
        1: "cartpole.json",
        2: "lander.json",
        3: "walker.json",
        4: "car_ctnus.json",
        5: "acrobot.json",
        6: "pendulum.json",
    }

    for seed in args.seeds:
        _set_global_seed(seed)
        for task_id in args.tasks:
            net_cfg = net_cfg_map.get(task_id, None)
            if net_cfg is None:
                raise ValueError(f"Unknown task id {task_id}")

            net_config_path = str(Path(args.net_config_dir) / net_cfg)
            task = prepare_task(task_id=task_id, net_config_path=net_config_path, step=args.steps)
            problemdim = int(task.net.get_dim())

            optimizer = prepare_bbo(
                algo_name=args.algo,
                config_path=args.algo_config,
                ckpt=args.ckpt,
                seed=seed,
                problemdim=problemdim,
                popsize=args.popsize,
            )

            curve = run_bbo(
                algo=optimizer,
                task=task,
                n_generations=args.generations,
                problemdim=problemdim,
                adapt_lr=args.adapt_lr,
                ss_interval=args.ss_interval,
                ssft_mode=args.ssft_mode,
                ssft_fit_temp=args.ssft_fit_temp,
                ssft_adv_norm=bool(args.ssft_adv_norm),
                elites_ratio=args.elites_ratio,
                bp_interval=args.bp_interval,
                popsize=args.popsize,
            )

            out = {
                "meta": {
                    "algo": args.algo,
                    "algo_config": args.algo_config,
                    "ckpt": args.ckpt,
                    "tasks": args.tasks,
                    "steps": args.steps,
                    "generations": args.generations,
                    "seeds": args.seeds,
                    "popsize": args.popsize,
                    "adapt_lr": args.adapt_lr,
                    "ss_interval": args.ss_interval,
                    "ssft_mode": args.ssft_mode,
                    "ssft_fit_temp": args.ssft_fit_temp,
                    "ssft_adv_norm": bool(args.ssft_adv_norm),
                    "elites_ratio": args.elites_ratio,
                    "bp_interval": args.bp_interval,
                },
                "task_id": task_id,
                "task_name": TASKS[task_id]["fid"],
                "problemdim": problemdim,
                "seed": int(seed),
                "best_curve": curve,
                "final_best": curve[-1] if len(curve) else None,
            }
            out_path = outdir / f"{args.algo}_task{task_id}_{TASKS[task_id]['fid']}_seed{seed}.json"
            out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
