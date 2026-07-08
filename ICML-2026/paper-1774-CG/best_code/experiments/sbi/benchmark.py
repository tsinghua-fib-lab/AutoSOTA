"""Run Calibrated Bayesian Guidance on the SBI benchmark tasks and score C2ST.

This is the integration point: each task's analytic diffusion posterior +
log-likelihood are fed to the shared ``calibrated_guidance`` estimator
(``build_reinforce_mean_fn`` gradient-free / ``build_reparam_mean_fn``
gradient-based) and sampled with ``flow_matching``; the guided samples are scored
against the reference posterior with C2ST (paper §6.1, Table 1).
"""

from __future__ import annotations

from typing import Optional

import torch

from calibrated_guidance.guidance import build_reinforce_mean_fn, build_reparam_mean_fn
from calibrated_guidance.diffusion_posterior.memory import MemoryDiffusionPosterior
from calibrated_guidance.inference import flow_matching

from experiments.sbi.c2st import c2st
from experiments.sbi.data_io import load_observation, load_reference_samples
from experiments.sbi.tasks import TASKS


def guided_samples(
    task_key: str,
    *,
    estimator: str = "reinforce",
    num_steps: int = 100,
    num_particles: int = 1000,
    n_samples: int = 10_000,
    num_observation: int = 1,
    device: str = "cpu",
    antithetic: bool = False,
    adaptive_alpha: float = 0.0,
    use_memory: bool = False,
    memory_fraction: float = 0.2,
    use_cv: bool = False,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Draw ``n_samples`` posterior samples for a task with CBG.

    estimator: "reinforce" (gradient-free) or "reparam" (gradient-based).
    num_steps (N) and num_particles (K) are the paper's outer steps / per-step
    candidate count (Table 4: N=100, K=1000 for grad-free).
    """
    task = TASKS[task_key]
    posterior = task.build_posterior(device)
    if use_memory:
        posterior = MemoryDiffusionPosterior(posterior, memory_fraction=memory_fraction)
    y = load_observation(task.name, num_observation).to(device)

    def log_likelihood(theta: torch.Tensor) -> torch.Tensor:  # [B,K,d] -> [B,K]
        # Evaluate in double and replace -inf (out-of-support, e.g. two-moons)
        # with a large finite value, so the estimator's softmax stays finite for
        # rows where every candidate is out of support. Mirrors the original SBI
        # eval's double-precision, nan_to_num-guarded weighting.
        ll = task.log_likelihood(theta, y).double()
        ll = torch.nan_to_num(ll, nan=-1e300, neginf=-1e300, posinf=1e300)
        # Softer relative clamping: worst particle at most 10 nats below best
        ll_max = ll.max(dim=-1, keepdim=True).values
        return torch.maximum(ll, ll_max - 10.0)

    build = build_reparam_mean_fn if estimator == "reparam" else build_reinforce_mean_fn
    mean_fn = build(posterior, log_likelihood, num_samples_per_step=num_particles, antithetic=antithetic, adaptive_alpha=adaptive_alpha, use_cv=use_cv, guidance_scale=guidance_scale)

    # t schedule 1 -> 0 (Appendix E). Uniform-prior posteriors divide by (1-t),
    # so nudge the first step off exactly t=1 (there the posterior is ~the prior).
    time_steps = torch.linspace(1.0, 0.0, num_steps, device=device)
    if task.prior == "uniform":
        time_steps[0] = 1.0 - 1e-3

    return flow_matching(
        mean_fn, time_steps, shape=(task.dim,), n_samples=n_samples, use_tqdm=False
    )


def run_task(
    task_key: str,
    *,
    estimator: str = "reinforce",
    num_steps: int = 100,
    num_particles: int = 1000,
    num_observation: int = 1,
    n_samples: Optional[int] = None,
    seed: int = 0,
    device: str = "cpu",
    antithetic: bool = False,
    adaptive_alpha: float = 0.0,
    use_memory: bool = False,
    memory_fraction: float = 0.2,
    use_cv: bool = False,
    guidance_scale: float = 1.0,
) -> float:
    """Sample a task and return the C2ST vs the reference posterior."""
    torch.manual_seed(seed)
    reference = load_reference_samples(TASKS[task_key].name, num_observation).to(device)
    n = n_samples if n_samples is not None else reference.shape[0]
    samples = guided_samples(
        task_key, estimator=estimator, num_steps=num_steps,
        num_particles=num_particles, n_samples=n,
        num_observation=num_observation, device=device,
        antithetic=antithetic,
        adaptive_alpha=adaptive_alpha,
        use_memory=use_memory,
        memory_fraction=memory_fraction,
        use_cv=use_cv,
        guidance_scale=guidance_scale,
    )
    return float(c2st(reference, samples).item())


def run_benchmark(
    *,
    estimator: str = "reinforce",
    num_steps: int = 100,
    num_particles: int = 1000,
    observations: tuple[int, ...] = (1,),
    seed: int = 0,
    device: str = "cpu",
    tasks: tuple[str, ...] = ("task1", "task2", "task3", "task4", "task5"),
) -> dict[str, dict]:
    """Run all tasks, averaging C2ST over ``observations``. Returns per-task
    ``{"c2st_mean", "c2st_std", "paper"}`` plus an ``"average"`` row."""
    results: dict[str, dict] = {}
    all_means = []
    for tk in tasks:
        vals = [run_task(tk, estimator=estimator, num_steps=num_steps,
                         num_particles=num_particles, num_observation=o,
                         seed=seed, device=device) for o in observations]
        v = torch.tensor(vals)
        results[tk] = {
            "c2st_mean": float(v.mean()),
            "c2st_std": float(v.std(unbiased=False)),
            "paper": TASKS[tk].paper_c2st_gradfree,
        }
        all_means.append(float(v.mean()))
    results["average"] = {"c2st_mean": float(torch.tensor(all_means).mean()),
                          "paper": 0.527}
    return results
