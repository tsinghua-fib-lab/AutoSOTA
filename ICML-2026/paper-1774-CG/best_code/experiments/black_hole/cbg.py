"""Clean Calibrated-Bayesian-Guidance inference for black-hole imaging.

This is the default (non-pool) inference path of the vendored
``algo.reinforce.CalibratedGuidanceReinforce.inference``, extracted into a small,
dependency-light function with no embedded W&B logging, no per-step disk caching
of trajectories, and no plotting. It reuses the *vendored* diffusion posteriors
(``InverseBenchModel`` for the slow DDPM inner loop, ``MeanFlowDiffusionPosterior``
for the fast one-step renoise sampler) and the shared ``calibrated_guidance``
library (``build_reinforce_mean_fn`` / ``build_reparam_mean_fn`` + ``edm_sde``),
so the estimator math is identical to the original.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from . import _engine  # noqa: F401  (adds third_party/InverseBench to sys.path)

# Vendored InverseBench engine (verbatim).
from algo.reinforce import InverseBenchModel  # noqa: E402
from algo.meanflow_posterior import MeanFlowDiffusionPosterior  # noqa: E402
from utils.diffusion import DiffusionSampler  # noqa: E402
from utils.scheduler import Scheduler  # noqa: E402

# Shared estimator library.
from calibrated_guidance.guidance import (  # noqa: E402
    build_reinforce_mean_fn,
    build_reparam_mean_fn,
)
from calibrated_guidance.inference import edm_sde  # noqa: E402

IMAGE_SHAPE = (1, 64, 64)  # black-hole images are single-channel 64x64


def build_log_likelihood_fn(forward_op, observation: torch.Tensor,
                            chunk: int = 128) -> Callable:
    """``log p(y|x0)`` for a batch of x0 candidates, as used by the estimators.

    Mirrors ``CalibratedGuidanceReinforce.get_log_likelihood_fn``: the forward
    operator's data-misfit ``loss`` is the negative log-likelihood, so we return
    ``-loss``. Input ``x0_samples`` has shape ``[1, K, 1, 64, 64]``; output is
    ``[1, K]``. The forward-operator loss is evaluated in chunks of at most
    ``chunk`` candidates to bound memory at large K (numerically identical to a
    single pass; ``chunk <= 0`` disables chunking).
    """

    def _ll(x0_samples: torch.Tensor) -> torch.Tensor:
        x = x0_samples.reshape(-1, *IMAGE_SHAPE)
        if chunk <= 0 or x.shape[0] <= chunk:
            return -forward_op.loss(x, observation).reshape(1, -1)
        parts = [-forward_op.loss(x[i:i + chunk], observation).reshape(-1)
                 for i in range(0, x.shape[0], chunk)]
        return torch.cat(parts, dim=0).reshape(1, -1)

    return _ll


def run_cbg_inference(
    *,
    net,
    forward_op,
    observation: torch.Tensor,
    posterior: str = "meanflow",
    scheduler_config: dict,
    reinf_k: int = 128,
    inner_loop_steps: int = 25,
    mf_net=None,
    mf_sibling_renoise_t: Optional[float] = None,
    mf_sigma_min: float = 0.002,
    guidance_scale: float = 0.003,
    reparam: bool = False,
    use_clamp: bool = False,
    stop_at_sigma: Optional[float] = 0.05,
    last_hop: bool = True,
    output_path: str = "/tmp/cbg_blackhole",
    device: str = "cuda",
    on_step: Optional[Callable] = None,
) -> torch.Tensor:
    """Reconstruct one black-hole image from its observation with CBG.

    Args:
        net: the pretrained VP diffusion prior (used by the ``ddpm`` posterior).
        forward_op: ``BlackHoleImaging`` forward operator.
        observation: the radio-interferometry measurement for one instance.
        posterior: ``"meanflow"`` (fast, needs ``mf_net``) or ``"ddpm"`` (slow).
        scheduler_config: kwargs for the outer ``Scheduler`` (VP schedule).
        reinf_k: number K of per-step candidate samples from ``p(x0|xt)``.
        inner_loop_steps: DDPM inner-loop length (``ddpm`` posterior only).
        mf_net: the mean-flow network (``meanflow`` posterior only).
        guidance_scale: likelihood temperature (paper black-hole default 3e-3).
        reparam: use the gradient-based estimator instead of gradient-free.
        use_clamp: clamp the guided x0 mean to [-1, 1] each step (paper: off).
        stop_at_sigma: truncate the SDE at the first step with sigma <= this
            (paper: 0.05); ``None`` runs the full schedule.
        last_hop: after truncation, take one deterministic guided mean step
            (paper: on). Together with ``stop_at_sigma`` this denoises the final
            sample and is required to match the reported PSNR.
        on_step: optional ``(xt, x0_samples, log_likelihoods, t)`` callback for
            diagnostics/logging (passed to the estimator as ``plot_samples``).

    Returns:
        the reconstructed (normalized) image, shape ``[1, 1, 64, 64]``.
    """
    scheduler = Scheduler(**dict(scheduler_config))

    if posterior == "meanflow":
        if mf_net is None:
            raise ValueError("posterior='meanflow' requires a mean-flow net (mf_net).")
        diffusion_posterior = MeanFlowDiffusionPosterior(
            scheduler, mf_net, output_path,
            sigma_min=mf_sigma_min,
            sibling_renoise_t=mf_sibling_renoise_t,
            enable_grad=reparam,
        )
    elif posterior == "ddpm":
        if reparam:
            raise ValueError("reparam estimator requires posterior='meanflow'.")
        diffusion_posterior = InverseBenchModel(scheduler, net, inner_loop_steps, output_path)
    else:
        raise ValueError(f"unknown posterior {posterior!r}; use 'meanflow' or 'ddpm'.")

    log_likelihood_fn = build_log_likelihood_fn(forward_op, observation)
    build_mean_fn = build_reparam_mean_fn if reparam else build_reinforce_mean_fn
    mean_fn = build_mean_fn(
        diffusion_posterior, log_likelihood_fn, reinf_k,
        plot_samples=on_step, guidance_scale=guidance_scale,
    )

    if use_clamp:
        inner_mean_fn = mean_fn

        def mean_fn(xt, t):  # noqa: F811 - intentional clamp wrapper
            return inner_mean_fn(xt, t).clamp(-1.0, 1.0)

    sigma_steps = torch.as_tensor(scheduler.sigma_steps).float().to(device)
    factor_steps = torch.as_tensor(scheduler.factor_steps).float().to(device)
    scaling_factor = torch.as_tensor(scheduler.scaling_factor).float().to(device)
    scaling_steps = torch.as_tensor(scheduler.scaling_steps).float().to(device)

    # Truncate the noisy SDE at the first step where sigma <= stop_at_sigma, then
    # finish with one deterministic guided "hop" (last_hop). This is the paper's
    # Table-2 configuration and is what reaches the reported PSNR.
    stop_steps = scheduler.num_steps
    if stop_at_sigma is not None:
        below = (sigma_steps <= float(stop_at_sigma)).nonzero(as_tuple=False)
        if below.numel() > 0:
            stop_steps = max(int(below[0].item()), 1)
        sigma_steps = sigma_steps[: stop_steps + 1]
        factor_steps = factor_steps[:stop_steps]
        scaling_factor = scaling_factor[:stop_steps]
        scaling_steps = scaling_steps[: stop_steps + 1]

    result = edm_sde(
        mean_fn,
        sigma_steps,
        factor_steps,
        scaling_factor,
        scaling_steps,
        shape=IMAGE_SHAPE,
        n_samples=1,
        x_t_start=DiffusionSampler(scheduler).get_start(
            torch.ones((1, *IMAGE_SHAPE), device=device)
        ),
    )

    if last_hop:
        last_t = 1.0 - stop_steps / scheduler.num_steps
        result = mean_fn(result, last_t)
    return result
