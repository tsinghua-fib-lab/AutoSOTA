from math import ceil

import torch
from tqdm import tqdm

# from calibrated_guidance.mcmc import PotentialRandomWalkKernel


def ddim(score_function, **ddim_hparams):
    raise NotImplementedError("DDIM sampling not implemented in this snippet.")


def edm(score_function, **edm_hparams):
    raise NotImplementedError("EDM sampling not implemented in this snippet.")


def ddpm(score_function, **ddpm_hparams):
    raise NotImplementedError("DDPM sampling not implemented in this snippet.")


def edm_sde(
        guided_mean_function,
        sigma_steps,
        factor_steps,
        scaling_factor,
        scaling_steps,
        shape,
        n_samples=1,
        sde=True,
        device=None,
        use_tqdm=True,
        return_intermediate=False,
        x_t_start=None,
):
    if device is None:
        device = sigma_steps.device if torch.is_tensor(sigma_steps) else "cuda"

    if x_t_start is None:
        sigma_max = sigma_steps[0] if torch.is_tensor(sigma_steps) else sigma_steps[0]
        x_t = torch.randn((n_samples, *shape), device=device) * sigma_max
    else:
        x_t = x_t_start.to(device)

    num_steps = len(sigma_steps) - 1
    step_range = tqdm(range(num_steps)) if use_tqdm else range(num_steps)

    trajectory = []
    if return_intermediate:
        trajectory.append(x_t)

    for i in step_range:
        sigma = sigma_steps[i]
        factor = factor_steps[i]
        s_factor = scaling_factor[i]
        s_step = scaling_steps[i]

        guided_mean = guided_mean_function(x_t, (1 - i / num_steps))
        assert not guided_mean.requires_grad, f"Guided mean function should not have gradients attached. {i}"
        score = (guided_mean - x_t / s_step) / sigma ** 2 / s_step

        if sde:
            epsilon = torch.randn_like(x_t)
            x_t = x_t * s_factor + factor * score + ((factor ** 0.5) * epsilon if i < num_steps - 1 else 0)
        else:
            x_t = x_t * s_factor + factor * score * 0.5

        if return_intermediate:
            trajectory.append(x_t)

    if return_intermediate:
        return x_t, trajectory
    return x_t


def flow_matching(
        mean_function,
        time_steps,
        shape,
        n_samples=1,
        device=None,
        use_tqdm=True,
        return_intermediate=False,
):
    """
    Standard flow matching
    guied_mean_function - Callable(xt, t), returns E[x0|xt,y]
    times - time schedule (decreasing)
    shape - shape of a single sample
    N - # of samples to draw
    device - for torch
    use_tqdm - progress bar
    """
    if device is None:
        device = time_steps.device

    x_t = torch.randn((n_samples, *shape), device=device)

    time_pairs = list(zip(time_steps[:-1], time_steps[1:]))
    if use_tqdm:
        time_pairs = tqdm(time_pairs, total=len(time_pairs))

    trajectory = []
    if return_intermediate:
        trajectory.append(x_t)
    for sigma_idx, (time, time_next) in enumerate(time_pairs):
        predicted_mean = mean_function(x_t, time)
        v = (predicted_mean - x_t) / time
        x_t = x_t + (time - time_next) * v
        if return_intermediate:
            trajectory.append(x_t)

    if return_intermediate:
        return x_t, trajectory
    return x_t


def mcmc(
        log_posterior,
        differentiable: bool,
        shape,
        num_chains=4,
        n_samples=2000,
        warmup_steps=1000,
):
    def potential_fn(z):
        if isinstance(z, dict):
            z = z["x"]
        return -log_posterior(z)

    import pyro.infer.mcmc as mcmc
    import torch

    if differentiable:
        kernel = mcmc.NUTS(potential_fn)
    else:
        kernel = PotentialRandomWalkKernel(potential_fn)
    initial_params = torch.zeros(*shape)
    if num_chains > 1:
        initial_params = initial_params.unsqueeze(0).repeat(num_chains, *[1] * len(shape))
    mcmc_run = mcmc.MCMC(
        kernel,
        num_samples=ceil(n_samples / num_chains),
        num_chains=num_chains,
        initial_params={"x": initial_params},
        warmup_steps=warmup_steps,
    )
    mcmc_run.run(torch.zeros(shape))
    return mcmc_run.get_samples()["x"]
