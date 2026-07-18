import math
from typing import Callable

import torch
from torch.nn.functional import normalize

from .reparametrization import uniform_string_repametrize_rn


def _schedule_to_steps(to_spend, max_algostep):
    """Split scheduled score effort into bounded algorithmic step sizes."""
    if to_spend <= 0:
        return
    if max_algostep is None or max_algostep <= 0 or to_spend <= max_algostep:
        yield to_spend
        return
    while to_spend > max_algostep:
        yield max_algostep
        to_spend = to_spend - max_algostep
    if to_spend > 0:
        yield to_spend


def _purify_schedule(schedule, n_timesteps, gamma_norm, dt):
    """Clamp a schedule to positive entries and normalize its total effort."""
    assert schedule.shape[0] == n_timesteps, (
        "Schedule length must match total number of timesteps."
    )
    active_schedule = schedule > 0
    purified = torch.zeros_like(schedule)
    if active_schedule.any():
        purified = normalize(schedule * active_schedule, p=1, dim=0) * gamma_norm * dt
    return active_schedule, purified

def b_transport(x, b_function, times, *field_args, **field_kwargs):
    """Integrate points through a time-dependent vector field with Euler steps."""
    dt = times[1:] - times[:-1]
    for i in range(len(times) - 1):
        x = x + b_function(x, times[i], *field_args, **field_kwargs) * dt[i]
    return x

def pure_string_sampling(
    string_latent,
    b_function,
    n_timesteps,
    reparam_every=1,
    *field_args,
    t_initial =0.0,
    t_final = 1.0,
    **field_kwargs,
):
    """Evolve a string under a vector field and periodically reparametrize it."""
    ts = torch.linspace(t_initial, t_final, n_timesteps + 1, device=string_latent.device)
    n_pts = string_latent.shape[0]
    string = string_latent.clone()
    if n_timesteps == 0:
        return string

    dt = ts[1] - ts[0]
    for i in range(n_timesteps):
        string = string + b_function(string, ts[i], *field_args, **field_kwargs) * dt
        if reparam_every is not None and (i + 1) % reparam_every == 0:
            string = uniform_string_repametrize_rn(string, n_pts)
    return string

def dynamic_string_method_T0_singular(
    string_latent: torch.Tensor,
    b_function,
    score_function,
    n_timesteps_total: int,
    convergence_times: torch.Tensor,
    convergence_weights: torch.Tensor,
    gamma_norm: float,
    max_algostep: float | None,
    reparam_every: int | None,
    *field_args,
    **field_kwargs,
):
    """Prototype dynamic string update with score-driven convergence near selected times."""
    raise NotImplementedError(
        "dynamic_string_method_T0_singular is not implemented yet."
    )

def dynamic_string_method_T0_continuous(
    string_latent: torch.Tensor,
    b_function,
    score_function,
    n_timesteps: int,
    schedule: torch.Tensor,
    gamma_norm: float,
    max_algostep: float | None,
    *field_args,
    **field_kwargs,
):
    """Prototype dynamic string update with score-driven convergence near selected times."""
    string_act = string_latent.clone()
    ts = torch.linspace(0.0, 1.0, n_timesteps + 1, device=string_latent.device)
    dt = ts[1] - ts[0]
    _, purified_schedule = _purify_schedule(schedule, n_timesteps, gamma_norm, dt)

    for i in range(n_timesteps):
        string_act = string_act + b_function(string_act, ts[i], *field_args, **field_kwargs) * dt
        string_act = uniform_string_repametrize_rn(string_act, string_act.shape[0])

        for step_size in _schedule_to_steps(purified_schedule[i], max_algostep):
            score = score_function(string_act, ts[i], *field_args, **field_kwargs)
            string_act = string_act + step_size * score
            string_act = uniform_string_repametrize_rn(string_act, string_act.shape[0])

    return string_act


def voronoi_cell_conserved(walkers, principal):
    """Check if each walker is in the same Voronoi cell as the principal string."""
    assert walkers.shape[1:] == principal.shape, "Walkers and principal string must have the same shape."
    n_walkers, n_pts = walkers.shape[:2]
    metric_dims = range(2, walkers.ndim)
    exp_principal = principal.view(1, *principal.shape).expand(n_walkers, *principal.shape)
    dist_to_principal = torch.norm(walkers - exp_principal, dim=metric_dims)
    is_inside = torch.ones((n_walkers, n_pts), dtype=torch.bool, device=walkers.device)
    for i in range(1, n_pts):
        dist_to_other = torch.norm(
            walkers - exp_principal.roll(shifts=i, dims=1),
            dim=metric_dims,
        )
        is_inside = is_inside & (dist_to_principal < dist_to_other)
    return is_inside

def static_string_method_Tfinite(
    string_principal: torch.Tensor,
    string_walkers: torch.Tensor,
    score_function,
    t: float,
    T: float,
    n_voronoi_cycles: int,
    n_steps_per_cycle: int,
    step_size: float,
    update_coefficient: float,
    *field_args,
    **field_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prototype static string update with score-driven convergence at finite time."""
    n_walkers, n_pts = string_walkers.shape[:2]
    for _ in range(n_voronoi_cycles):
        for _ in range(n_steps_per_cycle):
            flat_walkers = string_walkers.reshape(-1, *string_principal.shape[1:])
            score = score_function(
                flat_walkers, t, *field_args, **field_kwargs
            ).reshape_as(string_walkers)
            noise = math.sqrt(2 * step_size * T) * torch.randn_like(string_walkers)
            candidate = string_walkers + step_size * score + noise
            keep = voronoi_cell_conserved(candidate, string_principal)
            string_walkers = torch.where(
                keep.view(n_walkers, n_pts, *[1] * (string_walkers.ndim - 2)),
                candidate,
                string_walkers,
            )
        string_principal = (
            (1 - update_coefficient) * string_principal
            + update_coefficient * string_walkers.mean(dim=0)
        )
        string_principal = uniform_string_repametrize_rn(string_principal, n_pts)
    return string_principal, string_walkers

def dynamic_string_method_Tfinite_singular(
    string_latent: torch.Tensor,
    b_function: Callable,
    score_function,
    T,
    n_timesteps_total: int,
    convergence_times: torch.Tensor,
    convergence_weights: torch.Tensor,
    n_walkers: int,
    n_voronoi_cycles: int,
    n_steps_per_cycle: int,
    main_step_size: float,
    update_coefficient: float,
    *field_args,
    **field_kwargs,
):
    """Prototype dynamic string update with score-driven convergence near selected times."""
    string_act = string_latent.clone()
    n_intermediate_phases = len(convergence_times)
    convergence_times = torch.cat(
        [convergence_times, torch.tensor([1.0], device=string_latent.device)],
        dim=0,
    )
    convergence_weights = normalize(convergence_weights, p=1, dim=0)
    
    # Initial phase of pure b-transport
    if convergence_times[0] > 0.0:
        n_timesteps = int(n_timesteps_total * convergence_times[0])
        string_act = pure_string_sampling(
            string_act,
            b_function,
            n_timesteps,
            reparam_every=1,
            t_initial=0.0,
            t_final=convergence_times[0],
            *field_args,
            **field_kwargs,
        )
    
    walkers = None
    n_timesteps = 0
    for n_phase in range(n_intermediate_phases):
        walkers = string_act.clone().unsqueeze(0).expand(n_walkers, *string_act.shape)

        string_act, walkers = static_string_method_Tfinite(
            string_principal=string_act,
            string_walkers=walkers,
            score_function=score_function,
            t=convergence_times[n_phase],
            T=T,
            n_voronoi_cycles=n_voronoi_cycles,
            n_steps_per_cycle=n_steps_per_cycle,
            step_size=main_step_size * convergence_weights[n_phase],
            update_coefficient=update_coefficient,
            *field_args,
            **field_kwargs,
        )

        n_timesteps = int(
            n_timesteps_total
            * (convergence_times[n_phase + 1] - convergence_times[n_phase])
        )
        string_act = pure_string_sampling(
            string_act,
            b_function,
            n_timesteps,
            reparam_every=1,
            t_initial=convergence_times[n_phase],
            t_final=convergence_times[n_phase+1],
            *field_args,
            **field_kwargs,
        )

    if walkers is None or n_timesteps == 0:
        return string_act, walkers

    walkers = walkers.view(-1, *string_act.shape[1:])
    tsf = torch.linspace(
        convergence_times[n_intermediate_phases - 1],
        1.0,
        n_timesteps + 1,
        device=string_latent.device,
    )
    dt = tsf[1] - tsf[0]
    for i in range(n_timesteps):
        walkers = walkers + b_function(walkers, tsf[i], *field_args, **field_kwargs) * dt

    return string_act, walkers.view(n_walkers, *string_act.shape)



def dynamic_string_method_Tfinite_continuous(
    string_latent: torch.Tensor,
    b_function,
    score_function,
    T: float,
    n_timesteps: int,
    n_walkers: int,
    schedule: torch.Tensor,
    gamma_norm: float,
    update_coefficient: float,
    max_algostep: float | None,
    *field_args,
    **field_kwargs,
):
    """Prototype dynamic string update with score-driven convergence near selected times."""
    string_act = string_latent.clone()
    ts = torch.linspace(0.0, 1.0, n_timesteps + 1, device=string_latent.device)
    dt = ts[1] - ts[0]

    active_schedule, purified_schedule = _purify_schedule(
        schedule, n_timesteps, gamma_norm, dt
    )
    starting_algo_index = (
        torch.where(active_schedule)[0][0].item() if active_schedule.any() else n_timesteps
    )

    for i in range(starting_algo_index):
        string_act = string_act + b_function(string_act, ts[i], *field_args, **field_kwargs) * dt
        string_act = uniform_string_repametrize_rn(string_act, string_act.shape[0])

    walkers = string_act.clone().unsqueeze(0).expand(n_walkers, *string_act.shape)

    for i in range(starting_algo_index, n_timesteps):
        for step_size in _schedule_to_steps(purified_schedule[i], max_algostep):
            score = score_function(
                walkers.view(-1, *string_act.shape[1:]),
                ts[i],
                *field_args,
                **field_kwargs,
            ).view(walkers.shape)
            tmp = (
                walkers
                + step_size * score
                + math.sqrt(2 * step_size * T) * torch.randn_like(walkers)
            )
            keep = voronoi_cell_conserved(tmp, string_act)
            walkers = torch.where(
                keep.view(*walkers.shape[:2], *[1] * (string_act.ndim - 1)),
                tmp,
                walkers,
            )
            string_act = (
                (1 - update_coefficient) * string_act
                + update_coefficient * walkers.mean(dim=0)
            )
            string_act = uniform_string_repametrize_rn(string_act, string_act.shape[0])

        string_act = string_act + b_function(string_act, ts[i], *field_args, **field_kwargs) * dt
        string_act = uniform_string_repametrize_rn(string_act, string_act.shape[0])
        flat_walkers = walkers.view(-1, *string_act.shape[1:])
        walkers = walkers + b_function(
            flat_walkers, ts[i], *field_args, **field_kwargs
        ).view(walkers.shape) * dt

    return string_act
