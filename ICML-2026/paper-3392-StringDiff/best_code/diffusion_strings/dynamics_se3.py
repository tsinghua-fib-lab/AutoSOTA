import math
from typing import Callable

import torch
from torch.nn.functional import normalize

from .reparametrization import (
    reparaametrize_kabsh_se3,
    uniform_string_repametrize_se3,
)
from .so3 import rotmat_to_rotvec, rotvec_to_rotmat


def _assert_se3_shapes(translations, rotations):
    """Validate paired translation and rotation tensors as SE(3) states."""
    assert translations.shape[-1] == 3, "Translations must have shape (..., 3)."
    assert rotations.shape[-2:] == (3, 3), "Rotations must have shape (..., 3, 3)."
    assert translations.shape[:-1] == rotations.shape[:-2], (
        "Translations and rotations must describe the same batch/string shape."
    )


def _apply_se3_step(translations, rotations, trans_delta, rotvec_delta, step_size):
    """Apply an SE(3) tangent update with body-frame rotational increments."""
    _assert_se3_shapes(translations, rotations)
    assert trans_delta.shape == translations.shape
    assert rotvec_delta.shape == translations.shape

    new_translations = translations + step_size * trans_delta
    delta_rotations = rotvec_to_rotmat(step_size * rotvec_delta)
    new_rotations = torch.matmul(rotations, delta_rotations)
    return new_translations, new_rotations


def _call_field(
    field_function, translations, rotations, t, *field_args, **field_kwargs
):
    """Evaluate a vector field and validate its translational and rotational parts."""
    trans_delta, rotvec_delta = field_function(
        translations, rotations, t, *field_args, **field_kwargs
    )
    assert trans_delta.shape == translations.shape
    assert rotvec_delta.shape == translations.shape
    return trans_delta, rotvec_delta


def _center_translations(translations):
    """Center each string image using equal weights for all chain pieces."""
    if translations.ndim == 2:
        return torch.zeros_like(translations)
    return translations - translations.mean(dim=-2, keepdim=True)


def _reparametrize(
    translations,
    rotations,
    n_pts,
    corr_coeff,
    align_index=None,
    keep_centered=False,
):
    """Optionally center, then reparametrize protein shape and global pose."""
    if keep_centered:
        translations = _center_translations(translations)
    if align_index is not None:
        return reparaametrize_kabsh_se3(
            translations,
            rotations,
            n_new=n_pts,
            index=align_index,
            corr_coeff=corr_coeff,
        )
    return uniform_string_repametrize_se3(
        translations, rotations, n_pts, corr_coeff=corr_coeff
    )


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


def _principal_update(
    principal_translations,
    principal_rotations,
    walker_translations,
    walker_rotations,
    update_coefficient,
):
    """Move the principal SE(3) string toward the mean walker state."""
    mean_translations = walker_translations.mean(dim=0)
    relative_rotations = torch.matmul(
        principal_rotations.unsqueeze(0).transpose(-1, -2), walker_rotations
    )
    mean_rotvec = rotmat_to_rotvec(relative_rotations).mean(dim=0)
    return _apply_se3_step(
        principal_translations,
        principal_rotations,
        mean_translations - principal_translations,
        mean_rotvec,
        update_coefficient,
    )


def b_transport(
    translations,
    rotations,
    b_function,
    times,
    *field_args,
    **field_kwargs,
):
    """Integrate SE(3) states through a time-dependent vector field with Euler steps."""
    _assert_se3_shapes(translations, rotations)
    dt = times[1:] - times[:-1]
    translations = translations.clone()
    rotations = rotations.clone()
    for i in range(len(times) - 1):
        trans_delta, rotvec_delta = _call_field(
            b_function, translations, rotations, times[i], *field_args, **field_kwargs
        )
        translations, rotations = _apply_se3_step(
            translations, rotations, trans_delta, rotvec_delta, dt[i]
        )
    return translations, rotations


def pure_string_sampling(
    translations_latent,
    rotations_latent,
    b_function,
    n_timesteps,
    reparam_every=1,
    *field_args,
    t_initial=0.0,
    t_final=1.0,
    corr_coeff=1.0,
    align_index=None,
    keep_centered=False,
    **field_kwargs,
):
    """Evolve an SE(3) string and periodically reparametrize it by coupled arc length."""
    _assert_se3_shapes(translations_latent, rotations_latent)
    ts = torch.linspace(
        t_initial,
        t_final,
        n_timesteps + 1,
        dtype=translations_latent.dtype,
        device=translations_latent.device,
    )
    n_pts = translations_latent.shape[0]
    translations = translations_latent.clone()
    rotations = rotations_latent.clone()
    if n_timesteps == 0:
        return translations, rotations

    dt = ts[1] - ts[0]
    for i in range(n_timesteps):
        trans_delta, rotvec_delta = _call_field(
            b_function, translations, rotations, ts[i], *field_args, **field_kwargs
        )
        translations, rotations = _apply_se3_step(
            translations, rotations, trans_delta, rotvec_delta, dt
        )
        if reparam_every is not None and (i + 1) % reparam_every == 0:
            translations, rotations = _reparametrize(
                translations,
                rotations,
                n_pts,
                corr_coeff,
                align_index,
                keep_centered,
            )

        if torch.any(torch.isnan(translations)) or torch.any(torch.isnan(rotations)):
            raise ValueError(f"NaN values detected at timestep {i}.")
    return translations, rotations


def dynamic_string_method_T0_singular(
    translations_latent: torch.Tensor,
    rotations_latent: torch.Tensor,
    b_function,
    score_function,
    n_timesteps_total: int,
    convergence_times: torch.Tensor,
    convergence_weights: torch.Tensor,
    gamma_norm: float,
    max_algostep: float | None,
    reparam_every: int | None,
    *field_args,
    corr_coeff=1.0,
    align_index=None,
    keep_centered=False,
    **field_kwargs,
):
    """SE(3) T=0 dynamic string update with score steps at selected times."""
    _assert_se3_shapes(translations_latent, rotations_latent)
    n_pts = translations_latent.shape[0]
    translations = translations_latent.clone()
    rotations = rotations_latent.clone()
    convergence_weights = normalize(convergence_weights, p=1, dim=0) * gamma_norm

    phase_times = torch.cat(
        [
            torch.zeros(1, dtype=translations.dtype, device=translations.device),
            convergence_times.to(dtype=translations.dtype, device=translations.device),
            torch.ones(1, dtype=translations.dtype, device=translations.device),
        ],
        dim=0,
    )

    for phase in range(len(convergence_times)):
        n_timesteps = int(
            n_timesteps_total * (phase_times[phase + 1] - phase_times[phase])
        )
        translations, rotations = pure_string_sampling(
            translations,
            rotations,
            b_function,
            n_timesteps,
            reparam_every,
            *field_args,
            t_initial=phase_times[phase],
            t_final=phase_times[phase + 1],
            corr_coeff=corr_coeff,
            align_index=align_index,
            keep_centered=keep_centered,
            **field_kwargs,
        )

        for step_size in _schedule_to_steps(convergence_weights[phase], max_algostep):
            trans_delta, rotvec_delta = _call_field(
                score_function,
                translations,
                rotations,
                phase_times[phase + 1],
                *field_args,
                **field_kwargs,
            )
            translations, rotations = _apply_se3_step(
                translations, rotations, trans_delta, rotvec_delta, step_size
            )
            if reparam_every is not None:
                translations, rotations = _reparametrize(
                    translations,
                    rotations,
                    n_pts,
                    corr_coeff,
                    align_index,
                    keep_centered,
                )

    n_timesteps = int(n_timesteps_total * (phase_times[-1] - phase_times[-2]))
    return pure_string_sampling(
        translations,
        rotations,
        b_function,
        n_timesteps,
        reparam_every,
        *field_args,
        t_initial=phase_times[-2],
        t_final=phase_times[-1],
        corr_coeff=corr_coeff,
        align_index=align_index,
        keep_centered=keep_centered,
        **field_kwargs,
    )


def dynamic_string_method_T0_continuous(
    translations_latent: torch.Tensor,
    rotations_latent: torch.Tensor,
    b_function,
    score_function,
    n_timesteps: int,
    schedule: torch.Tensor,
    gamma_norm: float,
    max_algostep: float | None,
    *field_args,
    corr_coeff=1.0,
    align_index=None,
    keep_centered=False,
    **field_kwargs,
):
    """SE(3) T=0 dynamic string update with score effort spread over a schedule."""
    _assert_se3_shapes(translations_latent, rotations_latent)
    translations = translations_latent.clone()
    rotations = rotations_latent.clone()
    n_pts = translations.shape[0]
    ts = torch.linspace(
        0.0,
        1.0,
        n_timesteps + 1,
        dtype=translations.dtype,
        device=translations.device,
    )
    dt = ts[1] - ts[0]
    _, purified_schedule = _purify_schedule(schedule, n_timesteps, gamma_norm, dt)

    for i in range(n_timesteps):
        trans_delta, rotvec_delta = _call_field(
            b_function, translations, rotations, ts[i], *field_args, **field_kwargs
        )
        translations, rotations = _apply_se3_step(
            translations, rotations, trans_delta, rotvec_delta, dt
        )
        translations, rotations = _reparametrize(
            translations,
            rotations,
            n_pts,
            corr_coeff,
            align_index,
            keep_centered,
        )

        for step_size in _schedule_to_steps(purified_schedule[i], max_algostep):
            trans_delta, rotvec_delta = _call_field(
                score_function,
                translations,
                rotations,
                ts[i],
                *field_args,
                **field_kwargs,
            )
            translations, rotations = _apply_se3_step(
                translations, rotations, trans_delta, rotvec_delta, step_size
            )
            translations, rotations = _reparametrize(
                translations,
                rotations,
                n_pts,
                corr_coeff,
                align_index,
                keep_centered,
            )

    return translations, rotations


def voronoi_cell_conserved(
    walker_translations,
    walker_rotations,
    principal_translations,
    principal_rotations,
    corr_coeff=1.0,
):
    """Check whether each SE(3) walker remains in its principal string Voronoi cell."""
    assert walker_translations.shape[1:] == principal_translations.shape
    assert walker_rotations.shape[1:] == principal_rotations.shape
    _assert_se3_shapes(walker_translations, walker_rotations)
    _assert_se3_shapes(principal_translations, principal_rotations)

    n_walkers, n_pts = walker_translations.shape[:2]
    principal_t = principal_translations.unsqueeze(0).expand_as(walker_translations)
    principal_r = principal_rotations.unsqueeze(0).expand_as(walker_rotations)

    def distance_to(candidate_t, candidate_r):
        """Compute coupled translation-rotation distances from walkers to candidates."""
        trans_dist = torch.linalg.vector_norm(
            (walker_translations - candidate_t).reshape(n_walkers, n_pts, -1),
            ord=2,
            dim=-1,
        )
        rel_rot = torch.matmul(candidate_r.transpose(-1, -2), walker_rotations)
        rot_dist = torch.linalg.vector_norm(
            rotmat_to_rotvec(rel_rot).reshape(n_walkers, n_pts, -1),
            ord=2,
            dim=-1,
        )
        return trans_dist + corr_coeff * rot_dist

    dist_to_principal = distance_to(principal_t, principal_r)
    is_inside = torch.ones(
        (n_walkers, n_pts), dtype=torch.bool, device=walker_translations.device
    )
    for i in range(1, n_pts):
        dist_to_other = distance_to(
            principal_t.roll(shifts=i, dims=1), principal_r.roll(shifts=i, dims=1)
        )
        is_inside = is_inside & (dist_to_principal < dist_to_other)
    return is_inside


def static_string_method_Tfinite(
    principal_translations: torch.Tensor,
    principal_rotations: torch.Tensor,
    walker_translations: torch.Tensor,
    walker_rotations: torch.Tensor,
    score_function,
    t: float,
    T: float,
    n_voronoi_cycles: int,
    n_steps_per_cycle: int,
    step_size: float,
    update_coefficient: float,
    *field_args,
    corr_coeff=1.0,
    align_index=None,
    keep_centered=False,
    **field_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """SE(3) static string update with finite-temperature Voronoi walkers."""
    _assert_se3_shapes(principal_translations, principal_rotations)
    _assert_se3_shapes(walker_translations, walker_rotations)
    n_walkers = walker_translations.shape[0]
    n_pts = principal_translations.shape[0]

    for _ in range(n_voronoi_cycles):
        for _ in range(n_steps_per_cycle):
            flat_t = walker_translations.reshape(-1, *principal_translations.shape[1:])
            flat_r = walker_rotations.reshape(-1, *principal_rotations.shape[1:])
            score_t, score_r = _call_field(
                score_function, flat_t, flat_r, t, *field_args, **field_kwargs
            )
            score_t = score_t.reshape_as(walker_translations)
            score_r = score_r.reshape_as(walker_translations)
            noise_t = math.sqrt(2 * step_size * T) * torch.randn_like(
                walker_translations
            )
            noise_r = math.sqrt(2 * step_size * T) * torch.randn_like(
                walker_translations
            )
            tmp_t, tmp_r = _apply_se3_step(
                walker_translations,
                walker_rotations,
                score_t,
                score_r + noise_r / step_size,
                step_size,
            )
            tmp_t = tmp_t + noise_t

            keep = voronoi_cell_conserved(
                tmp_t,
                tmp_r,
                principal_translations,
                principal_rotations,
                corr_coeff=corr_coeff,
            )
            walker_translations = torch.where(
                keep.view(n_walkers, n_pts, *[1] * (walker_translations.ndim - 2)),
                tmp_t,
                walker_translations,
            )
            walker_rotations = torch.where(
                keep.view(n_walkers, n_pts, *[1] * (walker_rotations.ndim - 2)),
                tmp_r,
                walker_rotations,
            )

        principal_translations, principal_rotations = _principal_update(
            principal_translations,
            principal_rotations,
            walker_translations,
            walker_rotations,
            update_coefficient,
        )
        principal_translations, principal_rotations = _reparametrize(
            principal_translations,
            principal_rotations,
            n_pts,
            corr_coeff,
            align_index,
            keep_centered,
        )

    return (
        principal_translations,
        principal_rotations,
        walker_translations,
        walker_rotations,
    )


def dynamic_string_method_Tfinite_singular(
    translations_latent: torch.Tensor,
    rotations_latent: torch.Tensor,
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
    corr_coeff=1.0,
    align_index=None,
    keep_centered=False,
    **field_kwargs,
):
    """SE(3) finite-temperature dynamic string update at selected convergence times."""
    _assert_se3_shapes(translations_latent, rotations_latent)
    translations = translations_latent.clone()
    rotations = rotations_latent.clone()
    n_intermediate_phases = len(convergence_times)
    convergence_times = torch.cat(
        [
            convergence_times.to(dtype=translations.dtype, device=translations.device),
            torch.ones(1, dtype=translations.dtype, device=translations.device),
        ],
        dim=0,
    )
    convergence_weights = normalize(convergence_weights, p=1, dim=0)

    if convergence_times[0] > 0.0:
        n_timesteps = int(n_timesteps_total * convergence_times[0])
        translations, rotations = pure_string_sampling(
            translations,
            rotations,
            b_function,
            n_timesteps,
            reparam_every=1,
            *field_args,
            t_initial=0.0,
            t_final=convergence_times[0],
            corr_coeff=corr_coeff,
            align_index=align_index,
            keep_centered=keep_centered,
            **field_kwargs,
        )

    walker_t = None
    walker_r = None
    n_timesteps = 0
    for n_phase in range(n_intermediate_phases):
        walker_t = (
            translations.clone()
            .unsqueeze(0)
            .expand(n_walkers, *translations.shape)
            .clone()
        )
        walker_r = (
            rotations.clone().unsqueeze(0).expand(n_walkers, *rotations.shape).clone()
        )

        translations, rotations, walker_t, walker_r = static_string_method_Tfinite(
            translations,
            rotations,
            walker_t,
            walker_r,
            score_function,
            convergence_times[n_phase],
            T,
            n_voronoi_cycles,
            n_steps_per_cycle,
            main_step_size * convergence_weights[n_phase],
            update_coefficient,
            *field_args,
            corr_coeff=corr_coeff,
            align_index=align_index,
            keep_centered=keep_centered,
            **field_kwargs,
        )

        n_timesteps = int(
            n_timesteps_total
            * (convergence_times[n_phase + 1] - convergence_times[n_phase])
        )
        translations, rotations = pure_string_sampling(
            translations,
            rotations,
            b_function,
            n_timesteps,
            reparam_every=1,
            *field_args,
            t_initial=convergence_times[n_phase],
            t_final=convergence_times[n_phase + 1],
            corr_coeff=corr_coeff,
            align_index=align_index,
            keep_centered=keep_centered,
            **field_kwargs,
        )

    if walker_t is not None and n_timesteps > 0:
        flat_t = walker_t.reshape(-1, *translations.shape[1:])
        flat_r = walker_r.reshape(-1, *rotations.shape[1:])
        tsf = torch.linspace(
            convergence_times[n_intermediate_phases - 1],
            1.0,
            n_timesteps + 1,
            dtype=translations.dtype,
            device=translations.device,
        )
        dt = tsf[1] - tsf[0]
        for i in range(n_timesteps):
            trans_delta, rotvec_delta = _call_field(
                b_function, flat_t, flat_r, tsf[i], *field_args, **field_kwargs
            )
            flat_t, flat_r = _apply_se3_step(
                flat_t, flat_r, trans_delta, rotvec_delta, dt
            )
        walker_t = flat_t.reshape(n_walkers, *translations.shape)
        walker_r = flat_r.reshape(n_walkers, *rotations.shape)

    return translations, rotations, walker_t, walker_r


def dynamic_string_method_Tfinite_continuous(
    translations_latent: torch.Tensor,
    rotations_latent: torch.Tensor,
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
    corr_coeff=1.0,
    align_index=None,
    keep_centered=False,
    **field_kwargs,
):
    """SE(3) finite-temperature dynamic string update with scheduled convergence."""
    _assert_se3_shapes(translations_latent, rotations_latent)
    translations = translations_latent.clone()
    rotations = rotations_latent.clone()
    n_pts = translations.shape[0]
    ts = torch.linspace(
        0.0,
        1.0,
        n_timesteps + 1,
        dtype=translations.dtype,
        device=translations.device,
    )
    dt = ts[1] - ts[0]

    active_schedule, purified_schedule = _purify_schedule(
        schedule, n_timesteps, gamma_norm, dt
    )
    starting_algo_index = (
        torch.where(active_schedule)[0][0].item()
        if active_schedule.any()
        else n_timesteps
    )

    for i in range(starting_algo_index):
        trans_delta, rotvec_delta = _call_field(
            b_function, translations, rotations, ts[i], *field_args, **field_kwargs
        )
        translations, rotations = _apply_se3_step(
            translations, rotations, trans_delta, rotvec_delta, dt
        )
        translations, rotations = _reparametrize(
            translations,
            rotations,
            n_pts,
            corr_coeff,
            align_index,
            keep_centered,
        )

    walker_t = (
        translations.clone().unsqueeze(0).expand(n_walkers, *translations.shape).clone()
    )
    walker_r = (
        rotations.clone().unsqueeze(0).expand(n_walkers, *rotations.shape).clone()
    )

    for i in range(starting_algo_index, n_timesteps):
        for step_size in _schedule_to_steps(purified_schedule[i], max_algostep):
            flat_t = walker_t.reshape(-1, *translations.shape[1:])
            flat_r = walker_r.reshape(-1, *rotations.shape[1:])
            score_t, score_r = _call_field(
                score_function, flat_t, flat_r, ts[i], *field_args, **field_kwargs
            )
            score_t = score_t.reshape_as(walker_t)
            score_r = score_r.reshape_as(walker_t)

            noise_t = math.sqrt(2 * step_size * T) * torch.randn_like(walker_t)
            noise_r = math.sqrt(2 * step_size * T) * torch.randn_like(walker_t)
            tmp_t, tmp_r = _apply_se3_step(
                walker_t, walker_r, score_t, score_r + noise_r / step_size, step_size
            )
            tmp_t = tmp_t + noise_t

            keep = voronoi_cell_conserved(
                tmp_t, tmp_r, translations, rotations, corr_coeff=corr_coeff
            )
            walker_t = torch.where(
                keep.view(n_walkers, n_pts, *[1] * (walker_t.ndim - 2)),
                tmp_t,
                walker_t,
            )
            walker_r = torch.where(
                keep.view(n_walkers, n_pts, *[1] * (walker_r.ndim - 2)),
                tmp_r,
                walker_r,
            )
            translations, rotations = _principal_update(
                translations, rotations, walker_t, walker_r, update_coefficient
            )
            translations, rotations = _reparametrize(
                translations,
                rotations,
                n_pts,
                corr_coeff,
                align_index,
                keep_centered,
            )

        trans_delta, rotvec_delta = _call_field(
            b_function, translations, rotations, ts[i], *field_args, **field_kwargs
        )
        translations, rotations = _apply_se3_step(
            translations, rotations, trans_delta, rotvec_delta, dt
        )
        translations, rotations = _reparametrize(
            translations,
            rotations,
            n_pts,
            corr_coeff,
            align_index,
            keep_centered,
        )

        flat_t = walker_t.reshape(-1, *translations.shape[1:])
        flat_r = walker_r.reshape(-1, *rotations.shape[1:])
        trans_delta, rotvec_delta = _call_field(
            b_function, flat_t, flat_r, ts[i], *field_args, **field_kwargs
        )
        flat_t, flat_r = _apply_se3_step(flat_t, flat_r, trans_delta, rotvec_delta, dt)
        walker_t = flat_t.reshape_as(walker_t)
        walker_r = flat_r.reshape_as(walker_r)

    return translations, rotations
