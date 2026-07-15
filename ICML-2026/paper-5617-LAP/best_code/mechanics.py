import torch
from typing import Callable, List, Optional, Tuple, Dict, Union, Any
import numpy as np
import torch.nn.functional as F
from dice import get_s_derivatives


# ============================================================
# Integrator selection
# ============================================================

def pick_integrator(integrator: str):
    if integrator == "x":
        return x_verlet_auto, "x_verlet_auto"
    elif integrator == "v":
        return leapfrog_auto, "leapfrog_auto"
    elif integrator == "euler":
        return euler_maruyama_generalized, "euler_maruyama"

def resolve_gamma(friction) -> Union[float, torch.Tensor]:
    """
    Return the physical friction gamma (>=0) as float or 0-dim tensor.
    Accepts:
      - float: already gamma
      - torch.Tensor: already gamma (recommended: 0-dim scalar)
      - LearnableFriction-like module with friction_tensor()
    """
    if hasattr(friction, "friction_tensor") and callable(getattr(friction, "friction_tensor")):
        gamma = friction.friction_tensor()
    else:
        gamma = friction

    if isinstance(gamma, torch.Tensor):
        # ensure scalar-ish
        if gamma.numel() != 1:
            raise ValueError(f"friction tensor must be scalar; got shape {tuple(gamma.shape)}")
        gamma = gamma.reshape(())
    return gamma


def x_verlet_auto(
        x0: torch.Tensor,
        v0: torch.Tensor,
        accel: Callable[[torch.Tensor, float], torch.Tensor],
        dt: float,
        steps: int,
        friction: Union[float, torch.Tensor],
        return_all: bool = False,
        force_coeff=None,
        momentum=None,
        t_start = 0.0
):
    """
    Position-Verlet (x-Verlet) with friction, equivalent to leapfrog_auto.
    """
    x = x0
    v = v0
    t = t_start
    friction = resolve_gamma(friction)

    # 1. Compute Momentum (Half-Step Decay) - Same as leapfrog_auto
    if momentum is None:
        if isinstance(friction, torch.Tensor):
            gamma = friction  # already positive
            momentum_half = torch.exp(-0.5 * gamma * dt)
        else:
            momentum_half = 1.0 if friction < 1e-6 else np.exp(-0.5 * friction * dt)
    else:
        # Use provided momentum (assuming it's the half-step decay factor)
        momentum_half = momentum

    # 2. Compute Force Coefficient (The "Push") - Same as leapfrog_auto
    if force_coeff is None:
        target_inertial = dt / 2.0
        target_overdamped = 1.0

        # Interpolate based on momentum remaining
        force_coeff = momentum_half * target_inertial + (1.0 - momentum_half) * target_overdamped

    # 3. Compute Full-Step Decay Factor
    # Decay over a full step 'dt' is (e^(-0.5 * f * dt))^2 = e^(-f * dt)
    momentum_full = momentum_half * momentum_half


    if return_all:
        history = [x]

    # --- Initialization Step (Half-step Velocity Update for x-Verlet) ---
    # x-Verlet requires the half-step velocity v_{t+dt/2} to start the loop.
    # A. Force at t
    accel_t = accel(x, t)

    # B. Half-step Velocity: v_{t+dt/2} = decay * v_t + coeff * F_t
    v_half = momentum_half * v + force_coeff * accel_t


    for k in range(steps):
        # C. Full-step Position: x_{t+dt} = x_t + dt * v_{t+dt/2}
        x = x + dt * v_half
        t = t + dt

        # D. Force at new position
        accel_next_t = accel(x, t)

        # E. Full-step Velocity: v_{t+dt}
        # The full-step velocity update v_{t+dt} relies on v_{t+dt/2}
        v = momentum_half * v_half + force_coeff * accel_next_t

        # F. Prepare v_{t+3dt/2} for next loop iteration
        # v_{t+3dt/2} = decay * v_{t+dt} + coeff * F_{t+dt}
        # v_half = momentum_full * v_half + (momentum_half + 1.0) * force_coeff * accel_next_t - momentum_half * force_coeff * accel_t
        # Simplified equivalent to: v_half = momentum_half * v + force_coeff * accel_next_t
        v_half = momentum_half * v + force_coeff * accel_next_t
        accel_t = accel_next_t # Update accel_t for the next loop's full-step calculation

        if return_all:
            history.append(x)

    if return_all:
        return torch.stack(history, dim=1)
    else:
        return x, v


def leapfrog_auto(
        x0: torch.Tensor,
        v0: torch.Tensor,
        accel: Callable[[torch.Tensor, float], torch.Tensor],
        dt: float,
        steps: int,
        friction: Union[float, torch.Tensor],
        return_all: bool = False,
        force_coeff=None,
        momentum=None,
        t_start=0.0,
):
    """
    Leapfrog (velocity-Verlet) integrator with friction.
    """
    x = x0
    v = v0
    t = t_start
    friction = resolve_gamma(friction)

    # Half-step decay factor for velocity under friction γ:
    #   momentum = exp(-γ · dt/2)
    if isinstance(friction, torch.Tensor):
        momentum = torch.exp(-0.5 * friction * dt)
    else:
        momentum = 1.0 if friction < 1e-6 else np.exp(-0.5 * friction * dt)

    # Based on friction γ, to simulate at finite time, we interpolate force coefficient between the inertial (γ=0) and
    # overdamped (γ→∞) limits:
    #   - Inertial    (momentum=1): coeff = dt/2 (standard Verlet)
    #   - Overdamped  (momentum=0): coeff = 1.0  (unit mobility, v = F)
    if force_coeff is None:
        force_coeff = momentum * (dt / 2.0) + (1.0 - momentum) * 1.0

    if return_all:
        history = [x]

    accel_t = accel(x, t)

    for _ in range(steps):
        # Half-step velocity
        v_half = momentum * v + force_coeff * accel_t

        # Full-step position
        x = x + dt * v_half
        t = t + dt

        # Force at new position — also reused as next iteration's accel_t
        accel_t = accel(x, t)

        # Full-step velocity
        v = momentum * v_half + force_coeff * accel_t

        if return_all:
            history.append(x)

    if return_all:
        return torch.stack(history, dim=1)
    else:
        return x, v

def euler_maruyama_generalized(x0, drift_func, sigma, dt, steps, kill_condition=False, dt_EM=0.001):
    """
    Euler–Maruyama simulation with two modes:

    - kill_condition == True:
        Cross-sectional ("killed") data.
        At each time k*dt, sample fresh initial conditions from x0
        and simulate from 0 -> k*dt.

    - kill_condition == False or None:
        Standard trajectories evolving forward in time.

    No boundary killing, no rejection sampling, no censoring.
    """
    N, d = x0.shape
    device, dtype = x0.device, x0.dtype

    # micro-step setup
    if dt_EM is None or dt_EM >= dt:
        n_sub = 1
    else:
        n_sub = max(1, int(round(dt / dt_EM)))
    dt_sim = dt / n_sub

    sigma_t = torch.as_tensor(sigma, dtype=dtype, device=device)
    dt_sim_t = torch.as_tensor(dt_sim, dtype=dtype, device=device)
    noise_coef = sigma_t * torch.sqrt(dt_sim_t)

    X = torch.empty((N, steps + 1, d), dtype=dtype, device=device)

    if kill_condition is True:
        # empirical initial distribution
        X[:, 0] = x0.clone()

        for k in range(1, steps + 1):
            t_target = k * dt

            # resample initial states from empirical x0
            idx = torch.randint(0, N, (N,), device=device)
            x = x0[idx].clone()

            t = 0.0
            n_micro = k * n_sub
            for _ in range(n_micro):
                drift = drift_func(x, t)
                x = x + drift * dt_sim + noise_coef * torch.randn_like(x)
                t += dt_sim

            X[:, k] = x

        return X

    X[:, 0] = x0.clone()
    t_curr = 0.0

    for k in range(1, steps + 1):
        x = X[:, k - 1].clone()
        t = t_curr

        for _ in range(n_sub):
            drift = drift_func(x, t)
            x = x + drift * dt_sim + noise_coef * torch.randn_like(x)
            t += dt_sim

        X[:, k] = x
        t_curr += dt

    return X


class BundleVelocityProvider:
    """
    Returns the velocity at time t0 for each particle in x, as per the saved bundle from data_generator.py
    """

    def __init__(self, V_em: torch.Tensor, time_grid: torch.Tensor):
        self.V_em = V_em
        self.time_grid = time_grid

    @torch.no_grad()
    def __call__(
        self,
        p0_idx: int,
        x: torch.Tensor,
        t0: float,
        *,
        x_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Find time index
        tg = self.time_grid
        m = int(torch.argmin((tg - float(t0)).abs()).item())
        m = max(0, min(m, tg.numel() - 1))

        # 2. NaN-filter v at time m, preserving original particle order
        v_raw = self.V_em[int(p0_idx), :, m, :]  # (N_max, d)
        mask = torch.isfinite(v_raw).all(dim=-1)
        v = v_raw[mask]                          # (N_valid, d)

        # 3. If caller provided indices into NaN-filtered x, apply same to v
        if x_idx is not None:
            if x_idx.numel() != x.shape[0]:
                raise RuntimeError(
                    f"BundleVelocityProvider: x_idx has {x_idx.numel()} entries but "
                    f"x has {x.shape[0]} rows."
                )
            v = v[x_idx]
        else:
            # No indices passed → caller is using the full NaN-filtered population.
            if v.shape[0] != x.shape[0]:
                raise RuntimeError(
                    f"BundleVelocityProvider: v has {v.shape[0]} valid particles but "
                    f"x has {x.shape[0]}. Pass x_idx=indices_into_filtered_x if you "
                    f"subsampled, or upstream-align NaN masks of X_em and V_em."
                )

        return v.to(device=x.device, dtype=x.dtype)

class DiceVelocityProvider:
    def __init__(self, models: List[torch.nn.Module]):
        self.models = models

    def __call__(
        self,
        p0_idx: int,
        x: torch.Tensor,
        t0: float,
        *,
        x_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x_idx is ignored: DICE recomputes the score from x itself, so per-particle
        # correspondence is automatically preserved.
        del x_idx
        m = self.models[int(p0_idx)]
        m.eval()
        with torch.enable_grad():
            t = torch.full((x.shape[0], 1), float(t0), device=x.device, dtype=x.dtype)
            x_req = x.detach().requires_grad_(True)
            t_req = t.detach().requires_grad_(True)
            _, grad_s_x, _ = get_s_derivatives(m, x_req, t_req)
            return grad_s_x.detach()

def build_vel_provider(
        vel_mode: str,
        meta: Dict[str, Any],
        dice_models: Optional[List[torch.nn.Module]],
        *,
        V_em: Optional[torch.Tensor] = None,
        time_grid: Optional[torch.Tensor] = None,
) -> Optional[Callable[[int, torch.Tensor, float], torch.Tensor]]:
    """
    Build velocity provider.

    Options:
    - zero: Always return zero velocity
    - dice: Use DICE score models
    - bundle: Use ground truth velocities from data bundle
    """
    mode = str(vel_mode).lower()

    if mode == "zero":
        return lambda p0_idx, x, t0, x_idx=None: torch.zeros_like(x)

    if mode == "dice":
        if dice_models is None:
            raise ValueError("--vel dice requires DICE models (--dice-path or trained via HJ).")
        return DiceVelocityProvider(dice_models)

    if mode == "bundle":
        if V_em is None or time_grid is None:
            raise ValueError("--vel bundle requires V_em_torch in the data bundle. "
                             "Ensure data_generator.py computed velocities (vel_mode != None).")
        return BundleVelocityProvider(V_em, time_grid)

    raise ValueError(f"Unknown vel_mode: {mode}. Must be one of: zero, dice, bundle")