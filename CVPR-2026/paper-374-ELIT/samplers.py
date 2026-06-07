import torch
import numpy as np


def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score


def compute_diffusion(t_cur):
    return 2 * t_cur


def _velocity_step(model, x, t_val, y_cond, y_null, _dtype, device,
                   cfg_scale, guidance_low, guidance_high,
                   inference_budget, unconditional_inference_budget,
                   autoguidance=False):
    """Single velocity prediction with CFG / CCFG / autoguidance support.

    Autoguidance keeps the real class label on the "unconditional" path but
    runs it at the lower ``unconditional_inference_budget``, so guidance
    comes from the capacity gap rather than from dropping the condition.
    """
    in_guidance = cfg_scale > 1.0 and t_val <= guidance_high and t_val >= guidance_low

    if in_guidance and unconditional_inference_budget is not None:
        t_in = torch.ones(x.size(0), device=device, dtype=torch.float64) * t_val
        d_cond = model(x.to(_dtype), t_in.to(_dtype),
                       y=y_cond, inference_budget=inference_budget)[0].to(torch.float64)
        y_low = y_cond if autoguidance else y_null
        d_uncond = model(x.to(_dtype), t_in.to(_dtype),
                         y=y_low, inference_budget=unconditional_inference_budget)[0].to(torch.float64)
        return d_uncond + cfg_scale * (d_cond - d_uncond)

    if in_guidance:
        model_input = torch.cat([x, x], dim=0)
        y_cur = torch.cat([y_cond, y_null], dim=0)
    else:
        model_input = x
        y_cur = y_cond

    t_in = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_val
    d = model(model_input.to(_dtype), t_in.to(_dtype),
              y=y_cur, inference_budget=inference_budget)[0].to(torch.float64)

    if in_guidance:
        d_cond, d_uncond = d.chunk(2)
        d = d_uncond + cfg_scale * (d_cond - d_uncond)
    return d


def _drift_step_sde(model, x, t_val, y_cond, y_null, _dtype, device,
                    cfg_scale, guidance_low, guidance_high,
                    inference_budget, unconditional_inference_budget,
                    path_type, diffusion, autoguidance=False):
    """Single drift prediction for SDE with CFG / CCFG / autoguidance support."""
    in_guidance = cfg_scale > 1.0 and t_val <= guidance_high and t_val >= guidance_low

    if in_guidance and unconditional_inference_budget is not None:
        t_in = torch.ones(x.size(0), device=device, dtype=torch.float64) * t_val
        x_f64 = x.to(torch.float64)
        v_cond = model(x.to(_dtype), t_in.to(_dtype),
                       y=y_cond, inference_budget=inference_budget)[0].to(torch.float64)
        s_cond = get_score_from_velocity(v_cond, x_f64, t_in, path_type=path_type)
        d_cond = v_cond - 0.5 * diffusion * s_cond
        y_low = y_cond if autoguidance else y_null
        v_uncond = model(x.to(_dtype), t_in.to(_dtype),
                         y=y_low, inference_budget=unconditional_inference_budget)[0].to(torch.float64)
        s_uncond = get_score_from_velocity(v_uncond, x_f64, t_in, path_type=path_type)
        d_uncond = v_uncond - 0.5 * diffusion * s_uncond
        return d_uncond + cfg_scale * (d_cond - d_uncond)

    if in_guidance:
        model_input = torch.cat([x, x], dim=0)
        y_cur = torch.cat([y_cond, y_null], dim=0)
    else:
        model_input = x
        y_cur = y_cond

    t_in = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_val
    v = model(model_input.to(_dtype), t_in.to(_dtype),
              y=y_cur, inference_budget=inference_budget)[0].to(torch.float64)
    s = get_score_from_velocity(v, model_input.to(torch.float64), t_in, path_type=path_type)
    d = v - 0.5 * diffusion * s

    if in_guidance:
        d_cond, d_uncond = d.chunk(2)
        d = d_uncond + cfg_scale * (d_cond - d_uncond)
    return d



def _compute_cfg_schedule(cfg_scale, num_steps, schedule="constant", beta_a=3, beta_b=3):
    """Compute per-step CFG scale based on schedule type.
    
    Args:
        cfg_scale: Base CFG scale
        num_steps: Number of sampling steps
        schedule: "constant", "beta", or "linear"
        beta_a, beta_b: Beta distribution parameters
    
    Returns:
        List of cfg scales, one per step
    """
    if schedule == "constant" or cfg_scale <= 1.0:
        return [cfg_scale] * num_steps
    
    t_vals = torch.linspace(0, 1, num_steps + 1)[:-1]  # t at each step start
    
    if schedule == "beta":
        # Beta distribution: peak at middle, zero at edges
        import math
        # Use beta PDF-like curve: t^(a-1) * (1-t)^(b-1)
        raw = (t_vals ** (beta_a - 1)) * ((1 - t_vals) ** (beta_b - 1))
        max_val = raw.max()
        if max_val > 0:
            raw = raw / max_val  # normalize to [0, 1]
        scales = 1.0 + (cfg_scale - 1.0) * raw
    elif schedule == "linear":
        # Linear ramp: min at edges, max at center
        dist_from_center = 2.0 * torch.abs(t_vals - 0.5)  # 0 at center, 1 at edges
        ramp = 1.0 - dist_from_center  # 1 at center, 0 at edges
        ramp = torch.clamp(ramp, 0.0, 1.0)
        scales = 1.0 + (cfg_scale - 1.0) * ramp
    else:
        scales = torch.ones(num_steps) * cfg_scale
    
    return scales.tolist()



def compute_timesteps(num_steps, schedule="uniform", rho=7):
    """Compute timesteps for ODE solver.
    
    Args:
        num_steps: Number of steps
        schedule: "uniform", "quadratic", or "karras"
        rho: Power for quadratic schedule (>1 clusters at t=1)
    
    Returns:
        Tensor of timesteps from 1 to 0, length num_steps+1
    """
    if schedule == "uniform":
        return torch.linspace(1, 0, num_steps + 1, dtype=torch.float64)
    elif schedule == "quadratic":
        # Cluster points near t=1 (early denoising)
        # t_i = (1 - i/N)^rho → denser at t=1
        i_vals = torch.linspace(0, num_steps, num_steps + 1, dtype=torch.float64)
        t_steps = (1.0 - i_vals / num_steps) ** rho
        return t_steps
    elif schedule == "karras":
        # Karras et al. style: t_i = (t_max^(1/rho) + i/N * (t_min^(1/rho) - t_max^(1/rho)))^rho
        t_max = 1.0
        t_min = 0.0
        i_vals = torch.linspace(0, num_steps, num_steps + 1, dtype=torch.float64)
        rho_inv = 1.0 / rho
        t_steps = (t_max ** rho_inv + i_vals / num_steps * (t_min ** rho_inv - t_max ** rho_inv)) ** rho
        return t_steps
    else:
        return torch.linspace(1, 0, num_steps + 1, dtype=torch.float64)

def euler_sampler(
        model, latents, y,
        num_steps=20, heun=False,
        cfg_scale=1.0, guidance_low=0.0, guidance_high=1.0,
        path_type="linear",
        inference_budget=None, unconditional_inference_budget=None,
        autoguidance=False,
        cfg_schedule="constant",
        cfg_beta_a=3,
        cfg_beta_b=3,
        timestep_schedule="uniform",
        timestep_rho=7,
        ):
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
    else:
        y_null = None
    _dtype = latents.dtype
    t_steps = compute_timesteps(num_steps, schedule=timestep_schedule, rho=timestep_rho)
    x_next = latents.to(torch.float64)
    device = x_next.device

    # Compute per-step CFG scales
    cfg_scales = _compute_cfg_schedule(cfg_scale, num_steps, cfg_schedule, cfg_beta_a, cfg_beta_b)

    fwd = dict(model=model, y_cond=y, y_null=y_null, _dtype=_dtype, device=device,
               guidance_low=guidance_low, guidance_high=guidance_high,
               inference_budget=inference_budget,
               unconditional_inference_budget=unconditional_inference_budget,
               autoguidance=autoguidance)

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            step_cfg = cfg_scales[i]
            x_cur = x_next
            d_cur = _velocity_step(x=x_cur, t_val=t_cur, cfg_scale=step_cfg, **fwd)
            x_next = x_cur + (t_next - t_cur) * d_cur

            if heun and (i < num_steps - 1):
                d_prime = _velocity_step(x=x_next, t_val=t_next, cfg_scale=step_cfg, **fwd)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def euler_maruyama_sampler(
        model, latents, y,
        num_steps=20, heun=False,
        cfg_scale=1.0, guidance_low=0.0, guidance_high=1.0,
        path_type="linear",
        inference_budget=None, unconditional_inference_budget=None,
        autoguidance=False,
        ):
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
    else:
        y_null = None

    _dtype = latents.dtype
    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    drift = dict(model=model, y_cond=y, y_null=y_null, _dtype=_dtype, device=device,
                 cfg_scale=cfg_scale, guidance_low=guidance_low, guidance_high=guidance_high,
                 inference_budget=inference_budget,
                 unconditional_inference_budget=unconditional_inference_budget,
                 path_type=path_type, autoguidance=autoguidance)

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
            dt = t_next - t_cur
            x_cur = x_next
            diffusion = compute_diffusion(t_cur)
            eps_i = torch.randn_like(x_cur).to(device)
            deps = eps_i * torch.sqrt(torch.abs(dt))

            d_cur = _drift_step_sde(x=x_cur, t_val=t_cur, diffusion=diffusion, **drift)
            x_next = x_cur + d_cur * dt + torch.sqrt(diffusion) * deps

    # last step (no noise)
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    diffusion = compute_diffusion(t_cur)

    d_cur = _drift_step_sde(x=x_cur, t_val=t_cur, diffusion=diffusion, **drift)
    mean_x = x_cur + dt * d_cur

    return mean_x
