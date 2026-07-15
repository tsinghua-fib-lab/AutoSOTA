from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# -----------------------------------------------------------------------------
# torch.func helpers (requested): use torch.func.jvp/vjp when available.
# -----------------------------------------------------------------------------
try:
    from torch.func import jvp as _func_jvp, vjp as _func_vjp  # PyTorch >= 2.0
    _HAS_TORCH_FUNC = True
except Exception:  # noqa: BLE001
    _HAS_TORCH_FUNC = False
    try:
        from functorch import jvp as _func_jvp, vjp as _func_vjp  # type: ignore
        _HAS_TORCH_FUNC = True
    except Exception:  # noqa: BLE001
        _func_jvp = None  # type: ignore
        _func_vjp = None  # type: ignore

from torch.utils.data import DataLoader
from torchvision.utils import save_image
import yaml
import wandb
import setproctitle

import tqdm
import torch.nn as nn

from data import get_dataset
from forward_operators import get_operator, LatentWrapper
from model import get_model
from misc import (
    # Presets extracted from upstream YAMLs (configs/*/*.yaml)
    DATA_PRESETS,
    MODEL_PRESETS,
    SAMPLER_PRESETS,
    TASK_PRESETS,

    # Scheduler registry + evaluation + utilities
    Trajectory,
    get_diffusion_scheduler,
    Evaluator,
    get_eval_fn,
    get_eval_fn_cmp,
    calculate_fid,
    resize,
    safe_dir,
    tensor_to_pils,
    save_mp4_video,
)


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inner product over all entries (assumes same shape)."""
    return (a.reshape(-1) * b.reshape(-1)).sum()

def gmres_solve(matvec, b: torch.Tensor, K: int, tol: float = 1e-12) -> torch.Tensor:
    """GMRES for M(Δ)=b with a matrix-free matvec.

    This implements Alg. (GMRES) in the appendix, using classical Arnoldi with
    modified Gram–Schmidt.

    Args:
        matvec: callable(v) -> M(v) with same shape as b.
        b: RHS tensor.
        K: Krylov budget.
        tol: breakdown tolerance.

    Returns:
        Δ tensor with same shape as b.
    """
    # Δ0 = 0; here matvec is linear, so r0 = b - matvec(0) = b.
    r0 = b
    beta = torch.norm(r0.reshape(-1), p=2)
    if beta.item() < tol:
        return torch.zeros_like(b)

    V: List[torch.Tensor] = []
    V.append(r0 / beta)

    # Hessenberg (K+1, K)
    H = torch.zeros((K + 1, K), device=b.device, dtype=b.dtype)

    m = K
    for j in range(K):
        w = matvec(V[j])
        for l in range(j + 1):
            h_lj = _dot(V[l], w)
            H[l, j] = h_lj
            w = w - h_lj * V[l]

        h_next = torch.norm(w.reshape(-1), p=2)
        H[j + 1, j] = h_next
        if h_next.item() < tol:
            m = j + 1
            break
        V.append(w / h_next)

    # Solve least squares min ||Hbar y - e1|| (stay on-device to avoid host transfers).
    Hbar = H[: m + 1, :m]
    e1 = torch.zeros((m + 1, 1), device=Hbar.device, dtype=Hbar.dtype)
    e1[0, 0] = beta.to(dtype=Hbar.dtype)

    y = torch.linalg.lstsq(Hbar, e1).solution.squeeze(1).to(dtype=b.dtype)  # (m,)

    delta = torch.zeros_like(b)
    for j in range(m):
        delta = delta + y[j] * V[j]
    return delta


class CLAMP(nn.Module):
    """CLAMP sampler used by the release entry point."""

    def __init__(
        self,
        annealing_scheduler_config: Dict[str, Any],
        *,
        lambda_id: float = 2.0,
        gmres_iter: int = 5,
        gmres_tol: float = 1e-12,
        sigma_n: float = 0.01,
        noise_sigma: Optional[float] = None,
        lam_id: Optional[float] = None,
        prior_scale: float = 1.0,
        op_case: str = "auto",
        adjoint_mode: str = "autograd",
        metric_mode: str = "score",
        proj_cg: bool = False,
        proj_sigma: float = 0.0,
        beta: float = 1.0,
        lam_aniso: float = 1.0,
        rank1_direction: str = "aligned",
        log_mechanism_metrics: bool = False,
        direction_seed: int = 0,
        clamp: bool = True,
        latent: bool = False,
        lam_id_alpha: float = 0.0,
        beta_orth: float = None,
        adaptive_gmres: bool = False,
    ):
        super().__init__()
        self.annealing_scheduler = get_diffusion_scheduler(**annealing_scheduler_config)
        self.latent = latent
        if noise_sigma is not None:
            sigma_n = noise_sigma
        if lam_id is not None:
            lambda_id = lam_id
        self.noise_sigma2 = float(sigma_n) ** 2
        self.lam_id = float(lambda_id)
        self.lam_id_alpha = float(lam_id_alpha)
        self.beta_orth = float(beta_orth) if beta_orth is not None else None
        self.adaptive_gmres = bool(adaptive_gmres)
        self.prior_scale = float(prior_scale)
        self.cg_iter = max(1, int(gmres_iter))
        self.cg_tol = float(gmres_tol)
        self.op_case = str(op_case)
        if self.op_case not in {"auto", "linear", "nonlinear"}:
            raise ValueError("op_case must be one of: auto, linear, nonlinear")

        self.adjoint_mode = str(adjoint_mode)
        if self.adjoint_mode not in {"autograd", "explicit"}:
            raise ValueError("adjoint_mode must be one of: autograd, explicit")

        self.metric_mode = str(metric_mode)
        if self.metric_mode not in {"eps", "score"}:
            raise ValueError("metric_mode must be one of: eps, score")

        self.proj_cg = bool(proj_cg)
        self.proj_sigma = float(proj_sigma)
        self.beta = float(beta)
        self.lam_aniso = float(lam_aniso)
        self.rank1_direction = str(rank1_direction)
        self.log_mechanism_metrics = bool(log_mechanism_metrics)
        self.direction_seed = int(direction_seed)
        self.clamp = bool(clamp)

        self.last_sampling_time = 0.0
        self.last_sampling_peak_memory = None
        self.last_sampling_memory = None

    def get_start(self, batch_size: int, model) -> torch.Tensor:
        device = next(model.parameters()).device
        in_shape = model.get_in_shape()
        return torch.randn(batch_size, *in_shape, device=device) * self.annealing_scheduler.get_prior_sigma()

    def _tweedie_with_eps(self, model, x, sigma):
        out = model.tweedie(x, sigma)
        if isinstance(out, (tuple, list)):
            if len(out) >= 2:
                x_pred, eps = out[0], out[1]
            else:
                x_pred = out[0]
                sigma_t = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
                eps = (x - x_pred) / torch.clamp(sigma_t, min=1e-12)
        else:
            x_pred = out
            sigma_t = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
            eps = (x - x_pred) / torch.clamp(sigma_t, min=1e-12)
        return x_pred, eps

    @torch.no_grad()
    def sample(self, model, x_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        if record:
            self.trajectory = Trajectory()

        sigmas = self.annealing_scheduler.sigma_steps
        sigma_min_pos = float(sigmas[-2].item()) if torch.is_tensor(sigmas[-2]) else float(sigmas[-2])

        total_sampling_time = 0.0
        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(self.annealing_scheduler.num_steps - 1)
        xt = x_start
        sampling_peak_memory = None
        cuda_device = xt.device if xt.is_cuda else None
        if cuda_device is not None:
            torch.cuda.reset_peak_memory_stats(cuda_device)

        old_op_case = self.op_case
        if self.op_case == "auto":
            self.op_case, self.op_name = self._infer_op_case(operator)

        self.A = operator
        try:
            for step in pbar:
                sigma = float(torch.as_tensor(self.annealing_scheduler.sigma_steps[step]).item())
                sigma_next = float(torch.as_tensor(self.annealing_scheduler.sigma_steps[step + 1]).item())
                sigma_dev = torch.as_tensor(sigma, device=xt.device, dtype=xt.dtype)

                t0 = time.perf_counter()
                sigma_prior = max(sigma_next, sigma_min_pos, 1e-12)

                def x_eval_fn(z):
                    return self._tweedie_with_eps(model, z, sigma_dev)

                cur_gmres_iter = self._get_gmres_iter(sigma)
                Delta, g, x0hat, eps, g_unit = self._penalized_gn_step(
                    x_ref=xt,
                    y=measurement,
                    sigma_prior=float(sigma_prior),
                    x_eval_fn=x_eval_fn,
                    gmres_iter=cur_gmres_iter,
                )
                # Directional decomposition: scale aligned and orthogonal components independently
                if self.beta_orth is not None and g_unit is not None:
                    Delta_aligned = torch.dot(Delta.flatten(), g_unit.flatten()) * g_unit
                    Delta_orth = Delta - Delta_aligned
                    Delta = self.beta * Delta_aligned + self.beta_orth * Delta_orth
                else:
                    Delta *= self.beta

                if self.clamp:
                    x0hat = x0hat.clamp(-1, 1)

                eps_theta = eps
                _, sigma_up, sigma_down = self.sigma_split_geom_ref(sigma=float(sigma), sigma_next=float(sigma_next))
                xt = xt + Delta + (sigma_down - sigma) * eps_theta + sigma_up * torch.randn_like(x0hat)

                total_sampling_time += (time.perf_counter() - t0)
                if cuda_device is not None:
                    current_peak = torch.cuda.max_memory_allocated(device=cuda_device) / (1024 ** 3)
                    sampling_peak_memory = current_peak if sampling_peak_memory is None else max(sampling_peak_memory, current_peak)

                if sigma_next > 0.0:
                    x0y, _ = self._tweedie_with_eps(model, xt, sigma_next)
                else:
                    x0y = xt

                x0hat_results = x0y_results = {}
                if evaluator and "gt" in kwargs:
                    with torch.no_grad():
                        gt = kwargs["gt"]
                        x0hat_results = evaluator(gt, measurement, x0hat)
                        x0y_results = evaluator(gt, measurement, x0y)
                    if verbose and hasattr(pbar, "set_postfix"):
                        main_eval_fn_name = evaluator.main_eval_fn_name
                        pbar.set_postfix(
                            {
                                "x0hat" + "_" + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                                "x0y" + "_" + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                            }
                        )
                if record:
                    self._record(xt, x0y, x0hat, sigma, x0hat_results, x0y_results)
        finally:
            self.op_case = old_op_case

        self.last_sampling_time = float(total_sampling_time)
        self.last_sampling_peak_memory = sampling_peak_memory
        self.last_sampling_memory = sampling_peak_memory
        return xt

    def _get_gmres_iter(self, sigma):
        if not self.adaptive_gmres:
            return self.cg_iter
        s = float(sigma)
        if s > 50.0:
            return max(2, self.cg_iter - 2)
        elif s > 1.0:
            return self.cg_iter + 3
        else:
            return max(2, self.cg_iter - 1)

    def _record(self, xt, x0y, x0hat, sigma, x0hat_results, x0y_results):
        self.trajectory.add_tensor("xt", xt)
        self.trajectory.add_tensor("x0y", x0y)
        self.trajectory.add_tensor("x0hat", x0hat)
        self.trajectory.add_value("sigma", sigma)
        for name in x0hat_results.keys():
            self.trajectory.add_value(f"x0hat_{name}", x0hat_results[name])
        for name in x0y_results.keys():
            self.trajectory.add_value(f"x0y_{name}", x0y_results[name])

    def _infer_op_case(self, operator):
        op_obj = getattr(operator, "op", operator)
        name = getattr(op_obj, "name", None)
        if not isinstance(name, str):
            raise ValueError(
                "op_case='auto' requires operator to have a string `.name` attribute "
                "(expected CLAMP forward operator registration)."
            )
        if name in {"down_sampling", "inpainting", "gaussian_blur", "motion_blur"}:
            return "linear", name
        if name in {"phase_retrieval", "nonlinear_blur", "high_dynamic_range"}:
            return "nonlinear", name
        raise ValueError(
            f"Unknown operator name '{name}' for op_case='auto'.\n"
            "Expected one of:\n"
            "  linear: down_sampling, inpainting, gaussian_blur, motion_blur\n"
            "  nonlinear: phase_retrieval, nonlinear_blur, high_dynamic_range\n"
            "Fix: set sampler.op_case explicitly (linear/nonlinear) or register/rename the operator."
        )

    def sigma_split_geom_ref(self, sigma, sigma_next):
        sigma_down = sigma - np.sqrt(max(sigma ** 2 - sigma_next ** 2, 0.0))
        sigma_up = np.sqrt(max(sigma_next ** 2 - sigma_down ** 2, 0.0))
        lam = 1.0 - sigma_down / sigma
        return lam, sigma_up, sigma_down

    def A_forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.A, "forward") and callable(getattr(self.A, "forward")):
            return self.A.forward(x)
        return self.A(x)

    def _penalized_gn_step(self, *, x_ref, y, sigma_prior, x_eval_fn=None, gmres_iter=None):
        if _func_vjp is None or _func_jvp is None:
            raise RuntimeError("torch.func.vjp/jvp is required for CLAMP. Please upgrade to PyTorch >= 2.0.")

        sigma_n2 = max(float(self.noise_sigma2), 1e-12)
        x_req = x_ref.detach().requires_grad_(True)

        model_output, model_vjp_fn = _func_vjp(x_eval_fn, x_req)
        x0_hat = model_output[0]
        eps_t = model_output[1].detach()
        Ax, A_vjp_fn = _func_vjp(self.A_forward, x0_hat.detach())

        if self.metric_mode == "eps":
            g = (x_req - x0_hat) / max(sigma_prior, 1e-12)
        else:
            g = (x_req - x0_hat)

        def A_T(u):
            if self.adjoint_mode == "explicit":
                return self.A.rmatvec(x0_hat, u)
            return A_vjp_fn(u)[0]

        def total_vjp(u):
            return model_vjp_fn((A_T(u), torch.zeros_like(eps_t)))[0]

        def total_jvp(v):
            return _func_jvp(self.A_forward, (x0_hat,), (v,))[1].detach()

        res = Ax - y
        c = total_vjp(res / sigma_n2).detach()
        at_res_norm = float(c.flatten().norm().item()) * sigma_n2
        if res.shape[2] == 64:
            at_res_norm = at_res_norm * 16

        def apply_B(v):
            return total_vjp(total_jvp(v) / sigma_n2).detach()

        g_norm = float(g.flatten().norm().item())
        g_unit = (g / (g_norm + 1e-12)).detach() if g_norm > 1e-12 else None
        # Lambda-id scheduling: higher damping at high sigma (noisy curvature), lower at low sigma
        lam_id_eff = self.lam_id if self.lam_id > 0.0 else 1e-8
        if abs(self.lam_id_alpha) > 1e-8:
            import math
            sigma_ref = max(float(sigma_prior), 1e-12)
            lam_id_eff = lam_id_eff * (sigma_ref ** self.lam_id_alpha)

        def apply_H(v):
            out = lam_id_eff * v
            if g_unit is not None:
                out = out + torch.dot(g_unit.flatten(), v.flatten()) * g_unit
            return out

        alpha = (g_norm / max(at_res_norm, 1e-12)) / max(sigma_prior ** 2, 1e-12)

        def matvec(v):
            return apply_B(v) + alpha * apply_H(v)

        return self._gmres_solve(matvec, (-c).detach(), K=gmres_iter), g.detach(), x0_hat.detach(), eps_t.detach(), g_unit

    def _gmres_solve(self, matvec, b: torch.Tensor, K: int = None) -> torch.Tensor:
        device = b.device
        dtype = b.dtype
        shape = b.shape

        b_flat = b.flatten()
        r0 = b_flat
        beta = torch.norm(r0)

        if beta < 1e-12:
            return torch.zeros_like(b)

        v = [r0 / beta]
        K_eff = K if K is not None else self.cg_iter
        h = torch.zeros((K_eff + 1, K_eff), device=device, dtype=dtype)

        for j in range(K_eff):
            v_next_raw = matvec(v[j].view(shape)).flatten()

            for i in range(j + 1):
                h[i, j] = torch.dot(v[i], v_next_raw)
                v_next_raw = v_next_raw - h[i, j] * v[i]

            h[j + 1, j] = torch.norm(v_next_raw)
            if h[j + 1, j] < 1e-12:
                break
            v.append(v_next_raw / h[j + 1, j])

        actual_iter = j + 1
        H_sub = h[:actual_iter + 1, :actual_iter]
        e1 = torch.zeros(actual_iter + 1, device=device, dtype=dtype)
        e1[0] = beta

        y = torch.linalg.lstsq(H_sub, e1.unsqueeze(1)).solution.flatten()

        x_flat = torch.zeros_like(b_flat)
        for i in range(len(y)):
            x_flat += y[i] * v[i]

        return torch.nan_to_num(x_flat.view(shape), nan=0.0, posinf=0.0, neginf=0.0)


class LatentCLAMP(CLAMP):
    @torch.no_grad()
    def sample(self, model, z_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        if record:
            self.trajectory = Trajectory()

        sigmas = self.annealing_scheduler.sigma_steps
        sigma_min_pos = float(sigmas[-2].item()) if torch.is_tensor(sigmas[-2]) else float(sigmas[-2])
        total_sampling_time = 0.0

        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(self.annealing_scheduler.num_steps - 1)
        wrapped_operator = LatentWrapper(operator, model)

        old_op_case = self.op_case
        if self.op_case == "auto":
            self.op_case, self.op_name = self._infer_op_case(wrapped_operator)

        self.A = wrapped_operator
        zt = z_start
        try:
            for step in pbar:
                sigma = float(torch.as_tensor(self.annealing_scheduler.sigma_steps[step]).item())
                sigma_next = float(torch.as_tensor(self.annealing_scheduler.sigma_steps[step + 1]).item())

                t0 = time.perf_counter()
                sigma_dev = torch.as_tensor(sigma, device=zt.device, dtype=zt.dtype)
                sigma_prior = max(sigma_next, sigma_min_pos, 1e-12)

                def z_eval_fn(z):
                    return self._tweedie_with_eps(model, z, sigma_dev)

                Delta, g, z0hat, eps = self._penalized_gn_step(
                    x_ref=zt,
                    y=measurement,
                    sigma_prior=float(sigma_prior),
                    x_eval_fn=z_eval_fn,
                )
                Delta *= self.beta

                eps_theta = eps
                _, sigma_up, sigma_down = self.sigma_split_geom_ref(sigma=float(sigma), sigma_next=float(sigma_next))
                zt = zt + Delta + (sigma_down - sigma) * eps_theta + sigma_up * torch.randn_like(zt)
                total_sampling_time += (time.perf_counter() - t0)

                z0y, _ = self._tweedie_with_eps(model, zt, sigma_next)
                x0hat = model.decode(z0hat)
                x0y = model.decode(z0y)
                xt = model.decode(zt)

                if self.clamp:
                    x0y = x0y.clamp(-1, 1)

                x0hat_results = x0y_results = {}
                if evaluator and "gt" in kwargs:
                    gt = kwargs["gt"]
                    x0hat_results = evaluator(gt, measurement, x0hat)
                    x0y_results = evaluator(gt, measurement, x0y)
                    if verbose and hasattr(pbar, "set_postfix"):
                        main_eval_fn_name = evaluator.main_eval_fn_name
                        pbar.set_postfix(
                            {
                                "x0hat" + "_" + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                                "x0y" + "_" + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                            }
                        )

                if record:
                    self._record(xt, x0y, x0hat, sigma, x0hat_results, x0y_results)
        finally:
            self.op_case = old_op_case

        self.last_sampling_time = float(total_sampling_time)
        self.last_sampling_peak_memory = None
        self.last_sampling_memory = None
        return xt


def get_sampler(
    *,
    method: str,
    latent: bool,
    annealing_scheduler_config: Dict[str, Any],
    diffusion_scheduler_config: Dict[str, Any],
    mcmc_sampler_config: Dict[str, Any],
    clamp_lambda_id: float = 2.0,
    clamp_gmres_iter: int = 5,
    clamp_gmres_tol: float = 1e-12,
    clamp_sigma_n: Optional[float] = None,
    clamp_beta: float = 1.0,
    clamp_metric_mode: str = "score",
    clamp_lam_id_alpha: float = 0.0,
    clamp_beta_orth: float = None,
    clamp_adaptive_gmres: bool = False,
):
    """Factory for CLAMP samplers."""
    method = str(method).lower().strip()
    if method != "clamp":
        raise ValueError(f"Unknown method: {method}. This release only supports 'clamp'.")

    if clamp_sigma_n is None:
        clamp_sigma_n = float(mcmc_sampler_config.get("tau", 0.01))

    sampler_cls = LatentCLAMP if latent else CLAMP
    return sampler_cls(
        annealing_scheduler_config,
        lambda_id=float(clamp_lambda_id),
        gmres_iter=int(clamp_gmres_iter),
        gmres_tol=float(clamp_gmres_tol),
        sigma_n=float(clamp_sigma_n),
        beta=float(clamp_beta),
        metric_mode=str(clamp_metric_mode),
        lam_id_alpha=float(clamp_lam_id_alpha),
        beta_orth=float(clamp_beta_orth) if clamp_beta_orth is not None else None,
        adaptive_gmres=bool(clamp_adaptive_gmres),
    )


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean.")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLAMP 5-file release")

    # General
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", type=str, default="demo")
    p.add_argument("--save_dir", type=str, default="./results")
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--num_runs", type=int, default=1)

    p.add_argument("--wandb", type=_str2bool, default=False)
    p.add_argument("--project_name", type=str, default="CLAMP")

    p.add_argument("--save_samples", type=_str2bool, default=True)
    p.add_argument("--save_traj", type=_str2bool, default=True)
    p.add_argument("--save_traj_video", type=_str2bool, default=False)
    p.add_argument("--save_traj_raw_data", type=_str2bool, default=False)

    p.add_argument("--eval_fid", type=_str2bool, default=False)
    p.add_argument("--eval_fn_list", type=str, default="psnr,ssim,lpips")

    # Sampling method (algorithm)
    p.add_argument(
        "--method",
        type=str,
        choices=["clamp"],
        default="clamp",
        help="Sampling method. This release supports CLAMP.",
    )

    # CLAMP hyper-parameters
    p.add_argument("--clamp_lambda_id", type=float, default=2.0, help="λ_id in (Latent-)CLAMP")
    p.add_argument("--clamp_lam_id_alpha", type=float, default=0.0, help="Lambda-id scheduling exponent: lam_id * sigma^alpha")
    p.add_argument("--clamp_beta_orth", type=float, default=None, help="Beta scale for orthogonal component of Delta (None=disable directional decomp)")
    p.add_argument("--clamp_adaptive_gmres", type=_str2bool, default=False, help="Enable adaptive GMRES iterations per sigma level")
    p.add_argument("--clamp_gmres_iter", type=int, default=5, help="GMRES iterations K in (Latent-)CLAMP")
    p.add_argument("--clamp_gmres_tol", type=float, default=1e-12, help="GMRES breakdown tolerance")
    p.add_argument("--clamp_beta", type=float, default=1.0, help="Scale applied to the CLAMP correction Δ")
    p.add_argument("--clamp_metric_mode", type=str, choices=["score", "eps"], default="score", help="Metric mode for CLAMP")
    p.add_argument(
        "--clamp_sigma_n",
        type=float,
        default=None,
        help="Assumed measurement noise std σ_n for (Latent-)CLAMP. If None, uses --mcmc_tau / task preset tau.",
    )

    # Preset selectors
    p.add_argument("--data", type=str, choices=sorted(DATA_PRESETS.keys()), default="demo-ffhq")
    p.add_argument("--model", type=str, choices=sorted(MODEL_PRESETS.keys()), default="ffhq256ddpm")
    p.add_argument("--sampler", type=str, choices=sorted(SAMPLER_PRESETS.keys()), default="edm_daps")
    p.add_argument("--task", type=str, choices=sorted(TASK_PRESETS.keys()), default="phase_retrieval")
    p.add_argument("--task_group", type=str, choices=["pixel", "pixel_hmc", "ldm", "sd"], default="pixel")

    # Data overrides
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--data_resolution", type=int, default=None)
    p.add_argument("--data_start_id", type=int, default=None)
    p.add_argument("--data_end_id", type=int, default=None)

    # Sampler overrides (common)
    p.add_argument("--anneal_num_steps", type=int, default=None)
    p.add_argument("--diffusion_num_steps", type=int, default=None)
    p.add_argument("--anneal_sigma_max", type=float, default=None)
    p.add_argument("--anneal_sigma_min", type=float, default=None)
    p.add_argument("--timestep", type=str, default=None, help="Override annealing scheduler timestep (e.g., poly-3, poly-4, poly-7)")
    p.add_argument("--diffusion_sigma_min", type=float, default=None)

    # Operator overrides (most common)
    p.add_argument("--operator_sigma", type=float, default=None)
    p.add_argument("--phase_oversample", type=float, default=None)
    p.add_argument("--down_scale_factor", type=int, default=None)
    p.add_argument("--inpaint_mask_len", type=int, default=None)  # for box inpainting
    p.add_argument("--random_inpaint_prob", type=float, default=None)
    p.add_argument("--gaussian_kernel_size", type=int, default=None)
    p.add_argument("--gaussian_intensity", type=float, default=None)
    p.add_argument("--motion_kernel_size", type=int, default=None)
    p.add_argument("--motion_intensity", type=float, default=None)
    p.add_argument("--hdr_scale", type=float, default=None)
    p.add_argument("--bkse_opt_yml_path", type=str, default=None)

    # Task noise overrides
    p.add_argument("--mcmc_num_steps", type=int, default=None)
    p.add_argument("--mcmc_lr", type=float, default=None)
    p.add_argument("--mcmc_tau", type=float, default=None)
    p.add_argument("--mcmc_lr_min_ratio", type=float, default=None)
    p.add_argument("--mcmc_mc_algo", type=str, choices=["langevin", "hmc", "mh"], default=None)
    p.add_argument("--mcmc_prior_solver", type=str, choices=["gaussian", "score-min", "score-t", "exact"], default=None)
    p.add_argument("--mcmc_momentum", type=float, default=None)

    # Model overrides (checkpoint paths)
    p.add_argument("--ddpm_model_path", type=str, default=None)
    p.add_argument("--ldm_diffusion_path", type=str, default=None)
    p.add_argument("--sd_model_id", type=str, default=None)
    p.add_argument("--sd_inner_resolution", type=int, default=None)
    p.add_argument("--sd_target_resolution", type=int, default=None)
    p.add_argument("--sd_guidance_scale", type=float, default=None)
    p.add_argument("--sd_prompt", type=str, default=None)
    p.add_argument("--sd_hf_home", type=str, default=None)

    return p.parse_args()


def _apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Optional[Any]]) -> Dict[str, Any]:
    cfg = dict(cfg)
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def build_configs(args: argparse.Namespace) -> Dict[str, Any]:
    # Data
    data_cfg = dict(DATA_PRESETS[args.data])
    data_cfg = _apply_overrides(data_cfg, {
        "root": args.data_root,
        "resolution": args.data_resolution,
        "start_id": args.data_start_id,
        "end_id": args.data_end_id,
    })

    # Task (operator + mcmc)
    task_cfg = TASK_PRESETS[args.task][args.task_group]
    operator_cfg = dict(task_cfg["operator"])
    mcmc_cfg = dict(task_cfg["mcmc_sampler_config"])

    # operator overrides
    if args.operator_sigma is not None:
        operator_cfg["sigma"] = args.operator_sigma
    if args.phase_oversample is not None and operator_cfg.get("name") == "phase_retrieval":
        operator_cfg["oversample"] = args.phase_oversample
    if args.down_scale_factor is not None and operator_cfg.get("name") == "down_sampling":
        operator_cfg["scale_factor"] = args.down_scale_factor
    if args.inpaint_mask_len is not None and operator_cfg.get("name") == "inpainting":
        # upstream inpainting uses mask_len_range in YAML; for box inpainting we use a fixed len
        operator_cfg["mask_len_range"] = [args.inpaint_mask_len, args.inpaint_mask_len]
    if args.random_inpaint_prob is not None and operator_cfg.get("name") == "inpainting":
        operator_cfg["mask_prob_range"] = [args.random_inpaint_prob, args.random_inpaint_prob]
    if args.gaussian_kernel_size is not None and operator_cfg.get("name") == "gaussian_blur":
        operator_cfg["kernel_size"] = args.gaussian_kernel_size
    if args.gaussian_intensity is not None and operator_cfg.get("name") == "gaussian_blur":
        operator_cfg["intensity"] = args.gaussian_intensity
    if args.motion_kernel_size is not None and operator_cfg.get("name") == "motion_blur":
        operator_cfg["kernel_size"] = args.motion_kernel_size
    if args.motion_intensity is not None and operator_cfg.get("name") == "motion_blur":
        operator_cfg["intensity"] = args.motion_intensity
    if args.hdr_scale is not None and operator_cfg.get("name") == "high_dynamic_range":
        operator_cfg["scale"] = args.hdr_scale
    if args.bkse_opt_yml_path is not None and operator_cfg.get("name") == "nonlinear_blur":
        operator_cfg["opt_yml_path"] = args.bkse_opt_yml_path

    # mcmc overrides
    if args.mcmc_num_steps is not None:
        mcmc_cfg["num_steps"] = args.mcmc_num_steps
    if args.mcmc_lr is not None:
        mcmc_cfg["lr"] = args.mcmc_lr
    if args.mcmc_tau is not None:
        mcmc_cfg["tau"] = args.mcmc_tau
    if args.mcmc_lr_min_ratio is not None:
        mcmc_cfg["lr_min_ratio"] = args.mcmc_lr_min_ratio
    if args.mcmc_mc_algo is not None:
        mcmc_cfg["mc_algo"] = args.mcmc_mc_algo
    if args.mcmc_prior_solver is not None:
        mcmc_cfg["prior_solver"] = args.mcmc_prior_solver
    if args.mcmc_momentum is not None:
        mcmc_cfg["momentum"] = args.mcmc_momentum

    # Sampler
    sampler_cfg = dict(SAMPLER_PRESETS[args.sampler])
    # override step counts + sigma ranges
    if args.anneal_num_steps is not None:
        sampler_cfg["annealing_scheduler_config"] = dict(sampler_cfg["annealing_scheduler_config"])
        sampler_cfg["annealing_scheduler_config"]["num_steps"] = args.anneal_num_steps
    if args.diffusion_num_steps is not None:
        sampler_cfg["diffusion_scheduler_config"] = dict(sampler_cfg["diffusion_scheduler_config"])
        sampler_cfg["diffusion_scheduler_config"]["num_steps"] = args.diffusion_num_steps
    if args.anneal_sigma_max is not None:
        sampler_cfg["annealing_scheduler_config"] = dict(sampler_cfg["annealing_scheduler_config"])
        sampler_cfg["annealing_scheduler_config"]["sigma_max"] = args.anneal_sigma_max
    if args.anneal_sigma_min is not None:
        sampler_cfg["annealing_scheduler_config"] = dict(sampler_cfg["annealing_scheduler_config"])
        sampler_cfg["annealing_scheduler_config"]["sigma_min"] = args.anneal_sigma_min
    if args.timestep is not None:
        sampler_cfg["annealing_scheduler_config"] = dict(sampler_cfg["annealing_scheduler_config"])
        sampler_cfg["annealing_scheduler_config"]["timestep"] = args.timestep
    if args.diffusion_sigma_min is not None:
        sampler_cfg["diffusion_scheduler_config"] = dict(sampler_cfg["diffusion_scheduler_config"])
        sampler_cfg["diffusion_scheduler_config"]["sigma_min"] = args.diffusion_sigma_min

    # Model
    preset_model_cfg = dict(MODEL_PRESETS[args.model])
    model_cfg = dict(preset_model_cfg)

    # checkpoint overrides
    if model_cfg["name"] == "ddpm" and args.ddpm_model_path is not None:
        model_cfg["model_config"] = dict(model_cfg["model_config"])
        model_cfg["model_config"]["model_path"] = args.ddpm_model_path
    if model_cfg["name"] == "ldm" and args.ldm_diffusion_path is not None:
        model_cfg["diffusion_path"] = args.ldm_diffusion_path
    if model_cfg["name"] == "sdm":
        if args.sd_model_id is not None:
            model_cfg["model_id"] = args.sd_model_id
        if args.sd_inner_resolution is not None:
            model_cfg["inner_resolution"] = args.sd_inner_resolution
        if args.sd_target_resolution is not None:
            model_cfg["target_resolution"] = args.sd_target_resolution
        if args.sd_guidance_scale is not None:
            model_cfg["guidance_scale"] = args.sd_guidance_scale
        if args.sd_prompt is not None:
            model_cfg["prompt"] = args.sd_prompt
        if args.sd_hf_home is not None:
            model_cfg["hf_home"] = args.sd_hf_home

    return {
        "data_cfg": data_cfg,
        "operator_cfg": operator_cfg,
        "mcmc_cfg": mcmc_cfg,
        "sampler_cfg": sampler_cfg,
        "model_cfg": model_cfg,
    }


def sample_in_batch(sampler, model, x_start, operator, y, evaluator, args, root, run_id, gt):
    samples = []
    trajs = []
    sample_times = []
    sample_gpu_memory = []
    B = x_start.shape[0]
    for s in range(0, B, args.batch_size):
        cur_x_start = x_start[s:s + args.batch_size]
        cur_y = y[s:s + args.batch_size]
        cur_gt = gt[s:s + args.batch_size]
        cur_samples = sampler.sample(model, cur_x_start, operator, cur_y, evaluator, verbose=True, record=args.save_traj, gt=cur_gt)
        sample_times.append(float(getattr(sampler, "last_sampling_time", 0.0)))
        peak_memory = getattr(sampler, "last_sampling_memory", None)
        if peak_memory is None:
            peak_memory = getattr(sampler, "last_sampling_peak_memory", None)
        if peak_memory is not None:
            sample_gpu_memory.append(float(peak_memory))
        samples.append(cur_samples)
        if args.save_traj:
            cur_trajs = sampler.trajectory.compile()
            trajs.append(cur_trajs)

        # save individual samples
        if args.save_samples:
            pil_image_list = tensor_to_pils(cur_samples)
            image_dir = safe_dir(root / "samples")
            for idx in range(len(pil_image_list)):
                image_path = image_dir / "{:05d}_run{:04d}.png".format(idx + s, run_id)
                pil_image_list[idx].save(str(image_path))

        # save trajectory grids + optional mp4
        if args.save_traj:
            traj_dir = safe_dir(root / "trajectory")
            x0hat_traj = cur_trajs.tensor_data["x0hat"]
            x0y_traj = cur_trajs.tensor_data["x0y"]
            xt_traj = cur_trajs.tensor_data["xt"]
            cur_resized_y = resize(cur_y, cur_samples, operator.name)
            slices = np.linspace(0, len(x0hat_traj) - 1, 10).astype(int)
            slices = np.unique(slices)
            for idx in range(cur_samples.shape[0]):
                if args.save_traj_video:
                    video_path = str(traj_dir / "{:05d}_run{:04d}.mp4".format(idx + s, run_id))
                    save_mp4_video(cur_samples[idx], cur_resized_y[idx], x0hat_traj[:, idx], x0y_traj[:, idx], xt_traj[:, idx], video_path)
                selected_traj_grid = torch.cat([x0y_traj[slices, idx], x0hat_traj[slices, idx], xt_traj[slices, idx]], dim=0)
                traj_grid_path = str(traj_dir / "{:05d}_run{:04d}.png".format(idx + s, run_id))
                save_image(selected_traj_grid * 0.5 + 0.5, fp=traj_grid_path, nrow=len(slices))

    if args.save_traj:
        trajs = Trajectory.merge(trajs)

    avg_time_per_sample = sum(sample_times) / len(sample_times) if sample_times else 0.0
    total_sampling_time = sum(sample_times)
    print(f"Pure sampling time per sample: {avg_time_per_sample:.3f} s")
    print(f"Total pure sampling time: {total_sampling_time:.3f} s")

    avg_gpu_memory = None
    if sample_gpu_memory:
        avg_gpu_memory = sum(sample_gpu_memory) / len(sample_gpu_memory)
        max_gpu_memory = max(sample_gpu_memory)
        print(f"Avg peak CUDA memory per sample: {avg_gpu_memory:.3f} GiB")
        print(f"Max peak CUDA memory across samples: {max_gpu_memory:.3f} GiB")

    return torch.cat(samples, dim=0), trajs, avg_time_per_sample, avg_gpu_memory


def main():
    args = parse_args()

    # Seed + device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(f"cuda:{args.gpu}")

    setproctitle.setproctitle(args.name)

    cfgs = build_configs(args)
    data_cfg = cfgs["data_cfg"]
    operator_cfg = cfgs["operator_cfg"]
    mcmc_cfg = cfgs["mcmc_cfg"]
    sampler_cfg = cfgs["sampler_cfg"]
    model_cfg = cfgs["model_cfg"]

    # Build dataset
    dataset = get_dataset(**data_cfg, device=f"cuda:{args.gpu}")
    total_number = len(dataset)
    images = dataset.get_data(total_number, 0)

    # Operator + measurement
    operator = get_operator(**operator_cfg)
    y = operator.measure(images)

    # Sampler
    sampler = get_sampler(
        method=args.method,
        latent=sampler_cfg["latent"],
        annealing_scheduler_config=sampler_cfg["annealing_scheduler_config"],
        diffusion_scheduler_config=sampler_cfg["diffusion_scheduler_config"],
        mcmc_sampler_config=mcmc_cfg,
        clamp_lambda_id=args.clamp_lambda_id,
        clamp_gmres_iter=args.clamp_gmres_iter,
        clamp_gmres_tol=args.clamp_gmres_tol,
        clamp_sigma_n=args.clamp_sigma_n,
        clamp_beta=args.clamp_beta,
        clamp_metric_mode=args.clamp_metric_mode,
        clamp_lam_id_alpha=args.clamp_lam_id_alpha,
        clamp_beta_orth=args.clamp_beta_orth,
        clamp_adaptive_gmres=args.clamp_adaptive_gmres,
    )

    # Model
    # Device handled inside get_model classes; pass device string to match upstream.
    model_kwargs = dict(model_cfg)
    name = model_kwargs.pop("name")
    model = get_model(name=name, **model_kwargs, device=f"cuda:{args.gpu}")
    if str(args.method).lower().strip() == "clamp":
        model.requires_grad_(True)
        using_torch_func = _HAS_TORCH_FUNC and (_func_vjp is not None and _func_jvp is not None)
        if using_torch_func:
            # Disable embedded checkpoint wrappers for CLAMP+torch.func to avoid recompute overhead.
            def _no_checkpoint(func, inputs, params, flag):
                return func(*inputs)
            try:
                import model.ddpm.nn as ddpm_nn  # type: ignore
                ddpm_nn.checkpoint = _no_checkpoint
            except Exception:
                pass
            try:
                import model.ddpm.unet as ddpm_unet  # type: ignore
                ddpm_unet.checkpoint = _no_checkpoint
            except Exception:
                pass
            try:
                import model.ldm.modules.diffusionmodules.util as ldm_util  # type: ignore
                ldm_util.checkpoint = _no_checkpoint
            except Exception:
                pass

    # Evaluator
    eval_fn_list = [get_eval_fn(n.strip()) for n in args.eval_fn_list.split(",") if n.strip()]
    evaluator = Evaluator(eval_fn_list)

    # Output dir + config dump
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir = safe_dir(Path(args.save_dir))
    root = safe_dir(save_dir / args.name)

    full_config_dump = {
        "args": vars(args),
        "data": data_cfg,
        "operator": operator_cfg,
        "mcmc": mcmc_cfg,
        "sampler": sampler_cfg,
        "model": model_cfg,
    }
    with open(str(root / "config.yaml"), "w") as f:
        yaml.safe_dump(full_config_dump, f, default_flow_style=False, allow_unicode=True)

    if args.wandb:
        wandb.init(project=args.project_name, name=args.name, config=full_config_dump)

    # Main sampling loop
    full_samples = []
    full_trajs = []
    run_avg_times = []
    run_avg_memories = []
    for r in range(args.num_runs):
        print(f"Run: {r}")
        x_start = sampler.get_start(images.shape[0], model)
        samples, trajs, avg_time, avg_memory = sample_in_batch(sampler, model, x_start, operator, y, evaluator, args, root, r, images)
        full_samples.append(samples)
        full_trajs.append(trajs)
        run_avg_times.append(avg_time)
        run_avg_memories.append(avg_memory)
    full_samples = torch.stack(full_samples, dim=0)  # [num_runs, B, C, H, W]

    # Metrics
    results = evaluator.report(images, y, full_samples)
    if args.wandb:
        evaluator.log_wandb(results, args.batch_size)
    markdown_text = evaluator.display(results)
    with open(str(root / "eval.md"), "w") as f:
        f.write(markdown_text)
    print(markdown_text)

    # Grid results
    resized_y = resize(y, images, operator.name)
    stack = torch.cat([images, resized_y, full_samples.flatten(0, 1)])
    save_image(stack * 0.5 + 0.5, fp=str(root / "grid_results.png"), nrow=total_number)

    # Save raw trajectories (can be large)
    if args.save_traj_raw_data:
        traj_dir = safe_dir(root / "trajectory")
        traj_raw_data = safe_dir(traj_dir / "raw")
        for run, sde_traj in enumerate(full_trajs):
            print(f"saving trajectory run {run}...")
            torch.save(sde_traj, str(traj_raw_data / "trajectory_run{:04d}.pth".format(run)))

    # FID
    if args.eval_fid:
        print("Calculating FID...")
        fid_dir = safe_dir(root / "fid")

        eval_fn_cmp = get_eval_fn_cmp(evaluator.main_eval_fn_name)
        eval_values = np.array(results[evaluator.main_eval_fn_name]["sample"])  # [B, num_runs]
        if eval_fn_cmp == "min":
            best_idx = np.argmin(eval_values, axis=1)
        elif eval_fn_cmp == "max":
            best_idx = np.argmax(eval_values, axis=1)
        else:
            raise ValueError(f"Unknown cmp {eval_fn_cmp}")

        best_samples = full_samples[best_idx, np.arange(full_samples.shape[1])]
        best_sample_dir = safe_dir(fid_dir / "best_sample")
        pil_image_list = tensor_to_pils(best_samples)
        for idx in range(len(pil_image_list)):
            image_path = best_sample_dir / "{:05d}.png".format(idx)
            pil_image_list[idx].save(str(image_path))

        fake_dataset = get_dataset(data_cfg["name"], resolution=data_cfg["resolution"], root=str(best_sample_dir), device=f"cuda:{args.gpu}")
        real_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        fake_loader = DataLoader(fake_dataset, batch_size=100, shuffle=False)
        fid_score = calculate_fid(real_loader, fake_loader)
        print(f"FID Score: {fid_score.item():.4f}")
        with open(str(fid_dir / "fid.txt"), "w") as f:
            f.write(f"FID Score: {fid_score.item():.4f}")
        if args.wandb:
            wandb.log({"FID": fid_score.item()})

    overall_avg_time = sum(run_avg_times) / len(run_avg_times) if run_avg_times else 0.0
    overall_stats = {
        "average_time_per_sample_seconds": overall_avg_time,
    }
    valid_memories = [m for m in run_avg_memories if m is not None]
    if valid_memories:
        overall_avg_memory = sum(valid_memories) / len(valid_memories)
        overall_stats["average_peak_cuda_memory_gib"] = overall_avg_memory

    metrics_dict = {
        "metrics": results,
        "overall_statistics": overall_stats,
    }
    json.dump(metrics_dict, open(str(root / "metrics.json"), "w"), indent=4)

    print("\n=== Overall Statistics ===")
    print(f"Average time per sample across {args.num_runs} runs: {overall_avg_time:.3f} s")
    if valid_memories:
        print(f"Average peak CUDA memory per sample across {args.num_runs} runs: {overall_avg_memory:.3f} GiB")

    print(f"finish {args.name}!")


if __name__ == "__main__":
    main()
