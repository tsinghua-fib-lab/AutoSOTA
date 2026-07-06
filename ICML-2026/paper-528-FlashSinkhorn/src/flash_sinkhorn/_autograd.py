"""Autograd Functions for Sinkhorn OT cost and gradient computation.

Extracted from samples_loss.py following FlashAttention's pattern of
separating autograd machinery from the public API module.

Contains:
- _SinkhornConfig: Frozen dataclass capturing all SamplesLoss configuration
- _SinkhornGradFn: Custom backward for analytical gradient (supports HVP)
- _SinkhornCostFn: Custom backward for OT cost (calls solver + gradient)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from flash_sinkhorn.hvp import geomloss_to_ott_potentials, hvp_x_sqeuclid_from_potentials
from flash_sinkhorn.kernels.sinkhorn_triton_grad_sqeuclid import (
    sinkhorn_geomloss_online_grad_sqeuclid,
)
from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
    sinkhorn_flashstyle_alternating,
    sinkhorn_flashstyle_symmetric,
)


# ---------------------------------------------------------------------------
# Config dataclass: immutable snapshot of SamplesLoss parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SinkhornConfig:
    """Frozen snapshot of SamplesLoss configuration.

    Replaces the closure-captured ``self`` pattern. Stored on ``ctx`` in
    forward so that backward can access all parameters without a closure
    reference to the enclosing ``SamplesLoss`` instance.
    """

    # Solver config
    backend: str
    blur: float
    scaling: float
    use_epsilon_scaling: bool
    eps: Optional[float]
    n_iters: Optional[int]
    cost_scale: float
    rho_x: Optional[float]
    rho_y: Optional[float]
    last_extrapolation: bool
    # Kernel config
    allow_tf32: bool
    use_exp2: bool
    autotune: bool
    block_m: Optional[int]
    block_n: Optional[int]
    block_k: Optional[int]
    num_warps: Optional[int]
    num_stages: int
    # HVP config (used in double backward)
    hvp_tau2: float
    hvp_max_cg_iter: int
    hvp_cg_rtol: float
    hvp_cg_atol: float
    hvp_cg_stabilise_every: int
    hvp_preconditioner: str
    hvp_precond_terms: int
    hvp_use_preconditioner: bool
    # OTDD label cost weights (tensors passed separately to .apply())
    lambda_x: float
    lambda_y: float
    # Early stopping
    threshold: Optional[float]
    inner_iterations: int
    # Adaptive padding: original unpadded sizes for masked early stopping
    n_orig: Optional[int] = None
    m_orig: Optional[int] = None


# ---------------------------------------------------------------------------
# Gradient Function (double backward → HVP)
# ---------------------------------------------------------------------------


class _SinkhornGradFn(torch.autograd.Function):
    """Analytical gradient of the Sinkhorn cost w.r.t. point locations.

    Forward: computes grad_x, grad_y via the gradient kernel.
    Backward: computes Hessian-vector product (HVP) for Newton methods.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        f_grad: torch.Tensor,
        g_grad: torch.Tensor,
        eps: float,
        allow_tf32: bool,
        use_exp2: bool,
        autotune: bool,
        block_m: Optional[int],
        block_n: Optional[int],
        block_k: Optional[int],
        num_warps: Optional[int],
        num_stages: int,
        grad_scale: torch.Tensor,
        hvp_tau2: float,
        hvp_max_cg_iter: int,
        hvp_cg_rtol: float,
        hvp_cg_atol: float,
        hvp_cg_stabilise_every: int,
        hvp_preconditioner: str,
        hvp_precond_terms: int,
        hvp_use_preconditioner: bool,
        compute_grad_x: bool,
        compute_grad_y: bool,
        rho_x: Optional[float] = None,
        rho_y: Optional[float] = None,
        cost_scale: float = 1.0,
        label_x: Optional[torch.Tensor] = None,
        label_y: Optional[torch.Tensor] = None,
        label_cost_matrix: Optional[torch.Tensor] = None,
        lambda_x: float = 1.0,
        lambda_y: float = 0.0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        gx, gy = sinkhorn_geomloss_online_grad_sqeuclid(
            x,
            y,
            a,
            b,
            f_grad,
            g_grad,
            eps=float(eps),
            allow_tf32=bool(allow_tf32),
            use_exp2=bool(use_exp2),
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=int(num_stages),
            autotune=bool(autotune),
            grad_scale=grad_scale,
            compute_grad_x=bool(compute_grad_x),
            compute_grad_y=bool(compute_grad_y),
            cost_scale=float(cost_scale),
            label_x=label_x,
            label_y=label_y,
            label_cost_matrix=label_cost_matrix,
            lambda_x=float(lambda_x),
            lambda_y=float(lambda_y),
        )

        ctx.save_for_backward(x, y, a, b, f_grad, g_grad, grad_scale)
        ctx.eps = float(eps)
        ctx.allow_tf32 = bool(allow_tf32)
        ctx.use_exp2 = bool(use_exp2)
        ctx.autotune = bool(autotune)
        ctx.block_m = block_m
        ctx.block_n = block_n
        ctx.block_k = block_k
        ctx.num_warps = num_warps
        ctx.num_stages = int(num_stages)
        ctx.hvp_tau2 = float(hvp_tau2)
        ctx.hvp_max_cg_iter = int(hvp_max_cg_iter)
        ctx.hvp_cg_rtol = float(hvp_cg_rtol)
        ctx.hvp_cg_atol = float(hvp_cg_atol)
        ctx.hvp_cg_stabilise_every = int(hvp_cg_stabilise_every)
        ctx.hvp_preconditioner = str(hvp_preconditioner)
        ctx.hvp_precond_terms = int(hvp_precond_terms)
        ctx.hvp_use_preconditioner = bool(hvp_use_preconditioner)
        ctx.rho_x = rho_x
        ctx.rho_y = rho_y
        ctx.cost_scale = float(cost_scale)
        ctx.use_label_cost = (
            label_x is not None and label_y is not None
            and label_cost_matrix is not None and lambda_y != 0.0
        )
        return gx, gy

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_grad_x: Optional[torch.Tensor], grad_grad_y: Optional[torch.Tensor]
    ):
        x, y, a, b, f_grad, g_grad, grad_scale = ctx.saved_tensors

        if grad_grad_y is not None:
            raise NotImplementedError("Double backward w.r.t y is not implemented yet.")

        if ctx.use_label_cost:
            raise NotImplementedError(
                "Double backward (HVP) is not supported with OTDD label-augmented cost. "
                "Use gradient flow without HVP."
            )

        out_x = None
        if grad_grad_x is not None:
            if grad_grad_x.shape != x.shape:
                raise ValueError("grad_grad_x must have the same shape as x.")
            if not grad_grad_x.is_cuda:
                raise ValueError("grad_grad_x must be a CUDA tensor.")

            # Use no_grad to prevent autograd from tracking tensor operations
            # during backward. Triton autotune mutates tensor metadata (version
            # counter), which triggers unnecessary autograd bookkeeping without
            # this guard.
            # Follows FlashAttention's pattern (flash_attn_func.py line 1040).
            with torch.no_grad():
                f_hat, g_hat = geomloss_to_ott_potentials(
                    f_grad, g_grad, a, b, eps=ctx.eps
                )
                hvp_x, _ = hvp_x_sqeuclid_from_potentials(
                    x,
                    y,
                    f_hat,
                    g_hat,
                    grad_grad_x,
                    eps=ctx.eps,
                    rho_x=ctx.rho_x,
                    rho_y=ctx.rho_y,
                    cost_scale=ctx.cost_scale,
                    tau2=ctx.hvp_tau2,
                    max_cg_iter=ctx.hvp_max_cg_iter,
                    cg_rtol=ctx.hvp_cg_rtol,
                    cg_atol=ctx.hvp_cg_atol,
                    cg_stabilise_every=ctx.hvp_cg_stabilise_every,
                    preconditioner=ctx.hvp_preconditioner,
                    precond_terms=ctx.hvp_precond_terms,
                    use_preconditioner=ctx.hvp_use_preconditioner,
                    allow_tf32=False,  # HVP requires full fp32 precision
                    use_exp2=ctx.use_exp2,
                    block_m=ctx.block_m,
                    block_n=ctx.block_n,
                    block_k=ctx.block_k,
                    num_warps=int(ctx.num_warps or 4),
                    num_stages=ctx.num_stages,
                )
                out_x = hvp_x * grad_scale

        # Inputs: x,y,a,b,f_grad,g_grad,eps,allow_tf32,use_exp2,autotune,
        # block_m,block_n,block_k,num_warps,num_stages,grad_scale,
        # hvp_tau2,hvp_max_cg_iter,hvp_cg_rtol,hvp_cg_atol,hvp_cg_stabilise_every,
        # hvp_preconditioner,hvp_precond_terms,hvp_use_preconditioner,
        # compute_grad_x, compute_grad_y, rho_x, rho_y, cost_scale,
        # label_x, label_y, label_cost_matrix, lambda_x, lambda_y
        return (
            out_x,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # rho_x
            None,  # rho_y
            None,  # cost_scale
            None,  # label_x
            None,  # label_y
            None,  # label_cost_matrix
            None,  # lambda_x
            None,  # lambda_y
        )


# ---------------------------------------------------------------------------
# Cost Function (forward: run Sinkhorn solver, backward: analytical gradient)
# ---------------------------------------------------------------------------


class _SinkhornCostFn(torch.autograd.Function):
    """Compute Sinkhorn OT cost with analytical backward pass.

    This is a module-level autograd.Function (following FlashAttention's pattern).
    All configuration is passed via a frozen ``_SinkhornConfig`` dataclass rather
    than captured via closure.

    Args to .apply():
        x, y, a, b: Point clouds and weights (tensors, tracked by autograd)
        eps_list: Epsilon schedule (tuple of floats, non-tensor)
        config: _SinkhornConfig (frozen dataclass, non-tensor)
        label_x, label_y: Optional OTDD label tensors (may be None)
        label_cost_matrix: Optional (V, V) label distance matrix (may be None)
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        eps_list: Tuple[float, ...],
        config: _SinkhornConfig,
        label_x: Optional[torch.Tensor],
        label_y: Optional[torch.Tensor],
        label_cost_matrix: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # OTT backend: use alternating-update Sinkhorn (matches OTT-JAX)
        if config.backend == "alternating":
            eps = float(eps_list[-1])  # Fixed eps for OTT backend
            n_iters = len(eps_list)

            f_cost, g_cost = sinkhorn_flashstyle_alternating(
                x,
                y,
                a,
                b,
                eps=eps,
                n_iters=n_iters,
                cost_scale=config.cost_scale,
                rho_x=config.rho_x,
                rho_y=config.rho_y,
                autotune=config.autotune,
                allow_tf32=config.allow_tf32,
                use_exp2=config.use_exp2,
                n_orig=config.n_orig,
                m_orig=config.m_orig,
            )
            f_grad, g_grad = f_cost, g_cost

            ctx.save_for_backward(x, y, a, b, f_cost, g_cost, f_grad, g_grad)
            ctx.eps = eps
            ctx.rho_x = config.rho_x
            ctx.rho_y = config.rho_y
            ctx.allow_tf32 = config.allow_tf32
            ctx.use_exp2 = config.use_exp2
            ctx.autotune = config.autotune
            ctx.block_m = config.block_m
            ctx.block_n = config.block_n
            ctx.block_k = config.block_k
            ctx.num_warps = config.num_warps
            ctx.num_stages = config.num_stages
            ctx.label_x = None
            ctx.label_y = None
            ctx.label_cost_matrix_stored = None
            ctx.lambda_x = 1.0
            ctx.lambda_y = 0.0
            ctx.config = config

            return (a * f_cost).sum() + (b * g_cost).sum()

        # GeomLoss backend (default): symmetric-update Sinkhorn
        if config.last_extrapolation:
            f_cost, g_cost, f_grad, g_grad = sinkhorn_flashstyle_symmetric(
                x,
                y,
                a,
                b,
                blur=config.blur,
                scaling=config.scaling,
                use_epsilon_scaling=config.use_epsilon_scaling,
                eps=config.eps,
                n_iters=config.n_iters,
                allow_tf32=config.allow_tf32,
                use_exp2=config.use_exp2,
                autotune=config.autotune,
                cost_scale=config.cost_scale,
                rho_x=config.rho_x,
                rho_y=config.rho_y,
                last_extrapolation=True,
                return_prelast=True,
                label_x=label_x,
                label_y=label_y,
                label_cost_matrix=label_cost_matrix,
                lambda_x=config.lambda_x,
                lambda_y=config.lambda_y,
                threshold=config.threshold,
                check_every=config.inner_iterations,
                n_orig=config.n_orig,
                m_orig=config.m_orig,
            )
        else:
            f_cost, g_cost = sinkhorn_flashstyle_symmetric(
                x,
                y,
                a,
                b,
                blur=config.blur,
                scaling=config.scaling,
                use_epsilon_scaling=config.use_epsilon_scaling,
                eps=config.eps,
                n_iters=config.n_iters,
                allow_tf32=config.allow_tf32,
                use_exp2=config.use_exp2,
                autotune=config.autotune,
                cost_scale=config.cost_scale,
                rho_x=config.rho_x,
                rho_y=config.rho_y,
                last_extrapolation=False,
                label_x=label_x,
                label_y=label_y,
                label_cost_matrix=label_cost_matrix,
                lambda_x=config.lambda_x,
                lambda_y=config.lambda_y,
                threshold=config.threshold,
                check_every=config.inner_iterations,
                n_orig=config.n_orig,
                m_orig=config.m_orig,
            )
            f_grad, g_grad = f_cost, g_cost

        ctx.save_for_backward(x, y, a, b, f_cost, g_cost, f_grad, g_grad)
        ctx.eps = float(eps_list[-1])
        ctx.rho_x = config.rho_x
        ctx.rho_y = config.rho_y
        ctx.allow_tf32 = config.allow_tf32
        ctx.use_exp2 = config.use_exp2
        ctx.autotune = config.autotune
        ctx.block_m = config.block_m
        ctx.block_n = config.block_n
        ctx.block_k = config.block_k
        ctx.num_warps = config.num_warps
        ctx.num_stages = config.num_stages
        ctx.label_x = label_x
        ctx.label_y = label_y
        ctx.label_cost_matrix_stored = label_cost_matrix
        ctx.lambda_x = config.lambda_x
        ctx.lambda_y = config.lambda_y
        ctx.config = config

        # Cost computation: differs for balanced vs unbalanced/semi-unbalanced OT
        is_balanced_x = config.rho_x is None
        is_balanced_y = config.rho_y is None

        if is_balanced_x and is_balanced_y:
            return (a * f_cost).sum() + (b * g_cost).sum()
        else:
            is_semi_unbalanced = is_balanced_x != is_balanced_y

            if is_semi_unbalanced:
                return (a * f_cost).sum() + (b * g_cost).sum()
            else:
                cost = torch.tensor(0.0, device=x.device, dtype=torch.float32)
                unbal_weight_x = config.rho_x + ctx.eps / 2
                cost_a = (a * (1 - (-f_cost / config.rho_x).exp())).sum()
                cost = cost + unbal_weight_x * cost_a
                unbal_weight_y = config.rho_y + ctx.eps / 2
                cost_b = (b * (1 - (-g_cost / config.rho_y).exp())).sum()
                cost = cost + unbal_weight_y * cost_b
                return cost

    @staticmethod
    def backward(ctx, grad_out):
        x, y, a, b, f_cost, g_cost, f_grad, g_grad = ctx.saved_tensors
        config = ctx.config
        grad_x = grad_y = grad_a = grad_b = None

        if x.requires_grad or y.requires_grad:
            gx, gy = _SinkhornGradFn.apply(
                x,
                y,
                a,
                b,
                f_grad,
                g_grad,
                ctx.eps,
                ctx.allow_tf32,
                ctx.use_exp2,
                ctx.autotune,
                ctx.block_m,
                ctx.block_n,
                ctx.block_k,
                ctx.num_warps,
                ctx.num_stages,
                grad_out,
                config.hvp_tau2,
                config.hvp_max_cg_iter,
                config.hvp_cg_rtol,
                config.hvp_cg_atol,
                config.hvp_cg_stabilise_every,
                config.hvp_preconditioner,
                config.hvp_precond_terms,
                config.hvp_use_preconditioner,
                x.requires_grad,
                y.requires_grad,
                ctx.rho_x,
                ctx.rho_y,
                config.cost_scale,
                ctx.label_x,
                ctx.label_y,
                ctx.label_cost_matrix_stored,
                ctx.lambda_x,
                ctx.lambda_y,
            )
            grad_x = gx if x.requires_grad else None
            grad_y = gy if y.requires_grad else None

        if a.requires_grad:
            grad_a = grad_out * f_cost
        if b.requires_grad:
            grad_b = grad_out * g_cost

        # Returns: x, y, a, b, eps_list, config, label_x, label_y, label_cost_matrix
        return (
            grad_x,
            grad_y,
            grad_a,
            grad_b,
            None,  # eps_list
            None,  # config
            None,  # label_x
            None,  # label_y
            None,  # label_cost_matrix
        )
