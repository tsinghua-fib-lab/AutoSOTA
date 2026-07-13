"""FiniteModel-related helpers and custom OT solvers built on FiniteModel/OT."""

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn
from optimal_transport.ot import OT
from tools.feedback import logger
from models.helpers import FixedFirstIntercept


class TwoStageOptimizer(torch.optim.Optimizer):
    """
    Wrapper that runs Adam with a large learning rate until the gradient norm
    exceeds a threshold (or a maximum warmup step count), then switches to SGD
    with a steady learning rate for the remainder of training.
    """

    def __init__(
        self,
        params,
        steady_lr: float,
        warmup_lr: float,
        betas=(0.5, 0.9),
        grad_threshold: float = 0.5,
        max_warm_steps: int = 200,
        steady_momentum: float = 0.0,
        steady_weight_decay: float = 0.0,
    ):
        if warmup_lr <= 0 or steady_lr <= 0:
            raise ValueError("Learning rates must be positive.")
        if grad_threshold <= 0:
            raise ValueError("grad_threshold must be positive.")
        if max_warm_steps <= 0:
            raise ValueError("max_warm_steps must be positive.")
        if steady_momentum < 0:
            raise ValueError("steady_momentum must be non-negative.")
        if steady_weight_decay < 0:
            raise ValueError("steady_weight_decay must be non-negative.")

        params = list(params)
        if len(params) == 0:
            raise ValueError("Optimizer received an empty parameter list.")

        self._adam = torch.optim.Adam(params, lr=warmup_lr, betas=betas)
        self._sgd = torch.optim.SGD(
            params,
            lr=steady_lr,
            momentum=steady_momentum,
            weight_decay=steady_weight_decay,
        )
        self.warm_lr = warmup_lr
        self.steady_lr = steady_lr
        self.grad_threshold = grad_threshold
        self.max_warm_steps = max_warm_steps
        self.steady_momentum = steady_momentum
        self.steady_weight_decay = steady_weight_decay

        self._use_warm_lr = True
        self._warm_steps = 0
        self.param_groups = self._adam.param_groups
        self.state = self._adam.state

    def zero_grad(self):
        """Zero gradients for both Adam and SGD phases."""
        self._adam.zero_grad()
        self._sgd.zero_grad()

    def state_dict(self):
        """Return optimizer state for checkpointing across warm-to-steady transition."""
        return {
            "adam_state": self._adam.state_dict(),
            "sgd_state": self._sgd.state_dict(),
            "warm_lr": self.warm_lr,
            "steady_lr": self.steady_lr,
            "grad_threshold": self.grad_threshold,
            "max_warm_steps": self.max_warm_steps,
            "steady_momentum": self.steady_momentum,
            "steady_weight_decay": self.steady_weight_decay,
            "use_warm_lr": self._use_warm_lr,
            "warm_steps": self._warm_steps,
        }

    def load_state_dict(self, state_dict):
        """Restore optimizer state and adjust current phase accordingly."""
        self._adam.load_state_dict(state_dict["adam_state"])
        self._sgd.load_state_dict(state_dict["sgd_state"])
        self.warm_lr = state_dict["warm_lr"]
        self.steady_lr = state_dict["steady_lr"]
        self.grad_threshold = state_dict["grad_threshold"]
        self.max_warm_steps = state_dict["max_warm_steps"]
        self.steady_momentum = state_dict.get("steady_momentum", self.steady_momentum)
        self.steady_weight_decay = state_dict.get("steady_weight_decay", self.steady_weight_decay)
        self._use_warm_lr = state_dict["use_warm_lr"]
        self._warm_steps = state_dict["warm_steps"]
        self._set_lr(self.warm_lr if self._use_warm_lr else self.steady_lr)
        self.param_groups = self._adam.param_groups if self._use_warm_lr else self._sgd.param_groups

    def _set_lr(self, lr):
        """Apply same learning rate to both internal optimizers."""
        for group in self._adam.param_groups:
            group["lr"] = lr
        for group in self._sgd.param_groups:
            group["lr"] = lr

    def step(self, closure=None, grad_norm: float | None = None):
        """Perform a parameter update and switch to SGD once warmup criteria are met."""
        if self._use_warm_lr:
            trigger = False
            if grad_norm is not None and grad_norm >= self.grad_threshold:
                trigger = True
            elif self._warm_steps >= self.max_warm_steps:
                trigger = True

            if trigger:
                self._use_warm_lr = False
                self._set_lr(self.steady_lr)
                self.param_groups = self._sgd.param_groups
                self.state = self._sgd.state
                logger.info(
                    "[TwoStageOptimizer] Switching to SGD: grad_norm=%.3f, steps=%d",
                    grad_norm if grad_norm is not None else float("nan"),
                    self._warm_steps,
                )

            self._warm_steps += 1

        if self._use_warm_lr:
            self._adam.step(closure=closure)
        else:
            self._sgd.step(closure=closure)


class FCOTSeparable(OT):
    """
    Finitely-concave OT solver for separable costs using FiniteSeparableModel.
    
    This is a specialized version of FCOT that exploits cost separability:
        c(x,y) = sum_d c_d(x_d, y_d)
    
    Key advantages over standard FCOT:
      - Exponentially fewer parameters: O(d·|Y_0|) vs O(|Y_0|^d)
      - Exact discrete transforms (no numerical optimization needed)
      - Always converges since transforms are solved exactly on grids
    
    Parameterization:
      - Uses FiniteSeparableModel with discretized grids on [-R, R]
      - Kernel is the 1D version of the cost: kernel(x,y) = c(x,y) for scalars
      - Mode is "concave" for c-concave potential representation
    
    Dual objective:
        D = E_x[u(x)] + E_y[u^c(y)]
    
    Example:
        >>> from models import FiniteSeparableModel
        >>> 
        >>> # For L2^2 cost: c(x,y) = ||x-y||^2 = sum_d (x_d - y_d)^2
        >>> def kernel_1d(x, y):
        ...     return -(x - y)**2  # Negative for c-concave
        >>> 
        >>> # Create separable model
        >>> model = FiniteSeparableModel(
        ...     kernel=kernel_1d,
        ...     num_dims=2,
        ...     radius=5.0,
        ...     y_accuracy=0.1,
        ...     x_accuracy=0.1,
        ...     mode="concave"
        ... )
        >>> 
        >>> # Create solver
        >>> fcot_sep = FCOTSeparable(
        ...     input_dim=2,
        ...     model=model,
        ...     inverse_kx=lambda x, p: x - 0.5 * p  # For L2^2
        ... )
        >>> 
        >>> # Or use factory method
        >>> fcot_sep = FCOTSeparable.initialize_right_architecture(
        ...     dim=2,
        ...     radius=5.0,
        ...     n_params=100,           # 50 Y-grid points per dimension
        ...     x_accuracy=0.1,
        ...     kernel_1d=kernel_1d,
        ...     inverse_kx=lambda x, p: x - 0.5 * p
        ... )
    """

    @staticmethod
    def initialize_right_architecture(
        dim: int,
        radius: float,
        n_params: int,
        x_accuracy: float,
        kernel_1d,                   # 1D kernel function c_d(x, y) for scalars
        inverse_kx,                  # Inverse gradient for the full cost
        outer_lr: float = 1e-3,
        betas=(0.5, 0.9),
        device: str = "cpu",
        temp: float = 50.0,
        epsilon: float = 1e-4,
        cache_gradients: bool = False,
        warmup_lr: float | None = None,
        warmup_grad_threshold: float = 0.5,
        warmup_max_steps: int = 200,
        sgd_momentum: float = 0.0,
        sgd_weight_decay: float = 0.0,
        temp_min: float | None = None,
        temp_max: float | None = None,
        temp_warmup_iters: int | None = None,
        reactivate_every: int | None = None,
        reactivate_eps: float = 1e-2,
        full_refresh_every: int | None = None,
        **model_kwargs
    ):
        """
        Factory for building a grid-based separable FCOT solver.

        Parameters
        ----------
        dim : int
            Input/output dimensionality.
        radius : float
            Bound on coordinate magnitude (domain = [-radius, radius]^d).
        n_params : int
            Total number of intercepts (must be divisible by `dim`).
        x_accuracy : float
            Spacing for the X grid used in infimal convolution transforms.
        kernel_1d : callable
            1D kernel function `c_d(x, y)` describing separable costs.
        inverse_kx : callable
            Analytic inverse gradient mapping ∇_x c(x, y) → y used in Monge map.
        cache_gradients : bool
            Whether to precompute derivatives on the grid for faster transforms.
        Other params : See FCOTSeparable.__init__ for further hyperparameters.
        
        Returns
        -------
        FCOTSeparable
            Instantiated solver configured for the requested grid resolution.

        Notes
        -----
        Parameter budget satisfies `n_params = dim × |Y_0|`.
        """
        if n_params % dim != 0:
            raise ValueError(f"n_params={n_params} must be divisible by dim={dim}.")
        ny = n_params // dim
        if ny < 2:
            raise ValueError("Need at least two Y-grid points (n_params/dim >= 2).")
        y_accuracy = (2 * radius) / (ny - 1)
        actual_params = ny * dim
        
        logger.info(f"[FCOT-SEP ARCH] dim={dim}, radius={radius}, ny={ny}")
        logger.info(f"                x_accuracy={x_accuracy:.4f}")
        logger.info(f"                y_accuracy={y_accuracy:.4f}")
        logger.info(f"                actual_params={actual_params}")
        logger.info(f"                cache_gradients={cache_gradients}")
        
        # Build FiniteSeparableModel
        model_kwargs = dict(model_kwargs)
        
        # Backwards compatibility
        if 'lr' in model_kwargs and outer_lr == 1e-3:
            outer_lr = model_kwargs.pop('lr')
            logger.warning("[DEPRECATION] 'lr' argument is deprecated; use 'outer_lr' instead.")

        model = FiniteSeparableModel(
            kernel=kernel_1d,
            num_dims=dim,
            radius=radius,
            y_accuracy=y_accuracy,
            x_accuracy=x_accuracy,
            mode="concave",
            temp=temp,
            epsilon=epsilon,
            cache_gradients=cache_gradients,
            **model_kwargs,
        ).to(device)
        
        return FCOTSeparable(
            input_dim=dim,
            model=model,
            inverse_kx=inverse_kx,
            outer_lr=outer_lr,
            betas=betas,
            device=device,
            warmup_lr=warmup_lr,
            warmup_grad_threshold=warmup_grad_threshold,
            warmup_max_steps=warmup_max_steps,
            sgd_momentum=sgd_momentum,
            sgd_weight_decay=sgd_weight_decay,
            temp_min=temp_min,
            temp_max=temp_max,
            temp_warmup_iters=temp_warmup_iters,
            reactivate_every=reactivate_every,
            reactivate_eps=reactivate_eps,
            full_refresh_every=full_refresh_every,
        )

    def __init__(
        self,
        input_dim: int,
        model: nn.Module,        # FiniteSeparableModel(mode="concave")
        inverse_kx,              # (x, p) ↦ y solving ∇_x k(x,y) = p
        outer_lr: float = 1e-3,
        lr: float | None = None,  # deprecated alias
        betas=(0.5, 0.9),
        device: str = "cpu",
        warmup_lr: float | None = None,
        warmup_grad_threshold: float = 0.5,
        warmup_max_steps: int = 200,
        sgd_momentum: float = 0.0,
        sgd_weight_decay: float = 0.0,
        temp_min: float | None = None,
        temp_max: float | None = None,
        temp_warmup_iters: int | None = None,
        reactivate_every: int | None = None,
        reactivate_eps: float = 1e-2,
        full_refresh_every: int | None = None,
    ):
        """
        Initialize separable FCOT solver over a finite grid.

        Parameters
        ----------
        input_dim : int
            Dimensionality of source/target space.
        model : nn.Module
            FiniteSeparableModel providing u/u^c transforms on a grid.
        inverse_kx : callable
            Mapping (x, ∇u(x)) → y used to evaluate transport map.
        outer_lr : float
            Learning rate for the outer Adam optimizer.
        lr : float | None
            Deprecated alias for `outer_lr`.
        betas : Tuple[float, float]
            Adam beta coefficients.
        temp_min, temp_max, temp_warmup_iters : optional float
            Controls adaptive temperature scheduling for discrete transforms.
        reactivate_every, reactivate_eps, full_refresh_every : int/float
            Hooks that keep inactive intercepts viable by reactivating/refreshing them.
        """
        # Handle deprecated lr parameter before delegating to OT.__init__
        if lr is not None:
            logger.warning("[DEPRECATION] 'lr' argument is deprecated; use 'outer_lr' instead.")
            outer_lr = lr

        super().__init__(outer_lr=outer_lr, inner_lr=None)
        
        self.input_dim = input_dim
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.inverse_kx = inverse_kx
        self.outer_lr = outer_lr
        self.betas = betas
        self.inner_steps = 0
        self.inner_optimizer = None
        self.inner_tol = None
        self.inner_lam = None
        # Flag to skip generic warm-up in base OT (transforms are exact)
        self._skip_warmup = True
        self._temp_min = temp_min
        self._temp_max = temp_max
        self._temp_warmup_iters = temp_warmup_iters
        self._temp_step = 0
        self._grad_activity_eps = 1e-8
        self._reactivate_every = reactivate_every if reactivate_every and reactivate_every > 0 else None
        self._reactivate_eps = reactivate_eps
        self._full_refresh_every = full_refresh_every if full_refresh_every and full_refresh_every > 0 else None
        self._warned_refresh_unavailable = False
        self._step_count = 0

        # Outer optimizer for model parameters (intercepts)
        # Warm-start schedule (TwoStageOptimizer) is no longer used.
        if warmup_lr is not None:
            logger.warning(
                "[DEPRECATION] warmup_* arguments are ignored; using Adam with lr=outer_lr."
            )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=outer_lr,
            betas=betas,
        )
        
        # No inner optimization needed for separable case
        self.raise_on_inner_divergence = False
    
    def _compute_dual_objective(self, X, Y, sample_idx=None):
        """
        Compute the Kantorovich dual objective:
            D = E_x[u(x)] + E_y[u^c(y)]
        
        For separable costs, transforms are computed exactly on grids.
        """
        # First term: E_x[u(x)]
        _, u_X = self.model.forward(X, selection_mode="soft")
        term1 = u_X.mean()
        
        # Second term: E_y[u^c(y)]
        # Note: sample_idx not used for separable model (no warm starts needed)
        _, u_c_Y, converged = self.model.inf_transform(Y)
        term2 = u_c_Y.mean()
        
        # Dual objective to maximize
        dual_obj = term1 + term2
        
        # Separable model always converges (exact discrete optimization)
        inner_converged = converged
        
        return dual_obj, inner_converged

    def _dual_objective(self, x_batch, y_batch, idx_y=None, active_inner_steps=None):
        """
        Evaluate dual objective D = E_x[u(X)] + E_y[u^c(Y)] on a minibatch.

        Returns (D, u_mean, uc_mean, converged) for reporting.
        """
        _, u_vals = self.model.forward(x_batch, selection_mode="soft")
        u_mean = u_vals.mean()

        _, uc_vals, converged = self.model.inf_transform(y_batch)
        uc_mean = uc_vals.mean()

        D = u_mean + uc_mean
        return D, u_mean, uc_mean, converged

    def step(self, x_batch, y_batch, idx_y=None, active_inner_steps=None):
        """
        Perform a gradient step on the separable dual objective with diagnostics.

        Returns dict with dual objective, mean potentials, convergence status, gradient norm,
        and intercept activity statistics for logging.
        """
        self._maybe_update_temperature()
        self.optimizer.zero_grad()
        D, u_mean, uc_mean, converged = self._dual_objective(x_batch, y_batch, idx_y, active_inner_steps)
        #TODO fix ths adhoc scaling
        (-100 * D).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        grad_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm_sq += p.grad.norm().item() ** 2
        grad_norm = grad_norm_sq ** 0.5

        step_kwargs = {}
        if isinstance(self.optimizer, TwoStageOptimizer):
            step_kwargs["grad_norm"] = grad_norm
        self.optimizer.step(**step_kwargs)
        self._step_count += 1
        self._maybe_reactivate(x_batch)
        self._maybe_full_refresh()

        activity = None
        active_count = None
        total_count = None
        intercepts_param = getattr(self.model, "intercepts_param", None)
        theta_grad = None
        if intercepts_param is not None and hasattr(intercepts_param, "theta"):
            theta_grad = intercepts_param.theta.grad
        if theta_grad is not None:
            grad_vals = torch.zeros_like(self.model.intercepts, device=theta_grad.device, dtype=theta_grad.dtype)
            grad_vals[1:, :] = theta_grad.detach()
            active = (grad_vals.abs() > self._grad_activity_eps).sum().item()
            total = grad_vals.numel()
            activity = active / total if total > 0 else None
            active_count = active
            total_count = total

        return {
            "dual": float(D.detach().item()),
            "u_mean": float(u_mean.detach().item()),
            "uc_mean": float(uc_mean.detach().item()),
            "inner_converged": converged,
            "grad_norm": grad_norm,
            "intercept_active_frac": activity,
            "intercept_active_count": active_count,
            "intercept_total": total_count,
        }

    def _maybe_reactivate(self, x_batch):
        """
        Periodically bump inactive intercepts using the kernel transform to avoid stagnation.

        Guidelines:
            - Enabled by setting reactivate_every>0 at init.
            - Uses gradient activity below _grad_activity_eps to decide "inactive".
            - Lifts inactive intercepts toward a small margin (eps) above the
              current kernel lower envelope for the given minibatch.
        """
        if self._reactivate_every is None:
            return
        if self._step_count % self._reactivate_every != 0:
            return

        intercepts_param = getattr(self.model, "intercepts_param", None)
        theta_grad = None
        if intercepts_param is not None and hasattr(intercepts_param, "theta"):
            theta_grad = intercepts_param.theta.grad
        if theta_grad is None:
            return

        intercepts = self.model.intercepts
        full_grad = torch.zeros_like(intercepts, device=theta_grad.device, dtype=theta_grad.dtype)
        full_grad[1:, :] = theta_grad.detach()

        inactive_mask = full_grad.abs() <= self._grad_activity_eps
        if inactive_mask.sum().item() == 0:
            return

        x_batch = x_batch.detach().to(device=self.device, dtype=intercepts.dtype)
        if x_batch.dim() != 2:
            return

        y_grid = self.model.Y_grid.to(device=self.device, dtype=intercepts.dtype)
        eps = float(self._reactivate_eps)
        adjusted = 0

        with torch.no_grad():
            for dim in range(self.model.num_dims):
                mask_dim = inactive_mask[:, dim]
                if not mask_dim.any():
                    continue

                x_vals = x_batch[:, dim]
                kernel_vals = self.model.kernel_fn(
                    x_vals.unsqueeze(-1),
                    y_grid.unsqueeze(0),
                )

                column = intercepts[:, dim]
                scores = kernel_vals - column.unsqueeze(0)
                best_scores = scores.min(dim=1).values
                target = (kernel_vals - best_scores.unsqueeze(1) + eps).min(dim=0).values
                updated_column = torch.where(mask_dim, torch.maximum(column, target), column)
                if torch.any(updated_column != column):
                    if intercepts_param is not None and hasattr(intercepts_param, "set_column_from_raw_"):
                        intercepts_param.set_column_from_raw_(dim, updated_column)
                    else:
                        intercepts[:, dim].copy_(updated_column)
                adjusted += int(mask_dim.sum().item())

        if adjusted > 0:
            logger.info(
                "[FCOT-SEP] Reactivated %d intercepts (eps=%g) at step %d",
                adjusted,
                self._reactivate_eps,
                self._step_count,
            )

    def _maybe_full_refresh(self):
        """
        Periodically refresh intercepts/optimizer momentum and inject jitter.

        Enabled when full_refresh_every>0; calls refresh_intercepts_via_transform,
        resets optimizer momentum on the intercept parameter, and adds tiny noise
        to break ties.
        """
        if self._full_refresh_every is None:
            return
        if self._step_count % self._full_refresh_every != 0:
            return
        
        # Do full refresh + momentum reset + jitter
        self._refresh_reset_jitter()

    
    ###############################################################################
    # Helpers: reset optimizer momentum + jitter after refresh
    ###############################################################################

    @torch.no_grad()
    def _reset_intercept_momentum(self):
        """
        Reset momentum / Adam state ONLY for intercept parameters.
        Works for both Adam and TwoStageOptimizer.
        """
        if not hasattr(self, "optimizer") or self.optimizer is None:
            return
        
        # Identify the underlying parameter that stores intercepts.
        intercepts_param = getattr(self.model, "intercepts_param", None)
        target_param = None
        if intercepts_param is not None and hasattr(intercepts_param, "theta"):
            target_param = intercepts_param.theta
        else:
            target_param = getattr(self.model, "intercepts", None)
        if target_param is None:
            return
        
        # Find optimizer states that correspond to the intercept parameter
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p is target_param:
                    state = self.optimizer.state.get(p, None)
                    if state is not None:
                        state.clear()    # wipes momentum, exp_avg, exp_avg_sq, step
                    return


    @torch.no_grad()
    def _jitter_intercepts(self, strength: float = None):
        """
        Inject tiny Gaussian noise to break ties after b <- b^cc.
        """
        intercepts_param = getattr(self.model, "intercepts_param", None)
        target_param = None
        if intercepts_param is not None and hasattr(intercepts_param, "theta"):
            target_param = intercepts_param.theta
        else:
            target_param = getattr(self.model, "intercepts", None)
        if target_param is None:
            return
        
        # default strength relative to grid resolution
        if strength is None:
            strength = 1e-3 * float(self.model.y_accuracy)   # e.g., 2e-5 for your test
        
        noise = torch.randn_like(target_param) * strength
        target_param.add_(noise)


    @torch.no_grad()
    def _refresh_reset_jitter(self):
        """
        Unified operation: refresh → reset momentum → jitter intercepts.
        Called inside _maybe_full_refresh().
        """
        refresh_fn = getattr(self.model, "refresh_intercepts_via_transform", None)
        if refresh_fn is None:
            return
        
        changed = refresh_fn()
        if changed > 0:
            logger.info(
                f"[FCOT-SEP] Full refresh updated {changed} intercepts at step {self._step_count}"
            )
            self._reset_intercept_momentum()
            self._jitter_intercepts()


    def _maybe_update_temperature(self):
        """Cosine warmup from temp_max → temp_min over temp_warmup_iters steps (if set)."""
        if self._temp_min is None or self._temp_max is None:
            return
        if self._temp_warmup_iters is None or self._temp_warmup_iters <= 0:
            self.model.temp = self._temp_max
            return

        ratio = min(self._temp_step / self._temp_warmup_iters, 1.0)
        cos_scale = 0.5 * (1 - math.cos(math.pi * ratio))
        self.model.temp = self._temp_min + (self._temp_max - self._temp_min) * cos_scale
        self._temp_step += 1

    def save(self, address, iters_done):
        """
        Save model + optimizer checkpoint to disk for caching/resume.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iters_done": iters_done,
                "outer_lr": self.outer_lr,
                "betas": self.betas,
            },
            address,
        )

    def load(self, address):
        """
        Load saved checkpoint and return iterations already completed.
        """
        checkpoint = torch.load(address, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("iters_done", 0)
    
    def transport_X_to_Y(
        self,
        X,
        *,
        selection_mode: str = "hard",
        snap_to_grid: bool = True,
    ):
        """
        Transport samples `X` to `Y` via the c-gradient (Monge map).

        The method differentiates through the softmax-infimal transforms to obtain
        ∇u(x) and then applies the provided `inverse_kx`.
        """
        X = X.to(self.device)
        X.requires_grad_(True)
        
        # Compute u(x) with gradients
        # Use "hard" mode for envelope theorem gradients (more efficient)
        # Use "soft" mode for differentiable selection (smoother but slower)
        _, u_X = self.model.forward(
            X, selection_mode=selection_mode, snap_to_grid=snap_to_grid
        )
        
        # Compute ∇u(x) via backprop
        grad_u = torch.autograd.grad(
            u_X.sum(),
            X,
            create_graph=False
        )[0]
        
        # Apply inverse c-gradient
        Y_pred = self.inverse_kx(X.detach(), grad_u.detach())
        
        return Y_pred
    
    def _fit(
        self,
        X, Y,
        iters_done: int = 0,
        iters: int = 100,
        inner_steps: int | None = None,  # Not used for separable
        print_every: int | None = 1,
        callback=None,
        convergence_tol: float | None = 1e-4,
        convergence_patience: int | None = 10,
        batch_size: int | None = None,
        eval_every: int = 100,
        warmup_steps: int = 0,
        warmup_until_converged: bool = False,
        log_every: int | None = None,
        log_level: str | None = None,
    ):
        """
        Fit procedure delegating logging/optimization to the OT base class.

        inner_steps is always forced to 0 because separable transforms are exact.
        """
        if inner_steps is not None and inner_steps != 0:
            logger.warning(
                f"[FCOT-SEP] inner_steps={inner_steps} ignored for separable model "
                "(transforms computed exactly)"
            )
        
        # Call parent's _fit which handles all the logging and optimization loop
        effective_log_level = "info" if log_level is None else log_level
        return super()._fit(
            X, Y,
            iters_done=iters_done,
            iters=iters,
            inner_steps=0,  # Always 0 for separable
            print_every=print_every,
            callback=callback,
            convergence_tol=convergence_tol,
            convergence_patience=convergence_patience,
            batch_size=batch_size,
            eval_every=eval_every,
            warmup_steps=warmup_steps,
            warmup_until_converged=warmup_until_converged,
            log_every=log_every,
            log_level=effective_log_level,
        )


from typing import Optional
from torch import nn
import torch
from tools.utils import moded_max, moded_min
from models.inf_convolution import InfConvolution
from tools.feedback import logger

# =====================================================================
# FiniteSeparableModel: finitely-convex/concave on product kernels
# =====================================================================

class FiniteSeparableModel(nn.Module):
    """
    Separable finite model for product kernels K(x,y) = sum_d k(x_d, y_d).

    The model discretizes the 1-D Y-domain once (Y = Y_0^d) and keeps one
    intercept per (y_i, dimension) pair. Forward evaluations split across
    dimensions. Sup/inf transforms can optionally run a coarse-to-fine search:

        - coarse_x_factor subsamples the X grid by this stride for the coarse pass.
        - coarse_top_k keeps the top-k coarse candidates per transform call.
        - coarse_window refines within ±(coarse_window * stride) around each seed.

    If coarse_x_factor is None or <= 1, transforms reduce over the full X grid.
    """

    def __init__(
        self,
        kernel,
        num_dims: int,
        radius: float,
        y_accuracy: float = 1e-2,
        x_accuracy: float = 1e-2,
        mode: str = "convex",
        temp: float = 50.0,
        epsilon: float = 1e-4,
        cache_gradients: bool = False,
        coarse_x_factor: Optional[int] = None,
        coarse_top_k: int = 1,
        coarse_window: int = 0,
        ):
        super().__init__()
        assert mode in ("convex", "concave")

        self.kernel_fn = kernel
        self.num_dims = num_dims
        self.radius = radius
        self.y_accuracy = y_accuracy
        self.x_accuracy = x_accuracy
        self.mode = mode
        self.temp = temp
        self.epsilon = epsilon
        self.cache_gradients = cache_gradients

        # Discrete grids shared across dimensions
        ny = int(2 * radius / y_accuracy) + 1
        nx = int(2 * radius / x_accuracy) + 1
        Y_grid = torch.linspace(-radius, radius, ny)
        X_grid = torch.linspace(-radius, radius, nx)
        kernel_tensor = kernel(X_grid.reshape(-1, 1), Y_grid.reshape(1, -1))  # (nx, ny)

        self.register_buffer("Y_grid", Y_grid)
        self.register_buffer("X_grid", X_grid)
        self.register_buffer("kernel_tensor", kernel_tensor)
        if cache_gradients:
            dx_tensor, dy_tensor = self._precompute_kernel_derivatives()
        else:
            dx_tensor = torch.empty(0)
            dy_tensor = torch.empty(0)
        self.register_buffer("kernel_dx", dx_tensor)
        self.register_buffer("kernel_dy", dy_tensor)

        # Intercept parameterization with fixed first row (gauge: b[0, :] = 0)
        self.intercepts_param = FixedFirstIntercept(ny=ny, dim=num_dims)
        # Track last fully refreshed intercepts to make refresh idempotent
        self._last_refreshed_intercepts: Optional[torch.Tensor] = None

        # Optional coarse-to-fine transform search settings
        stride = coarse_x_factor if (coarse_x_factor is not None and coarse_x_factor > 1) else None
        if stride is not None:
            coarse_idx = torch.arange(0, nx, stride, dtype=torch.long)
            if coarse_idx[-1] != nx - 1:
                coarse_idx = torch.cat([coarse_idx, coarse_idx.new_tensor([nx - 1])])
            coarse_top_k = max(1, coarse_top_k)
            coarse_window = max(0, coarse_window)
            window_radius = coarse_window * stride
            offsets = torch.arange(-window_radius, window_radius + 1, dtype=torch.long)
        else:
            coarse_idx = torch.empty(0, dtype=torch.long)
            offsets = torch.empty(0, dtype=torch.long)
            coarse_top_k = 0
            coarse_window = 0
        self.register_buffer("coarse_idx", coarse_idx)
        self.register_buffer("coarse_offsets", offsets)
        self.coarse_stride = stride if stride is not None else 1
        self.coarse_top_k = coarse_top_k
        self.coarse_window = coarse_window
        self.use_coarse_search = coarse_idx.numel() > 0
    @property
    def intercepts(self) -> torch.Tensor:
        """
        Full intercept matrix b of shape (ny, num_dims) with gauge b[0, :] = 0.
        """
        return self.intercepts_param.value

    def refresh_intercepts_via_transform(self) -> int:
        """
        Recompute every intercept column via discrete biconjugation.

        For a separable model with scores K - b:
          - In concave mode, we compute:
                u(x) = min_y [K(x,y) - b(y)]
                b_new(y) = min_x [K(x,y) - u(x)]
          - In convex mode, we compute:
                u(x) = max_y [K(x,y) - b(y)]
                b_new(y) = max_x [K(x,y) - u(x)]

        This is the inf–inf / sup–sup biconjugation used in the main
        algorithm; intercepts are still defined up to a per-column gauge.
        """
        K = self.kernel_tensor.to(device=self.intercepts_param.theta.device, dtype=self.intercepts_param.theta.dtype)
        # If intercepts already match the last refreshed state, do nothing (idempotent).
        current = self.intercepts.detach()
        if self._last_refreshed_intercepts is not None and torch.equal(
            current, self._last_refreshed_intercepts
        ):
            return 0

        changed = 0
        with torch.no_grad():
            for dim in range(self.num_dims):
                column = self.intercepts[:, dim]
                scores = K - column.unsqueeze(0)
                if self.mode == "concave":
                    u_grid = scores.min(dim=1).values
                    refreshed = (K - u_grid.unsqueeze(1)).min(dim=0).values
                else:
                    u_grid = scores.max(dim=1).values
                    refreshed = (K - u_grid.unsqueeze(1)).max(dim=0).values

                changed += (refreshed - column).abs().gt(1e-12).sum().item()
                # Write refreshed column back through parameterization
                self.intercepts_param.set_column_from_raw_(dim, refreshed)

            # Cache refreshed intercepts snapshot for idempotence checks
            self._last_refreshed_intercepts = self.intercepts.detach().clone()

        return changed


    def project(self, T: torch.Tensor) -> torch.Tensor:
        """Clamp tensor entries to [-R+ε, R-ε]."""
        return torch.clamp(T, -self.radius + self.epsilon, self.radius - self.epsilon)

    def _kernel_grad_wrt_x(self, x_scalar: torch.Tensor, y_scalar: torch.Tensor) -> torch.Tensor:
        """Compute ∂k/∂x at (x_scalar, y_scalar) using the continuous kernel."""
        with torch.enable_grad():
            x_temp = x_scalar.detach().clone().requires_grad_(True)
            y_temp = y_scalar.detach().clone().reshape(1, 1)
            x_temp_view = x_temp.reshape(1, 1)
            k_val = self.kernel_fn(x_temp_view, y_temp).squeeze()
            dk_dx = torch.autograd.grad(k_val, x_temp, retain_graph=False, create_graph=False)[0]
        return dk_dx.detach()

    def _kernel_grad_wrt_y(self, x_scalar: torch.Tensor, y_scalar: torch.Tensor) -> torch.Tensor:
        """Compute ∂k/∂y at (x_scalar, y_scalar) using the continuous kernel."""
        with torch.enable_grad():
            y_temp = y_scalar.detach().clone().requires_grad_(True)
            x_temp = x_scalar.detach().clone().reshape(1, 1)
            y_temp_view = y_temp.reshape(1, 1)
            k_val = self.kernel_fn(x_temp, y_temp_view).squeeze()
            dk_dy = torch.autograd.grad(k_val, y_temp, retain_graph=False, create_graph=False)[0]
        return dk_dy.detach()

    def _precompute_kernel_derivatives(self):
        """
        Precompute ∂k/∂x and ∂k/∂y on the (X_grid, Y_grid) lattice
        for use in cached straight-through gradients.
        """
        device = self.X_grid.device
        dtype = self.X_grid.dtype

        with torch.enable_grad():
            dx_list = []
            for y in self.Y_grid:
                x_vals = self.X_grid.clone().detach().requires_grad_(True)
                y_tensor = y.clone().detach().to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
                vals = self.kernel_fn(x_vals.unsqueeze(-1), y_tensor).squeeze()
                grad_x = torch.autograd.grad(
                    vals, x_vals, grad_outputs=torch.ones_like(vals), retain_graph=False, create_graph=False
                )[0]
                dx_list.append(grad_x.detach())
            kernel_dx = torch.stack(dx_list, dim=1)

            dy_list = []
            for x in self.X_grid:
                y_vals = self.Y_grid.clone().detach().requires_grad_(True)
                x_tensor = x.clone().detach().to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
                vals = self.kernel_fn(x_tensor, y_vals.unsqueeze(0)).squeeze()
                grad_y = torch.autograd.grad(
                    vals, y_vals, grad_outputs=torch.ones_like(vals), retain_graph=False, create_graph=False
                )[0]
                dy_list.append(grad_y.detach())
            kernel_dy = torch.stack(dy_list, dim=0)

        return kernel_dx, kernel_dy

    def _cached_dx(self, x_vals: torch.Tensor, y_vals: torch.Tensor) -> torch.Tensor:
        """Interpolate cached ∂k/∂x for each (x, y) pair."""
        if self.kernel_dx.numel() == 0:
            raise RuntimeError("Cached gradients requested but kernel_dx not precomputed.")
        nx = self.X_grid.numel()
        ny = self.Y_grid.numel()

        x_pos = (x_vals + self.radius) / self.x_accuracy
        x_low = torch.floor(x_pos).long().clamp(0, nx - 1)
        x_high = torch.clamp(x_low + 1, 0, nx - 1)
        weight = (x_pos - x_low.float()).clamp(0.0, 1.0)

        y_idx = ((y_vals + self.radius) / self.y_accuracy).round().long().clamp(0, ny - 1)

        grad_low = self.kernel_dx[x_low, y_idx]
        grad_high = self.kernel_dx[x_high, y_idx]
        return grad_low + (grad_high - grad_low) * weight

    def _cached_dy(self, x_idx: torch.Tensor, y_vals: torch.Tensor) -> torch.Tensor:
        """Interpolate cached ∂k/∂y for fixed x-grid indices."""
        if self.kernel_dy.numel() == 0:
            raise RuntimeError("Cached gradients requested but kernel_dy not precomputed.")
        ny = self.Y_grid.numel()

        y_pos = (y_vals + self.radius) / self.y_accuracy
        y_low = torch.floor(y_pos).long().clamp(0, ny - 1)
        y_high = torch.clamp(y_low + 1, 0, ny - 1)
        weight = (y_pos - y_low.float()).clamp(0.0, 1.0)

        grad_low = self.kernel_dy[x_idx, y_low]
        grad_high = self.kernel_dy[x_idx, y_high]
        return grad_low + (grad_high - grad_low) * weight

    def _per_dim_forward(
        self,
        X: torch.Tensor,
        along: int = 0,
        selection_mode: str = "soft",
        snap_to_grid: bool = True,
    ):
        """
        Compute per-dimension contribution:
            convex → max_i k(x_d, y_i) - b_i^d
            concave → min_i k(x_d, y_i) - b_i^d

        Notes
        -----
        - snap_to_grid=True (hard mode) rounds x_d to the nearest X_grid point before
          scoring; False evaluates the continuous kernel directly.
        - In hard mode with gradients, a straight-through update is used:
          if cache_gradients=True and precomputed dx exist, it interpolates them;
          otherwise it backpropagates through per-sample autograd calls. Either way,
          the forward choice stays discrete and gradients are approximate.
        """
        x_vals = X[:, along]
        x_vals_clamped = self.project(x_vals)

        if selection_mode == "hard":
            if snap_to_grid:
                x_indices = self.get_indices_for_x(x_vals_clamped.detach()).long()
                kernel_scores = self.kernel_tensor[x_indices, :]
            else:
                kernel_scores = self.kernel_fn(
                    x_vals_clamped.unsqueeze(-1),
                    self.Y_grid.unsqueeze(0),
                )
        else:
            kernel_scores = self.kernel_fn(
                x_vals_clamped.unsqueeze(-1),
                self.Y_grid.unsqueeze(0),
            )

        b = self.intercepts[:, along].unsqueeze(0)
        scores = kernel_scores - b
        Y_candidates = self.Y_grid.view(1, -1, 1)

        if self.mode == "convex":
            choice, f_x, _ = moded_max(scores, Y_candidates, dim=1, temp=self.temp, mode=selection_mode)
        else:
            choice, f_x, _ = moded_min(scores, Y_candidates, dim=1, temp=self.temp, mode=selection_mode)

        choice = choice.squeeze(-1)

        if selection_mode == "hard" and x_vals.requires_grad:
            # Straight-through gradient: reuse cached derivatives if available, otherwise compute on the fly.
            if self.cache_gradients and self.kernel_dx.numel() > 0:
                approx_grads = self._cached_dx(x_vals_clamped, choice.detach())
            else:
                approx_grads = []
                for xs, ys in zip(x_vals_clamped, choice.detach()):
                    approx_grads.append(self._kernel_grad_wrt_x(xs, ys))
                approx_grads = torch.stack(approx_grads, dim=0)
            f_x = f_x + approx_grads * (x_vals_clamped - x_vals_clamped.detach())

        return choice, f_x

    def forward(
        self,
        X: torch.Tensor,
        selection_mode: str = "soft",
        snap_to_grid: bool = True,
    ):
        """
        Evaluate f(X) and argmax/argmin selections on the separable grid.

        Parameters
        ----------
        X : torch.Tensor
            Input batch of shape (num_samples, num_dims).
        selection_mode : {"soft", "hard", "ste"}
            Soft/Hard/stochastic selection; "hard" uses straight-through gradients.
        snap_to_grid : bool
            If True, x-values are rounded to the X grid before scoring (only relevant in "hard").

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (choice, f_x) with per-dimension choices and summed scores.
        """
        args = []
        values = []
        for dim in range(self.num_dims):
            arg, val = self._per_dim_forward(
                X, along=dim, selection_mode=selection_mode, snap_to_grid=snap_to_grid
            )
            args.append(arg)
            values.append(val)

        f_x = torch.stack(values, dim=1).sum(dim=1)
        choice = torch.stack(args, dim=1)
        return choice, f_x

    def sup_transform(self, Z):
        return self._transform_core(Z, maximize=True)

    def inf_transform(self, Z):
        return self._transform_core(Z, maximize=False)

    def _transform_core(self, Z: torch.Tensor, maximize: bool = True):
        """
        Batch transform across dimensions with optional coarse-to-fine search.

        Parameters
        ----------
        Z : torch.Tensor
            Targets of shape (num_samples, num_dims).
        maximize : bool
            True for sup-transform, False for inf-transform.

        Notes
        -----
        If coarse_x_factor/coarse_top_k/coarse_window were provided at init,
        each per-dimension transform first scores a subsampled X grid, seeds
        the best few candidates, then refines within the configured window;
        otherwise it searches the full grid. Setting coarse_x_factor<=1
        disables the coarse path.
        """
        num_samples, num_dims = Z.shape
        device = Z.device
        dtype = Z.dtype
        X_points = []
        total_values = torch.zeros(num_samples, device=device, dtype=dtype)

        for d in range(num_dims):
            x_opt_d, value_d = self._per_dimension_transform_batch(Z[:, d], along=d, maximize=maximize)
            X_points.append(x_opt_d)
            total_values = total_values + value_d

        X_opt = torch.stack(X_points, dim=1)
        return X_opt.detach(), total_values, True

    def _per_dimension_transform_batch(self, z_vals: torch.Tensor, along: int, maximize: bool):
        """
        Vectorized per-dimension transform for an entire batch of z values.

        Gradients:
            If z_vals.requires_grad, the method adds a straight-through term:
            - with cache_gradients=True and precomputed kernel_dy, it interpolates
              cached ∂k/∂y at the chosen (x*, z) pairs;
            - otherwise it falls back to per-sample autograd on the continuous kernel.
            Forward choices remain discrete; gradients are approximate.
        """
        device = z_vals.device
        dtype = z_vals.dtype
        z_clamped = self.project(z_vals)
        z_idx = self.get_indices_for_y(z_clamped.detach()).long()

        kernel_tensor = self.kernel_tensor
        kernel_vals = kernel_tensor[:, z_idx]  # (nx, batch)

        rows = kernel_tensor
        b = self.intercepts[:, along]
        scores = rows - b.unsqueeze(0)
        if self.mode == "convex":
            f_subset = scores.max(dim=1).values
        else:
            f_subset = scores.min(dim=1).values

        objective_all = kernel_vals - f_subset.unsqueeze(1)
        nx, batch = objective_all.shape

        if not self.use_coarse_search or self.coarse_idx.numel() == 0:
            if maximize:
                best_pos = objective_all.argmax(dim=0)
            else:
                best_pos = objective_all.argmin(dim=0)
            best_values = objective_all[best_pos, torch.arange(batch, device=device)]
            best_idx = best_pos
        else:
            # Fully vectorized coarse-to-fine search over the batch
            coarse_idx = self.coarse_idx
            coarse_obj = objective_all.index_select(0, coarse_idx)  # (n_coarse, batch)
            k = min(self.coarse_top_k, coarse_idx.numel())
            if maximize:
                _, top_pos = torch.topk(coarse_obj, k=k, dim=0, largest=True)
            else:
                _, top_pos = torch.topk(-coarse_obj, k=k, dim=0, largest=True)
            # Seed indices on the full grid, shape (k, batch)
            seed_idx = coarse_idx[top_pos]

            if self.coarse_offsets.numel() == 0:
                refine_idx = seed_idx.view(-1, batch)
            else:
                offsets = self.coarse_offsets.view(1, 1, -1)              # (1,1,n_offsets)
                refine_idx = seed_idx.unsqueeze(-1) + offsets             # (k,batch,n_offsets)
                refine_idx = refine_idx.view(-1, batch)                   # (k*n_offsets, batch)
            refine_idx = refine_idx.clamp(0, nx - 1)

            # Gather candidate objective values for all refined indices
            candidate_obj = objective_all.gather(0, refine_idx)           # (n_candidates, batch)
            if maximize:
                best_rows = candidate_obj.argmax(dim=0)
            else:
                best_rows = candidate_obj.argmin(dim=0)
            best_values = candidate_obj[best_rows, torch.arange(batch, device=device)]
            best_idx = refine_idx[best_rows, torch.arange(batch, device=device)]

        x_opt = self.X_grid[best_idx].to(device=device, dtype=dtype)
        values = best_values

        if z_vals.requires_grad:
            if self.cache_gradients and self.kernel_dy.numel() > 0:
                approx_grad = self._cached_dy(best_idx, z_clamped)
            else:
                grads = []
                for x_opt_val, z_val in zip(x_opt, z_clamped):
                    grads.append(self._kernel_grad_wrt_y(x_opt_val, z_val))
                approx_grad = torch.stack(grads, dim=0)
            values = values + approx_grad * (z_clamped - z_clamped.detach())

        return x_opt, values.to(device=device, dtype=dtype)

    def _objective_on_indices(self, indices, kernel_vals, along):
        """
        Evaluate kernel(x_j, z) - f^d(x_j) for a subset of X-grid indices.

        Args:
            indices: 1-D LongTensor with rows of the X grid to evaluate.
            kernel_vals: precomputed k(x_j, z) for all j.
            along: dimension index.
        """
        rows = self.kernel_tensor.index_select(0, indices)  # (subset, ny)
        b = self.intercepts[:, along]                       # (ny,)
        scores = rows - b.unsqueeze(0)
        if self.mode == "convex":
            f_subset = scores.max(dim=1).values
        else:
            f_subset = scores.min(dim=1).values
        return kernel_vals[indices] - f_subset

    def _build_refine_indices(self, seeds):
        """
        Expand coarse seed indices by the configured window.
        """
        if self.coarse_offsets.numel() == 0:
            return seeds.unique(sorted=True)
        neighbors = seeds.unsqueeze(1) + self.coarse_offsets.unsqueeze(0)
        neighbors = neighbors.reshape(-1)
        neighbors = neighbors.clamp(0, self.X_grid.numel() - 1)
        return neighbors.unique(sorted=True)

    def _coarse_transform(self, kernel_vals, along, maximize):
        """
        Run coarse-to-fine search for transforms:
            1. Evaluate objective on subsampled coarse grid.
            2. Select top-k coarse seeds.
            3. Refine by evaluating all fine-grid points in the window.
        """
        coarse_idx = self.coarse_idx
        coarse_obj = self._objective_on_indices(coarse_idx, kernel_vals, along)
        k = min(self.coarse_top_k, coarse_idx.numel())
        if maximize:
            _, top_pos = torch.topk(coarse_obj, k=k, largest=True)
        else:
            _, top_pos = torch.topk(-coarse_obj, k=k, largest=True)
        seed_idx = coarse_idx[top_pos]
        refine_idx = self._build_refine_indices(seed_idx)
        obj_subset = self._objective_on_indices(refine_idx, kernel_vals, along)
        if maximize:
            best_pos = obj_subset.argmax()
        else:
            best_pos = obj_subset.argmin()
        best_idx = refine_idx[best_pos]
        value = obj_subset[best_pos]
        return best_idx, value

    def get_indices_for_x(self, X):
        idx = ((X + self.radius) / self.x_accuracy).round().long()
        nx = self.X_grid.numel()
        return idx.clamp(0, nx - 1)

    def get_indices_for_y(self, Y):
        idx = ((Y + self.radius) / self.y_accuracy).round().long()
        ny = self.Y_grid.numel()
        return idx.clamp(0, ny - 1)


# =====================================================================
# FiniteModel: unified finitely-convex / finitely-concave representation
# =====================================================================

class FiniteModel(nn.Module):
    """
    Unified finite model representing either finitely convex or finitely concave functions.

    **convex mode**:
        f(x) = max_j [ k(x, y_j) - b_j ]

    **concave mode**:
        f(x) = min_j [ k(x, y_j) - b_j ]

    Supports:
    - soft / hard / ste selection (via moded_max and moded_min)
    - numerical sup/inf transforms
    - full batching support with *global warm-start* storage indexed by sample_idx
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        num_candidates: int,
        num_dims: int,
        kernel,                    # k(x,y)
        mode: str = "convex",
        temp: float = 50.0,
        is_y_parameter: bool = True,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        original_dist_to_bounds: float = 1e-1,
        is_there_default: bool = False,
        default_intercept: float = 0.0,
        sorted_model: bool = False,
    ):
        """
        Initialize a finite convex/concave model over a discrete candidate set.

        Parameters
        ----------
        num_candidates : int
            Number of non-default candidate points (columns in Y_rest).
        num_dims : int
            Dimensionality of each candidate/vector.
        kernel : callable
            Pairwise kernel k(x, y) used to score candidates.
        mode : {"convex", "concave"}
            Whether to take max or min over candidates.
        temp : float
            Softmax temperature used in soft/ste selection.
        is_y_parameter : bool
            If True, Y is trainable; otherwise treated as a buffer.
        y_min, y_max : float | None
            Optional bounds; when provided, Y is rescaled/clamped into [y_min, y_max].
        original_dist_to_bounds : float
            Slack added when only one bound is set to avoid collapsing candidates.
        is_there_default : bool
            Include an extra default option with its own intercept.
        default_intercept : float
            Intercept value for the default option.
        sorted_model : bool
            If True, enforce nondecreasing coordinates per candidate via softplus increments.
        """
        super().__init__()
        assert mode in ("convex", "concave")

        self.num_dims = num_dims
        self.num_candidates = num_candidates
        self.temp = temp
        self.mode = mode
        self.is_y_parameter = is_y_parameter
        self.is_there_default = is_there_default
        self.y_min = y_min
        self.y_max = y_max
        self.original_dist_to_bounds = original_dist_to_bounds
        self.sorted_model = sorted_model
        self.default_intercept = default_intercept

        self.kernel_fn = kernel

        # ---------------------------------------------------------
        # Initialize Y candidates
        # ---------------------------------------------------------
        Y_init = torch.randn(1, num_candidates, num_dims)
        min_val, max_val = Y_init.min(), Y_init.max()

        if sorted_model:
            Y_init = self._prepare_sorted_Y_init(Y_init)
        else:
            Y_init = self._initialize_Y_init(Y_init, min_val, max_val)

        if is_y_parameter:
            self._Y_rest_param = nn.Parameter(Y_init)
        else:
            self.register_buffer("_Y_rest_param", Y_init)

        self.intercept_rest = nn.Parameter(torch.zeros(1, num_candidates))

        # Optional default option
        self.register_buffer("Y0", torch.zeros(1,1,num_dims))
        intercept0_value = Y_init.new_full((1, 1), default_intercept)
        self.register_buffer("intercept0", intercept0_value)

        # ---------------------------------------------------------
        # GLOBAL per-datapoint warm start buffers
        # Will be lazily allocated when sample_idx is first seen.
        # ---------------------------------------------------------
        self._warm_X_global = None
        self._num_global_points = None

    def _initialize_Y_init(self, Y_init: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
        if (self.y_min is not None) and (self.y_max is not None):
            centered = Y_init - Y_init.mean()
            max_abs = centered.abs().max() + 1e-4
            mid = 0.5 * (self.y_min + self.y_max)
            half = 0.5 * (self.y_max - self.y_min)
            return mid + half * centered / max_abs
        elif self.y_min is not None:
            return Y_init - min_val + (self.y_min + self.original_dist_to_bounds)
        elif self.y_max is not None:
            return max_val - Y_init + (self.y_max - self.original_dist_to_bounds)
        return Y_init

    def _prepare_sorted_Y_init(self, Y_init: torch.Tensor) -> torch.Tensor:
        """Apply one-time rescale for sorted models so later updates are not clamped."""
        # sorted path builds Y via softplus increments + cumsum; rescale once here
        increments = F.softplus(Y_init)
        sorted_Y = torch.cumsum(increments, dim=-1)
        if (self.y_min is not None) or (self.y_max is not None):
            sorted_Y = self._rescale_to_range(sorted_Y)
            diffs = torch.cat(
                [sorted_Y[..., :1], sorted_Y[..., 1:] - sorted_Y[..., :-1]],
                dim=-1,
            )
            safe_diffs = diffs.clamp_min(1e-8)
            Y_init = torch.log(torch.expm1(safe_diffs))
        return Y_init


    # ============================================================
    # Helpers: Y accessors / full intercepts
    # ============================================================

    @property
    def Y_rest(self):
        base = self._Y_rest_param
        if not self.sorted_model:
            # Unsorted model: just return the Y
            return base

        increments = F.softplus(base)
        sorted_Y = torch.cumsum(increments, dim=-1)
        return sorted_Y

    def _rescale_to_range(self, Y: torch.Tensor) -> torch.Tensor:
        if self.y_min is None and self.y_max is None:
            return Y

        if self.y_min is not None and self.y_max is not None:
            raw_min = Y.amin(dim=1, keepdim=True)
            raw_max = Y.amax(dim=1, keepdim=True)
            span = raw_max - raw_min
            if (span <= 1e-6).all():
                return self._rescale_bounds_clamp(Y)
            target_span = self.y_max - self.y_min
            return (Y - raw_min) / span * target_span + self.y_min

        if self.y_min is not None:
            raw_min = Y.amin(dim=1, keepdim=True)
            return (Y + (self.y_min - raw_min)).clamp(min=self.y_min)

        raw_max = Y.amax(dim=1, keepdim=True)
        return (Y + (self.y_max - raw_max)).clamp(max=self.y_max)

    def _rescale_bounds_clamp(self, Y: torch.Tensor) -> torch.Tensor:
        lower = self.y_min if self.y_min is not None else float("-inf")
        upper = self.y_max if self.y_max is not None else float("inf")
        return torch.clamp(Y, min=lower, max=upper)

    @property
    def Y_rest_raw(self):
        return self._Y_rest_param

    def full_Y(self):
        if self.is_there_default:
            return torch.cat([self.Y0, self.Y_rest], dim=1)
        return self.Y_rest

    def full_intercept(self):
        if self.is_there_default:
            return torch.cat([self.intercept0, self.intercept_rest], dim=1)
        return self.intercept_rest


    # ============================================================
    # Forward: MAX or MIN over candidates
    # ============================================================

    def forward(self, X: torch.Tensor, selection_mode: str = "soft", already_sorted: bool = False):
        """
        Compute f(x) for x ∈ R^{num_samples × num_dims} using either
        max_j or min_j of the kernel scores.
        """
        if self.sorted_model and not already_sorted:
            X = torch.sort(X, dim=-1).values
        Y = self.full_Y()             # (1, num_candidates, num_dims)
        b = self.full_intercept()     # (1, num_candidates)
        batch_size = getattr(self, "kernel_batch_size", None)
        def _compute_chunk(X_chunk):
            scores_chunk = self.kernel_fn(X_chunk[:, None, :], Y) - b
            if self.mode == "convex":
                return moded_max(scores_chunk, Y, dim=1, temp=self.temp, mode=selection_mode)
            return moded_min(scores_chunk, Y, dim=1, temp=self.temp, mode=selection_mode)

        if batch_size is None or batch_size <= 0 or X.shape[0] <= batch_size:
            choice, f_x, aux = _compute_chunk(X)
        else:
            choices = []
            fxs = []
            last_aux = {}
            for start in range(0, X.shape[0], batch_size):
                end = start + batch_size
                X_chunk = X[start:end]
                choice_chunk, f_x_chunk, aux_chunk = _compute_chunk(X_chunk)
                choices.append(choice_chunk)
                fxs.append(f_x_chunk)
                last_aux = aux_chunk
            choice = torch.cat(choices, dim=0)
            f_x = torch.cat(fxs, dim=0)
            aux = last_aux

        # Diagnostics
        self._last_weights = aux.get("weights")
        self._last_idx = aux.get("idx")
        self._last_mean_max_weight = aux.get("mean_max")
        self._last_eff_temp = aux.get("eff_temp")

        return choice, f_x


    # ============================================================
    # PUBLIC sup/inf transforms (batch-aware sample_idx added)
    # ============================================================

    def sup_transform(self, Z, sample_idx=None, **kw):
        """ 
        Compute sup_x [k(x,z) - f(x)] for each z.
        
        Returns:
            tuple: (X_opt, values, converged) where
                - X_opt: Optimized positions (detached)
                - values: Transform values (with gradients)
                - converged: Boolean indicating if optimization converged
        """
        return self._transform_core(Z, sample_idx, maximize=True, **kw)

    def inf_transform(self, Z, sample_idx=None, **kw):
        """ 
        Compute inf_x [k(x,z) - f(x)] for each z.
        
        Returns:
            tuple: (X_opt, values, converged) where
                - X_opt: Optimized positions (detached)
                - values: Transform values (with gradients)
                - converged: Boolean indicating if optimization converged
        """
        return self._transform_core(Z, sample_idx, maximize=False, **kw)


    # ============================================================
    # CORE routine with per-sample warm start via sample_idx
    # ============================================================

    def _transform_core(
        self,
        Z: torch.Tensor,
        sample_idx: Optional[torch.Tensor],
        maximize: bool,
        steps: int = 50,
        lr: float = 1e-3,
        optimizer: str = "lbfgs",
        lam: float = 1e-3,
        tol: float = 1e-3,
        patience: int = 5,
    ):
        """
        Main routine for sup_x or inf_x:

            maximize=True  → sup_x (negated inf)
            maximize=False → inf_x

        Supports random mini-batching through sample_idx,
        enabling per-datapoint warm-start reuse.
        
        Uses InfConvolution for implicit differentiation.
        
        Returns:
            tuple: (X_opt, values, converged) where
                - X_opt: Optimized positions (detached)
                - values: Transform values (with gradients)
                - converged: Boolean indicating if optimization converged

        Convergence Detection:
            - LBFGS: Always considered converged (uses internal line search)
            - Adam/GD: Converged if |loss[i] - loss[i-1]| < tol for any step i
            - If max steps reached without meeting tolerance, converged=False

        Warm starts:
            If sample_idx is provided, per-sample warm-starts are stored and
            reused across calls (and grown dynamically as new indices appear);
            otherwise Z is used as the initialization.

        Optimizer controls:
            `optimizer` selects between "lbfgs" and first-order updates; `lam`
            is the implicit-diff regularizer; `patience` gates early stopping
            for first-order modes. The returned `converged` flag reflects these
            criteria only.
        """
        num_samples, num_dims = Z.shape
        device = Z.device

        # -------------------------------------------------------------
        # Allocate GLOBAL warm-start storage if first call
        # -------------------------------------------------------------
        if sample_idx is not None:
            max_idx = int(sample_idx.max())

            if self._warm_X_global is None:
                # First time: allocate storage for all possible Y indices
                self._num_global_points = max_idx + 1
                self._warm_X_global = torch.zeros(
                    self._num_global_points, num_dims, device=device
                )
            elif max_idx + 1 > self._num_global_points:
                # Expand storage if new larger index appears
                new_size = max_idx + 1
                new_tensor = torch.zeros(new_size, num_dims, device=device)
                new_tensor[:self._num_global_points] = self._warm_X_global
                self._warm_X_global = new_tensor
                self._num_global_points = new_size

            # Extract warm starts for this batch
            X_init = self._warm_X_global[sample_idx].clone()

        else:
            # Fallback: no sample_idx → use Z as initialization
            X_init = Z.clone()

        # -------------------------------------------------------------
        # Create a wrapper module for f(x) to work with InfConvolution
        # InfConvolution expects f_net(x) to return scalar value
        # and x will be a 1D tensor of shape (num_dims,)
        # -------------------------------------------------------------
        class FWrapper(nn.Module):
            def __init__(self, finite_model, negate=False):
                super().__init__()
                self.finite_model = finite_model
                self.negate = negate
            
            def forward(self, x):
                # x is (num_dims,) - need to add batch dimension
                # FiniteModel.forward expects (num_samples, num_dims)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                # self.finite_model.forward returns (choice, f_x)
                _, f_x = self.finite_model.forward(x, selection_mode="soft")
                # f_x is (num_samples,) or scalar - return scalar
                result = f_x.squeeze()
                return -result if self.negate else result
        
        # -------------------------------------------------------------
        # Define kernel function that handles maximize flag and dimensions
        # InfConvolution expects K(x, y) where x and y are 1D tensors
        # 
        # For maximize=True (sup):
        #   sup_x [k(x,z) - f(x)] = -inf_x [-(k(x,z) - f(x))]
        #                         = -inf_x [f(x) - k(x,z)]
        #   InfConvolution computes: inf_x [K(x,z) - F(x)]
        #   So we need: K(x,z) = -k(x,z) and F(x) = -f(x)
        #   Then: inf_x [-k(x,z) - (-f(x))] = inf_x [f(x) - k(x,z)]
        #   And negate result: -inf_x [f(x) - k(x,z)] = sup_x [k(x,z) - f(x)]
        # -------------------------------------------------------------
        if maximize:
            f_wrapper = FWrapper(self, negate=True)
            
            def K_wrapper(x, z):
                # Ensure x and z are 2D for kernel_fn
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                if z.dim() == 1:
                    z = z.unsqueeze(0)
                result = -self.kernel_fn(x, z)
                return result.squeeze()
        else:
            f_wrapper = FWrapper(self, negate=False)
            
            def K_wrapper(x, z):
                # Ensure x and z are 2D for kernel_fn
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                if z.dim() == 1:
                    z = z.unsqueeze(0)
                result = self.kernel_fn(x, z)
                return result.squeeze()

        # -------------------------------------------------------------
        # Apply InfConvolution for each sample in Z
        # InfConvolution now returns (g_value, converged, x_star) so we
        # don't need to re-solve the optimization
        # -------------------------------------------------------------
        values_list = []
        converged_list = []
        X_opt_list = []

        for i in range(num_samples):
            z_i = Z[i]  # 1D tensor of shape (num_dims,)
            x_init_i = X_init[i]  # 1D tensor of shape (num_dims,)
            
            # InfConvolution.apply returns (g_value, converged, x_star)
            # This computes the value with proper gradients via implicit differentiation
            # and returns the optimal x for warm-starting
            g_i, conv_i, x_star_i = InfConvolution.apply(
                z_i,
                f_wrapper,
                K_wrapper,
                x_init_i,
                steps,
                lr,
                optimizer,
                lam,
                tol,
                patience,
                *list(self.parameters())
            )
            
            values_list.append(g_i)
            converged_list.append(conv_i)
            X_opt_list.append(x_star_i)
        
        values = torch.stack(values_list)
        converged = all(converged_list)
        X_opt = torch.stack(X_opt_list)
        
        # Negate values if maximize
        if maximize:
            values = -values

        # -------------------------------------------------------------
        # Save warm starts
        # -------------------------------------------------------------
        with torch.no_grad():
            if sample_idx is not None:
                self._warm_X_global[sample_idx] = X_opt.detach()

        return X_opt.detach(), values, converged
