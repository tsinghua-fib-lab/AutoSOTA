"""Separable optimal transport solver with custom optimizer heuristics."""

import math

import torch
from torch import nn
from optimal_transport.ot import OT
from models import FiniteSeparableModel
from tools.feedback import logger


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
        """Reset gradients for both warm-up and steady optimizers."""
        self._adam.zero_grad()
        self._sgd.zero_grad()

    def state_dict(self):
        """Serialize optimizer state for checkpointing."""
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
        """Restore optimizer state and resume appropriate learning phase."""
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
        """Apply the same learning rate to both optimizers."""
        for group in self._adam.param_groups:
            group["lr"] = lr
        for group in self._sgd.param_groups:
            group["lr"] = lr

    def step(self, closure=None, grad_norm: float | None = None):
        """Perform a parameter update, switching optimizers when criteria met using grad norm."""
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
        Factory method for creating FCOTSeparable with specified grid resolution.
        
        Parameters
        ----------
        dim : int
            Input/output dimension
        radius : float
            Domain radius (X, Y ⊆ [-R, R]^d)
        n_params : int
            Total number of intercept parameters (must equal dim × |Y_0|)
        x_accuracy : float
            Spacing for the X grid (controls X discretization used in transforms)
        kernel_1d : callable
            1D kernel c_d(x, y) where x, y are scalars (or 1D tensors)
        inverse_kx : callable
            Inverse gradient: (x, p) ↦ y solving ∇_x c(x,y) = p
        outer_lr : float
            Learning rate for outer optimization (Adam on intercepts)
        betas : tuple
            Beta parameters for Adam optimizer
        device : str
            Device to place model on
        temp : float
            Temperature for soft max/min in forward pass
        epsilon : float
            Boundary margin for clamping inputs to valid domain
        cache_gradients : bool
            If True, precompute ∂k/∂x and ∂k/∂y on the grid to speed up hard selections
        
        Returns
        -------
        FCOTSeparable
            Configured solver instance
            
        Notes
        -----
        Parameter count:
            n_params = dim × |Y_0|
        
        Each dimension has n_params / dim intercept parameters (y positions fixed on Y-grid).
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
        Initialize the separable FCOT solver with finite-grid model and optimizer.

        Parameters
        ----------
        input_dim : int
            Dimensionality of each sample in X/Y.
        model : torch.nn.Module
            FiniteSeparableModel instance representing u/b potentials.
        inverse_kx : callable
            Function mapping (x, ∇u(x)) ↦ y for the separable cost.
        outer_lr : float
            Learning rate for the outer Adam optimizer.
        lr : float | None
            Deprecated alias for `outer_lr` kept for backward compatibility.
        betas : tuple
            Beta parameters for the outer optimizer.
        device : str
            Device string for torch tensors.
        warmup_lr : float | None
            Deprecated warm-up learning rate (ignored with warning).
        warmup_grad_threshold etc. :
            Legacy arguments ignored for deterministic separable transforms.
        reactivate_every : int | None
            Interval (in outer steps) to attempt reactivating inactive intercepts.
        reactivate_eps : float
            Margin used when reactivating intercepts.
        full_refresh_every : int | None
            Interval to perform full intercept refresh and jitter.
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
        Compute dual objective D = E_x[u(X)] + E_y[u^c(Y)] on the given batch.
        No inner optimization is required for the separable model.
        """
        _, u_vals = self.model.forward(x_batch, selection_mode="soft")
        u_mean = u_vals.mean()

        _, uc_vals, converged = self.model.inf_transform(y_batch)
        uc_mean = uc_vals.mean()

        D = u_mean + uc_mean
        return D, u_mean, uc_mean, converged

    def step(self, x_batch, y_batch, idx_y=None, active_inner_steps=None):
        """
        Perform one gradient step on the separable dual objective and update intercepts.

        Returns a dict summarizing current dual objective, mean potentials,
        convergence status, gradient norm, and intercept activity statistics.
        """
        self._maybe_update_temperature()
        self.optimizer.zero_grad()
        D, u_mean, uc_mean, converged = self._dual_objective(
            x_batch, y_batch, idx_y, active_inner_steps
        )
        # Scale by negative constant to convert maximization into stable gradient descent.
        (-10 * D).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

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
        # Gradient activity based on underlying intercept parameters
        intercepts_param = getattr(self.model, "intercepts_param", None)
        theta_grad = None
        if intercepts_param is not None and hasattr(intercepts_param, "theta"):
            theta_grad = intercepts_param.theta.grad
        if theta_grad is not None:
            # Map parameter gradient to full intercept shape by padding first row with zeros
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
        """Refresh intercepts whose gradients have gone to zero to avoid dead regions."""
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

        # Build full gradient matrix (pad first row with zeros to match intercept shape)
        intercepts = self.model.intercepts
        full_grad = torch.zeros_like(intercepts, device=theta_grad.device, dtype=theta_grad.dtype)
        full_grad[1:, :] = theta_grad.detach()

        # Identify intercepts whose gradient magnitude has dropped below threshold
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
                # Increase rows that are inactive by comparing to noisy target
                updated_column = torch.where(mask_dim, torch.maximum(column, target), column)
                if torch.any(updated_column != column):
                    # Write back through parameterization if available
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
        """Perform a complete intercept refresh/jitter routine at regular intervals."""
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
        Save model checkpoint for caching/resume consistency with other OT subclasses.
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
        Load checkpoint saved via `save` and return the number of completed iterations.
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
        Computes the finite-separable Monge map using analytical transforms.

        Parameters
        ----------
        X : torch.Tensor
            Source samples to transport.
        selection_mode : str
            How to select grid points ("soft" vs "hard").
        snap_to_grid : bool
            Whether to snap to the grid during forward pass.

        Returns
        -------
        torch.Tensor
            Transported target locations.
        """
        X = X.to(self.device)
        X.requires_grad_(True)
        _, u_X = self.model.forward(
            X, selection_mode=selection_mode, snap_to_grid=snap_to_grid
        )
        grad_u = torch.autograd.grad(u_X.sum(), X, create_graph=False)[0]
        return self.inverse_kx(X.detach(), grad_u.detach())

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
        Fit the model by maximizing the dual objective.
        
        Note: inner_steps is ignored for separable model since transforms
        are computed exactly on grids (no numerical optimization).
        
        Returns
        -------
        dict
            Logs collected by the parent `_fit`.
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
