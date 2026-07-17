"""Flow matching with optimal transport potentials (OTP-FM)."""

import logging
from collections import OrderedDict

import torch
from torch import Tensor, nn

from .networks import FlowNetMLP, create_ema_flownet
from .potentials import Potential
from .solvers import get_solver

logger = logging.getLogger(__name__)


class OTPFM(nn.Module):
    """
    Trains and samples a given velocity predictor network v(x, t₁, t₂) with the OTP-FM algorithm.

    ``forward_with_loss()`` implements:

    - Computing the base conditional velocity targets (as in vanilla conditional flow matching)
    - Computing the OTP corrections to the targets, scaled according to the OTP-FM learning curriculum
    - Computing the consistency training loss against the targets (improved MeanFlow by default)

    ``sample()`` samples trajectories using the learnt consistency model.

    Args:
        d (int): Dimension of the data.
        tks (list[float]): List of intermediate time points where marginal constraints are enforced.
        potentials (OrderedDict[float, Potential]): Dictionary mapping times to their corresponding potentials.
        flownet (nn.Module | None): Custom velocity network. If provided, flownet_args is ignored.
            Must implement forward(x, t, dt) -> v and optionally get_log_var(t, dt) -> log_var if lossfn is "weighted".
        flownet_args (dict): Arguments for the FlowNetMLP (ignored if flownet is provided).
        consistency_loss (str): Consistency loss type. One of {"meanflow", "imf", "lsd"}. Default: "meanflow".
        lossfn (str): Loss function type. One of {"mse", "adaptive", "weighted"}. Default: "adaptive".
        ema_ot (bool): Whether to use EMA model for X_tk predictions. Default: True.
        time_sampler (str): Time sampling strategy. Default: "uniform_scaled_marginal".
        diag_prob (float): Probability of t1 = t2 (diagonal samples). Default: 0.75.
        euler_steps (int): Number of Euler steps for X_tk predictions. Default: 2.
        ema_decay (float): EMA decay rate (0 to disable). Default: 0.99.
        jvp_clamp (float): Clamp value for JVP to prevent loss explosion. Default: 100.
        picard_steps (int): Number of fixed-point iterations. Default: 5.
        solver_type (str | None): Fixed-point solver type. Auto-selected if None.
        solver_kwargs (dict | None): Additional arguments for the solver.
        adaptive_exp (float): Exponent for adaptive loss. Default: 1.0.
        x_pred (bool): Does FlowNet do v-prediction or x-prediction (as in JiT). Default: False (v-pred).
    """

    X_PRED_DEFAULT_KWARGS = {
        "lhopital": False,
        "denom_clamp": 0.05,
        "x_sampling": False,
    }

    def __init__(
        self,
        d: int,
        tks: list[float],
        potentials: OrderedDict[float, Potential],
        flownet: nn.Module | None = None,
        flownet_args: dict = None,
        consistency_loss: str = "meanflow",
        lossfn: str = "adaptive",
        ema_ot: bool = True,
        time_sampler: str = "uniform_scaled_marginal",
        diag_prob: float = 0.75,
        euler_steps: int = 2,
        ema_decay: float = 0.99,
        jvp_clamp: float = 100,
        picard_steps: int = 5,
        solver_type: str = None,
        solver_kwargs: dict | None = None,
        adaptive_exp: float = 1.0,
        x_pred: bool = False,
        x_pred_kwargs: dict = None,
    ):
        super().__init__()
        assert not (ema_ot and ema_decay == 0), "EMA decay must be non-zero if EMA OT is enabled"

        self.d = d
        self.tks = torch.tensor([0.0] + list(tks) + [1.0])
        self.potentials = potentials
        self.consistency_loss = consistency_loss
        self.lossfn = lossfn
        self.time_sampler = time_sampler
        self.diag_prob = diag_prob
        self.euler_steps = euler_steps
        self.ema_decay = ema_decay
        self.ema_ot = ema_ot
        self.jvp_clamp = jvp_clamp
        self.adaptive_exp = adaptive_exp
        self.x_pred = x_pred

        # Solver (default: DirectSolver for W2 potentials, AndersonSafe for non-W2 potentials)
        solver_kwargs = solver_kwargs or {}
        self.solver_type = solver_type
        if solver_type is None:
            potential_type = list(self.potentials.values())[0].__class__.__name__
            if potential_type in ["W2InfPotential", "W2Potential"]:
                solver_type = "direct"
            else:
                solver_type = "anderson_safe"
        self.solver = get_solver(solver_type, picard_steps, **solver_kwargs)
        logger.debug(f"Solver: {self.solver}")

        # Create or use provided velocity network
        self.flownet_args = flownet_args if flownet_args is not None else {}

        if flownet is not None:
            # Use provided custom velocity network
            self.flownet = flownet
            logger.debug("Using custom velocity network")
            logger.debug(self.flownet)
        else:
            # Create FlowNetMLP velocity network
            self.flownet = FlowNetMLP(
                d,
                **self.flownet_args,
                predict_log_var=(lossfn == "weighted"),
            )
            logger.debug("Created velocity network: FlowNetMLP")
            logger.debug(self.flownet)

        # EMA model for stable evaluation - add_module to ensure proper registration
        if ema_decay > 0:
            self.add_module("flownet_ema", create_ema_flownet(self.flownet))
        else:
            self.flownet_ema = None

        # X-prediction kwargs
        self.x_pred_kwargs = {**self.X_PRED_DEFAULT_KWARGS, **(x_pred_kwargs or {})}

        self._process_potentials()

    def _process_potentials(self):
        """
        Pre-compute useful quantities and process widths of potentials if set to "auto".
        """
        self.dtks = self.tks[1:] - self.tks[:-1]
        self._precompute_x_time_deps()
        for i, potential in enumerate(self.potentials.values()):
            if potential.width == "auto":
                ave_dtk = (self.dtks[i] + self.dtks[i + 1]) / 2
                potential.set_width(ave_dtk / 2)

    def _precompute_x_time_deps(self):
        """
        Precompute the time-dependent factors x_time_dep[i, j] for all pairs of potentials, used when calculating the OTP corrections.
        x_time_dep[i, j] = factor for potential j's contribution to position at t_i

        If using the DirectSolver, also precomputes A[i,j] = x_time_dep[i,j] * strength[j] and (I - A)^{-1}
        (Eq. 20 of :cite:t:`kansal2026otpfm`).
        """
        potentials_list = list(self.potentials.values())
        tks_list = list(self.potentials.keys())
        K = len(potentials_list)

        if K == 0:
            self._x_time_deps_base = None
            self._I_minus_A_inv = None
            self._A_matrix = None
            return

        # Compute (K, K) matrix of time factors
        x_time_deps = torch.zeros(K, K, dtype=torch.float32)
        for i, tk_i in enumerate(tks_list):
            tk_tensor = torch.tensor([[tk_i]], dtype=torch.float32)  # (1, 1)
            for j, potential_j in enumerate(potentials_list):
                x_time_deps[i, j] = self.x_time_dep(potential_j, tk_tensor).item()

        # Store as (K, K, 1, 1) for broadcasting with (bs, d)
        self._x_time_deps_base = x_time_deps.view(K, K, 1, 1)

        # Precompute A matrix and (I - A)^{-1} for DirectSolver
        if self.solver_type == "direct":
            strengths = torch.tensor([p.strength for p in potentials_list], dtype=torch.float32)
            A = x_time_deps * strengths.unsqueeze(0)  # (K, K)
            I_minus_A = torch.eye(K, dtype=torch.float32) - A
            self._A_matrix = A
            self._I_minus_A_inv = torch.linalg.inv(I_minus_A)
            logger.debug("Precomputed A matrix and (I - A)^{-1} for DirectSolver")
        else:
            self._A_matrix = None
            self._I_minus_A_inv = None

    @torch.no_grad()
    def update_ema(self):
        """Update EMA model weights, should be called after each optimizer step."""
        if self.flownet_ema is None:
            return

        # Update EMA parameters
        for ema_param, param in zip(self.flownet_ema.parameters(), self.flownet.parameters()):
            ema_param.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def v_func(self, x, t1, t2, net=None):
        """Mean predicted velocity.
        Implements either direct velocity prediction or x-prediction as in
        :cite:t:`li2026basics,lu2026onestep`.
        """
        dt = t2 - t1
        if net is None:
            net = self.flownet

        if self.x_pred:
            x_out = net(x, t1, dt)
            # clipped as in JiT paper
            v = (x_out - x) / torch.clamp(dt.view(-1, 1), min=self.x_pred_kwargs["denom_clamp"])
        else:
            v = net(x, t1, dt)

        return v

    def v_func_ema(self, x, t1, t2):
        """Get velocity using EMA model (for stable evaluation)"""
        return self.v_func(x, t1, t2, net=self.flownet_ema)

    def x_time_dep(self, potential: Potential, t: float | torch.Tensor):
        """Get the time-dependent factor for the potential"""
        lambdafn = potential.lambda_fn
        x_time_dep = lambdafn.double_integrated(t) - lambdafn.double_integrated_at_1() * t
        return x_time_dep

    def v_time_dep(self, potential: Potential, t: float | torch.Tensor):
        """Get the time-dependent factor for the potential"""
        lambdafn = potential.lambda_fn
        v_time_dep = lambdafn.integrated(t) - lambdafn.double_integrated_at_1()
        return v_time_dep

    def x_t(self, x0: Tensor, t: Tensor, ema: bool = True):
        """
        Get the predicted position x(t) using the learnt mean velocity v(x0, 0, t)
        from starting positions x0 in 1 Euler step: x(t) = x0 + t * v(x0, 0, t)
        """
        v_func = self.v_func_ema if ema else self.v_func
        curr_t = torch.zeros_like(t)
        step_size = t / self.euler_steps
        for _ in range(self.euler_steps):
            next_t = curr_t + step_size
            x0 = x0 + step_size * v_func(x0, curr_t, next_t)
            curr_t = next_t
        return x0

    def compute_X_tks_learnt(
        self,
        x0: Tensor,
        tks_list: list[float],
        ema: bool = True,
    ) -> Tensor:
        """
        Compute learned positions progressively at each t_k using the MeanFlow / consistency velocity for fewer evaluations.

        Args:
            x0: Source samples, shape (bs, d)
            tks_list: List of K time values
            ema: Whether to use EMA model for velocity

        Returns:
            X_tks_learnt: (K, bs, d) learned positions at each t_k
        """
        device = x0.device
        bs = x0.shape[0]
        K = len(tks_list)

        if K == 0:
            return torch.empty(0, bs, self.d, device=device)

        v_func = self.v_func_ema if ema else self.v_func
        X_tks_learnt = []
        x_curr = x0
        prev_t = torch.zeros(bs, 1, device=device)

        for tk_val in tks_list:
            tk_tensor = torch.ones(bs, 1, device=device) * tk_val
            # Jump from position at t_{k-1} to current t_k using MeanFlow velocity
            dt = tk_tensor - prev_t
            step_size = dt / self.euler_steps
            for _ in range(self.euler_steps):
                next_t = prev_t + step_size
                x_curr = x_curr + step_size * v_func(x_curr, prev_t, next_t)
                prev_t = next_t

            X_tks_learnt.append(x_curr.clone())
            prev_t = tk_tensor

        return torch.stack(X_tks_learnt)  # (K, bs, d)

    def solve_X_tks(
        self,
        x0: Tensor,
        u_base: Tensor,
        xms: Tensor,  # (K, bs, d)
        potentials_list: list[Potential],
        tks_list: list[float],
        otp_alpha: float,
        debug: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Solve the self-consistent equations for all X_tk simultaneously using a fixed-point solver
        (Eq. 18 of :cite:t:`kansal2026otpfm`).

        The coupled system is (for K potentials at times t_1, ..., t_K):
        X(t_k) = X_base(t_k) + Σ_j x_time_dep_j(t_k) * dV_j(xm_j, X(t_j))

        Args:
            x0: Source samples, shape (bs, d)
            u_base: Base velocity, shape (bs, d)
            xms: Stacked intermediate marginal samples, shape (K, bs, d)
            potentials_list: List of K Potential objects
            tks_list: List of K time values
            otp_alpha: Progressive loss weight (only for initialization)
            debug: Whether to print debug info

        Returns:
            X_tks: (K, bs, d) position tensors at each t_k (for debugging)
            dV_tks: (K, bs, d) potential gradient tensors (AKA force)
            residual_norms: (K, bs) per-sample residual norms (for debugging)
        """
        device = x0.device
        bs = x0.shape[0]
        K = len(potentials_list)

        if K == 0:
            return (
                torch.empty(0, bs, self.d, device=device),
                torch.empty(0, bs, self.d, device=device),
                torch.empty(0, bs, device=device),
            )

        # x_time_deps: (K, K, 1, 1) -> broadcast to (K, K, bs, 1)
        x_time_deps = self._x_time_deps_base.to(device=device, dtype=x0.dtype)

        # Compute base positions at each t_k: X_base_k = x0 + u_base * t_k
        # tks: (K,) -> (K, 1, 1) for broadcasting
        tks_tensor = torch.tensor(tks_list, device=device, dtype=x0.dtype).view(K, 1, 1)
        X_bases = x0.unsqueeze(0) + u_base.unsqueeze(0) * tks_tensor  # (K, bs, d)

        if self.solver_type == "direct":
            # Direct solver does not need X_tks
            X_inits = X_bases
        else:
            # Initialize X_tks from learned positions (no gradient through this) and use weighted average with base
            with torch.no_grad():
                X_tks_learnt = self.compute_X_tks_learnt(x0, tks_list, ema=self.ema_ot)
            X_inits = X_bases * (1 - otp_alpha) + otp_alpha * X_tks_learnt

        X_tks, dV_tks, residual_norms = self.solver(
            X_inits=X_inits,
            X_bases=X_bases,
            x_time_deps=x_time_deps,
            potentials=potentials_list,
            xms=xms,
            debug=debug,
            A_matrix=(
                self._A_matrix.to(device=device, dtype=x0.dtype)
                if self._A_matrix is not None
                else None
            ),
            I_minus_A_inv=(
                self._I_minus_A_inv.to(device=device, dtype=x0.dtype)
                if self._I_minus_A_inv is not None
                else None
            ),
        )

        return X_tks, dV_tks, residual_norms

    def sample_time(self, bs: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample times t1 and t2 for MeanFlow training, uniformly from [0, 1].

        Args:
            bs (int): batch size
            device (torch.device): device

        Returns:
            t1 (torch.Tensor, shape (bs)): first timepoint
            t2 (torch.Tensor, shape (bs)): second timepoint
        """
        times = torch.rand((bs, 2), device=device)

        if self.time_sampler == "uniform_min":
            # simply choose t1 and t2 as min and max
            # (t1 therefore is biased towards 0)
            t1 = times.min(dim=1).values
            t2 = times.max(dim=1).values
        elif self.time_sampler.startswith("uniform_scaled"):
            # t2 is after t1 by construction
            # this means t1 remains a uniform distribution
            t1 = times[:, 0]
            t2 = times[:, 1]

            if self.time_sampler.startswith("uniform_scaled_marginal"):
                interval = torch.randint(len(self.potentials) + 1, (bs,), device=device)
                tks_device = self.tks.to(device)
                prev_tk = tks_device[interval]
                next_tk = tks_device[interval + 1]
                t1 = prev_tk + (1 - t1) * (next_tk - prev_tk)

            t2 = t1 + (1 - t1) * t2

            if self.time_sampler.startswith("uniform_scaled_marginal_t1t2"):
                # t2 is in same marginal interval with a 75% probability
                t2_interval = t1 + (next_tk - t1) * t2
                interval_mask = torch.rand(bs, device=device) < 0.75
                t2 = torch.where(interval_mask, t2_interval, t2)

        # Sometimes t1 = t2
        if self.diag_prob > 0:
            mask = torch.rand(bs, device=device) < self.diag_prob
            t2 = torch.where(mask, t1, t2)

        return t1.view(-1, 1), t2.view(-1, 1)

    def forward_with_loss(
        self,
        xs: Tensor,
        otp_alpha: float,
        do_otp: bool = True,
        debug: bool = False,
    ):
        """
        Forward pass with loss computation.

        Args:
            xs (Tensor, shape ``(batch_size, num_marginals, *dim)``): samples from each marginal
                num_marginals = K + 2 where K is the number of intermediate marginals
                xs[:, 0] = source samples (µ_0)
                xs[:, 1:K+1] = intermediate marginal samples (µ_{t_k} for k=1,...,K)
                xs[:, -1] = target samples (µ_1)
            otp_alpha (float): weighting of the learnt velocities and positions used relative to base.
                We start training with 0 and gradually increase to 1.
            do_otp (bool): Whether to perform OTP-FM or simply flow between the endpoints. Default is True.
        """
        device = xs.device
        bs = xs.shape[0]
        x0, x1 = xs[:, 0], xs[:, -1]
        num_potentials = len(self.potentials)
        do_otp = do_otp and otp_alpha > 1e-3

        # sample times and compute base conditional path (same as regular FM / MeanFlow)
        t1, t2 = self.sample_time(bs, device)
        u_base = x1 - x0
        X_base_t1 = x0 + u_base * t1

        if do_otp:
            # Compute OTP correctios to the conditional path
            potentials_list = list(self.potentials.values())
            tks_list = list(self.potentials.keys())
            xms = torch.stack([xs[:, k + 1] for k in range(num_potentials)])  # (K, bs, d)

            # Get the force (∂V/∂t) of each potential by solving for fixed points (K, bs, d)
            dV_tks = self.solve_X_tks(
                x0, u_base, xms, potentials_list, tks_list, otp_alpha, debug=debug
            )[1]

            # x_time_dep_t1[k] = x_time_dep for potential k at time t1 (K, bs, 1)
            x_time_dep_t1 = torch.stack(
                [self.x_time_dep(potential, t1) for potential in potentials_list]
            )

            # Accumulate corrections: sum over k of dV_tks[k] * x_time_dep_t1[k]
            X_t1_corr_total = (dV_tks * x_time_dep_t1).sum(dim=0)  # (bs, d)
            # Apply accumulated corrections with otp_alpha
            X_t1 = X_base_t1 + X_t1_corr_total * otp_alpha

            if self.consistency_loss in ["meanflow", "imf"]:
                # (K, bs, 1)
                v_time_dep_t1 = torch.stack(
                    [self.v_time_dep(potential, t1) for potential in potentials_list]
                )
                u_corr_t1_total = (dV_tks * v_time_dep_t1).sum(dim=0)  # (bs, d)
                u_t1 = u_base + u_corr_t1_total * otp_alpha
            elif self.consistency_loss == "lsd":
                # (K, bs, 1)
                v_time_dep_t2 = torch.stack(
                    [self.v_time_dep(potential, t2) for potential in potentials_list]
                )
                u_corr_t2_total = (dV_tks * v_time_dep_t2).sum(dim=0)  # (bs, d)
                u_t2 = u_base + u_corr_t2_total * otp_alpha
        else:
            X_t1 = X_base_t1
            u_t1 = u_base
            u_t2 = u_base

        if debug:
            logger.debug(f"t1: {t1[0]}")
            logger.debug(f"t2: {t2[0]}")
            logger.debug(f"X_base_t1: {X_base_t1[0]}")
            logger.debug(f"u_base: {u_base[0]}")
            logger.debug(f"X_t1: {X_t1[0]}")
            if self.consistency_loss in ["meanflow", "imf"]:
                logger.debug(f"u_t1: {u_t1[0]}")
            elif self.consistency_loss == "lsd":
                logger.debug(f"u_t2: {u_t2[0]}")

        match self.consistency_loss:
            case "meanflow":
                loss = self.meanflow_loss(X_t1, t1, t2, u_t1, self.v_func, debug=debug)[0]
            case "imf":
                loss = self.improved_meanflow_loss(
                    X_t1, t1, t2, u_t1, self.v_func, debug=debug, x_func=self.flownet
                )[0]
            case "lsd":
                loss = self.lsd_loss(X_t1, t1, t2, u_t2, self.v_func, debug=debug)[0]
            case _:
                raise ValueError(f"Invalid consistency loss: {self.consistency_loss}")

        return loss

    def meanflow_loss(self, X_t1, t1, t2, u_t1, v_func: callable, debug: bool = False) -> Tensor:
        """
        Implements the MeanFlow loss from :cite:t:`geng2025meanflow` but going forward in time
        instead of backwards; see App. C.2 of :cite:t:`kansal2026otpfm`.

        Args:
            X_t1 (Tensor, shape ``(batch_size, *dim)``): position at time t1
            t1 (Tensor, shape ``(batch_size,)``): start timepoint
            t2 (Tensor, shape ``(batch_size,)``): end timepoint
            u_t1 (Tensor, shape ``(batch_size, *dim)``): instantaneous velocity to learn at time t1
        """
        ones = torch.ones_like(t1)
        zeros = torch.zeros_like(t1)

        # Ensures full precision for JVP calculation
        with torch.amp.autocast("cuda", enabled=False):
            v_pred, dvdt1 = torch.func.jvp(v_func, (X_t1, t1, t2), (u_t1, ones, zeros))

            # Clamp dvdr to prevent loss explosion from extreme derivatives
            dvdt1_clamped = dvdt1.clamp(-self.jvp_clamp, self.jvp_clamp)

            # switches MeanFlow formulation to privilege t1 instead of t2 (jumping forward instead of backward in time)
            v_tgt = ((t2 - t1) * dvdt1_clamped + u_t1).detach()  # no gradient through this

            if debug:
                logger.debug(f"v_pred: {v_pred[0]}")
                logger.debug(f"v_tgt: {v_tgt[0]}")
                logger.debug(f"dvdt1: {dvdt1[0]}, clamped: {dvdt1_clamped[0]}")

            # Get log_var for weighted loss (if enabled)
            log_var = None
            if self.lossfn == "weighted":
                log_var = self.flownet.get_log_var(t1.squeeze(), (t2 - t1).squeeze())

            loss = self.get_loss(v_pred, v_tgt, log_var=log_var)

            return loss, v_pred

    def improved_meanflow_loss(
        self, X_t1, t1, t2, u_t1, v_func: callable, debug: bool = False, x_func: callable = None
    ) -> Tensor:
        """
        Implements the Improved MeanFlow loss from :cite:t:`geng2026improved`,
        going forward in time instead of backwards; see App. C.2 of :cite:t:`kansal2026otpfm`.
        """
        ones = torch.ones_like(t1)
        zeros = torch.zeros_like(t1)

        # Ensures full precision for JVP calculation
        with torch.amp.autocast("cuda", enabled=False):
            # Marginal instantaneous velocity prediction from model
            if not self.x_pred or not self.x_pred_kwargs["lhopital"]:
                v_pred_t1 = v_func(X_t1, t1, t1)
            else:
                # Use L'Hopital's rule to get the instantaneous velocity at t1 as ∂x/∂t2 | t2 = t1
                v_pred_t1 = torch.func.jvp(lambda t2_in: x_func(X_t1, t1, t2_in), (t1,), (ones,))[
                    1
                ].detach()

            v_pred, dvdt1 = torch.func.jvp(v_func, (X_t1, t1, t2), (v_pred_t1, ones, zeros))

            # Clamp dvdt1 to prevent loss explosion from extreme derivatives
            # Detach to stop gradient through JVP
            dvdt1_clamped = dvdt1.detach().clamp(-self.jvp_clamp, self.jvp_clamp)

            # Re-parameterized output to regress against conditional instantaneous velocity
            # Switches MeanFlow formulation to privilege t1 instead of t2 (jumping forward instead of backward in time)
            U_t1 = v_pred - (t2 - t1) * dvdt1_clamped

            if debug:
                logger.debug(f"v_pred_t1: {v_pred_t1[0]}")
                logger.debug(f"v_pred: {v_pred[0]}")
                logger.debug(f"U_t1: {U_t1[0]}")
                logger.debug(f"dvdt1: {dvdt1[0]}, clamped: {dvdt1_clamped[0]}")

            # Get log_var for weighted loss (if enabled)
            log_var = None
            if self.lossfn == "weighted":
                log_var = self.flownet.get_log_var(t1.squeeze(), (t2 - t1).squeeze())

            loss = self.get_loss(U_t1, u_t1, log_var=log_var)

            return loss, v_pred

    def lsd_loss(self, X_t1, t1, t2, u_t2, v_func: callable, debug: bool = False) -> Tensor:
        """
        Implements the Lagrangian Self-Distillation loss from :cite:t:`boffi2025buildconsistencymodel`.
        """
        ones = torch.ones_like(t1)
        zeros_t = torch.zeros_like(t1)
        zeros_x = torch.zeros_like(X_t1)

        # Ensures full precision for JVP calculation
        with torch.amp.autocast("cuda", enabled=False):
            v_pred, dvdt2 = torch.func.jvp(v_func, (X_t1, t1, t2), (zeros_x, zeros_t, ones))
            dvdt2_clamped = dvdt2.clamp(-self.jvp_clamp, self.jvp_clamp)

            v_tgt = (u_t2 - (t2 - t1) * dvdt2_clamped).detach()  # no gradient through this

            if debug:
                logger.debug(f"v_pred: {v_pred[0]}")
                logger.debug(f"v_tgt: {v_tgt[0]}")
                logger.debug(f"dvdt2: {dvdt2[0]}, clamped: {dvdt2_clamped[0]}")

            # Get log_var for weighted loss (if enabled)
            log_var = None
            if self.lossfn == "weighted":
                log_var = self.flownet.get_log_var(t1.squeeze(), (t2 - t1).squeeze())

            loss = self.get_loss(v_pred, v_tgt, log_var=log_var)

            return loss, v_pred

    def get_loss(self, y_pred, y_true, log_var: Tensor = None) -> Tensor:
        """
        Compute loss based on lossfn type.

        Implements:

        - MSE loss
        - Adaptive loss: divide MSE by [MSE ^ (adaptive_exp) + 1e-3]
        - Weighted loss: EDM2-style *learned* loss weighting :cite:p:`karras2024analyzing`,
          L = MSE * exp(-log_var) + log_var

        Args:
            y_pred: Predicted values
            y_true: Target values
            log_var: Predicted log variance per sample (for lossfn="weighted")
        """
        sq_error = (y_pred - y_true) ** 2

        match self.lossfn:
            case "mse":
                loss_per_sample = sq_error.sum(dim=1)  # (bs,)
            case "adaptive":
                loss = sq_error.sum(dim=1)
                adp_wt = (loss.detach() ** self.adaptive_exp) + 1e-3
                loss_per_sample = loss / adp_wt
            case "weighted":
                assert log_var is not None, "log_var must be provided for weighted loss"
                weighted_sq_error = sq_error * torch.exp(-log_var)
                loss_per_sample = weighted_sq_error.sum(dim=1) + log_var.squeeze()
            case _:
                raise ValueError(f"Invalid loss function: {self.lossfn}")

        return loss_per_sample.mean()

    def _sample_n_steps_per_marginal_v(self, x0s: Tensor, n_steps: int, v_func: callable):
        """
        Sample the model in `n_steps` time steps per marginal using the given `v_func`,
        consistency model style: moving forward based on the average velocity.
        """
        device = x0s.device
        xs = [x0s.detach().cpu()]
        ts = []
        curr_tk = 0.0
        for j, next_tk in enumerate(self.tks[1:]):
            next_tk_val = next_tk.item()  # Convert to Python float
            step_size = (next_tk_val - curr_tk) / n_steps
            for i in range(n_steps):
                t1_val = i * step_size + curr_tk
                t2_val = (i + 1) * step_size + curr_tk
                t1 = torch.full((x0s.shape[0],), t1_val, device=device)
                t2 = torch.full((x0s.shape[0],), t2_val, device=device)
                v = v_func(x0s, t1, t2)
                x0s = x0s + (t2 - t1).view(-1, 1) * v
                xs.append(x0s.detach().cpu())
                ts.append(t1_val)
            curr_tk = next_tk_val
        ts.append(1.0)
        return torch.stack(xs), torch.tensor(ts)

    def _sample_n_steps_per_marginal_x(self, x0s: Tensor, n_steps: int, ema: bool = True):
        """
        Sample the model in `n_steps` time steps per marginal using x_prediction,
        i.e. simply use the flownet to jump to the next position.
        """
        device = x0s.device
        xs = [x0s.detach().cpu()]
        ts = []
        curr_tk = 0.0
        net = self.flownet_ema if ema else self.flownet

        for j, next_tk in enumerate(self.tks[1:]):
            next_tk_val = next_tk.item()  # Convert to Python float
            step_size = (next_tk_val - curr_tk) / n_steps
            dt = torch.full((x0s.shape[0],), step_size, device=device)
            for i in range(n_steps):
                t1_val = i * step_size + curr_tk
                t1 = torch.full((x0s.shape[0],), t1_val, device=device)
                x0s = net(x0s, t1, dt)
                xs.append(x0s.detach().cpu())
                ts.append(t1_val)
            curr_tk = next_tk_val
        ts.append(1.0)
        return torch.stack(xs), torch.tensor(ts)

    def sample(self, x0s: Tensor, n_steps: int, ema: bool = True):
        """
        Sample trajectories from the model.

        Args:
            x0s: Initial positions, shape (batch_size, dim)
            n_steps: Number of time steps per marginal segment
            ema: Whether to use EMA model for velocity

        Returns:
            xs: Trajectory tensor of shape (total_steps + 1, batch_size, dim)
            t_eval: Time points tensor of shape (total_steps + 1,)
        """
        if not self.x_pred or not self.x_pred_kwargs["x_sampling"]:
            return self._sample_n_steps_per_marginal_v(
                x0s, n_steps, v_func=self.v_func_ema if ema else self.v_func
            )
        else:
            return self._sample_n_steps_per_marginal_x(x0s, n_steps, ema=ema)


__all__ = ["OTPFM"]
