import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.func import functional_call

from networks.network_interface import Network, FisherInterface
from networks.layers import BP_layer
from networks.activation_function import Softplus, Linear


class CSQN_network(Network, FisherInterface):
    """
    Continual Learning with Sampled Quasi-Newton (CSQN).

    Vander Eeckt & Van Hamme (2025), "Continual Learning with Quasi-Newton
    Methods", IEEE Access.

    Extends EWC by replacing the diagonal Fisher with a low-rank + diagonal
    Hessian approximation obtained via Sampled Quasi-Newton methods.

    For SR1:  B = B0 + ZZ^T, where B0 = diag(sum of per-task Fishers)
    For BFGS: B = B0 - [B0*S  Y] M^{-1} [B0*S  Y]^T  (compact form, Eq. 8)

    Regularisation loss (Eq. 11):
        L_{t+1}(θ) = L^ce_{t+1}(θ) + (λ/2)(θ - θ*)^T [Σ_j B^(j)] (θ - θ*)

    Supports reduction strategies (Section III-E3):
        None: store all Z / (S,Y) per task  (linear memory in T)
        'ct':  SVD-reduce accumulated Z after each task (constant memory)
        'mrt': CSQN for most recent task, EWC for older tasks (constant memory)
    """

    def __init__(self, config, name="CSQN_network"):
        Network.__init__(self, BP_layer, Softplus, Linear, config, name)
        FisherInterface.__init__(self)

        # Regularisation strength
        self.importance = config.importance_ewc

        # SQN hyperparameters
        self.M = getattr(config, "csqn_M", 20)
        self.csqn_epsilon = getattr(config, "csqn_epsilon", 1e-4)
        self.csqn_kappa = getattr(config, "csqn_kappa", 1e-12)
        self.qn_method = getattr(config, "csqn_method", "sr1")  # 'sr1' | 'bfgs'
        self.reduce_strategy = getattr(config, "csqn_reduce", None)  # None | 'ct' | 'mrt'

        # Accumulated diagonal Hessian:  B0 = Σ_j Ω^(j)  (dict, same keys as named_parameters)
        self._B0 = {}

        # SR1 low-rank factor:  Z  s.t.  low-rank part = ZZ^T   [N_params, cols]
        self._Z = None

        # BFGS per-task storage (only when reduce_strategy is None)
        self._S_list = []  # list of [N_params, M_j] tensors
        self._Y_list = []  # list of [N_params, M_j] tensors

        # Cached parameter metadata for fast flatten / unflatten
        self._param_meta = None

    # ------------------------------------------------------------------
    #  Parameter flattening helpers
    # ------------------------------------------------------------------

    def _ensure_param_meta(self):
        if self._param_meta is None:
            self._param_meta = [
                (n, p.shape, p.numel())
                for n, p in self.named_parameters()
                if p.requires_grad
            ]

    def _flat_names(self):
        self._ensure_param_meta()
        return [m[0] for m in self._param_meta]

    def _flatten_dict(self, d):
        """Flatten a {name: tensor} dict in canonical parameter order."""
        self._ensure_param_meta()
        return torch.cat([d[n].flatten() for n, _, _ in self._param_meta])

    def _flatten_params(self):
        self._ensure_param_meta()
        return torch.cat(
            [p.data.flatten() for n, p in self.named_parameters() if p.requires_grad]
        )

    def _unflatten_to_dict(self, flat, requires_grad=False):
        """Unflatten a vector back into a {name: tensor} dict."""
        self._ensure_param_meta()
        d = {}
        offset = 0
        for name, shape, numel in self._param_meta:
            t = flat[offset : offset + numel].reshape(shape)
            if requires_grad:
                t = t.clone().detach().requires_grad_(True)
            d[name] = t
            offset += numel
        return d

    def _flat_param_diff(self):
        """v = θ − θ*  as a single flat vector (for regularisation loss)."""
        self._ensure_param_meta()
        return torch.cat(
            [
                (p - self._theta_star[n]).flatten()
                for n, p in self.named_parameters()
                if p.requires_grad
            ]
        )

    # ------------------------------------------------------------------
    #  Training:  backward  +  regularisation loss
    # ------------------------------------------------------------------

    def backward(self, y):
        loss = self.loss_fn(self.y_hat, y)
        if not self._first_task:
            loss += self.csqn_loss()
        loss.backward()

    def csqn_loss(self):
        """
        (λ/2) v^T B v   where  B = B0 + low-rank correction.

        Diagonal part:  Σ_n B0_n (θ_n − θ*_n)^2
        Low-rank part:  method-dependent
        """
        # --- diagonal (EWC-like) term ---
        diag_loss = 0.0
        for n, p in self.named_parameters():
            if n in self._theta_star and n in self._B0:
                diag_loss += torch.sum(self._B0[n] * (p - self._theta_star[n]) ** 2)

        # --- low-rank correction ---
        if self.qn_method == "sr1":
            lr_loss = self._sr1_lowrank_loss()
        else:
            lr_loss = self._bfgs_lowrank_loss()

        return 0.5 * self.importance * (diag_loss + lr_loss)

    def _sr1_lowrank_loss(self):
        """‖Z^T v‖²  for SR1 (B − B0 = ZZ^T)."""
        if self._Z is None or self._Z.shape[1] == 0:
            return 0.0
        v = self._flat_param_diff()
        Ztv = self._Z.t() @ v
        return torch.sum(Ztv**2)

    def _bfgs_lowrank_loss(self):
        """
        BFGS compact-form correction (Eq. 8):
            B − B0 = −[B0S  Y] M^{-1} [B0S  Y]^T
        Quadratic form:  v^T (B−B0) v = −w^T M^{-1} w
        where  w = [(B0S)^T v ;  Y^T v].
        """
        if not self._S_list:
            # When using ct/mrt with BFGS, the correction is stored as Z
            if self._Z is not None and self._Z.shape[1] > 0:
                return self._sr1_lowrank_loss()  # ZZ^T form
            return 0.0

        v = self._flat_param_diff()
        b0 = self._flatten_dict(self._B0)

        S = torch.cat(self._S_list, dim=1)
        Y = torch.cat(self._Y_list, dim=1)
        m = S.shape[1]

        B0S = b0.unsqueeze(1) * S

        # D_ii = s_i^T y_i
        D = torch.einsum("ni,ni->i", S, Y)

        # L lower-triangular:  L_ij = s_i^T y_j  for i > j
        StY = S.t() @ Y
        L_mat = torch.tril(StY, diagonal=-1)

        StB0S = S.t() @ B0S

        # Block matrix  M = [[S^T B0 S, L], [L^T, -diag(D)]]
        M_mat = torch.zeros(2 * m, 2 * m, device=v.device)
        M_mat[:m, :m] = StB0S
        M_mat[:m, m:] = L_mat
        M_mat[m:, :m] = L_mat.t()
        M_mat[m:, m:] = -torch.diag(D)

        w = torch.cat([B0S.t() @ v, Y.t() @ v])

        try:
            Minv_w = torch.linalg.solve(M_mat, w)
            return -(w @ Minv_w)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    #  Task completion  (Algorithm 3)
    # ------------------------------------------------------------------

    def complete_task(self, dataloader):
        """
        After training on task t:
          1.  Compute diagonal Fisher  Ω^(t)
          2.  Store θ*
          3.  Accumulate  B0 += Ω^(t)
          4.  Sample S, Y  (Algorithm 1)
          5.  Compute Z  (Algorithm 2 for SR1) or store S, Y (BFGS)
          6.  Optionally reduce memory
        """
        # 1. Diagonal Fisher (reuse FisherInterface)
        omega = self._calculate_fisher(dataloader)

        # 2. Reference parameters
        self._theta_star = {
            n: p.data.clone()
            for n, p in self.named_parameters()
            if p.requires_grad
        }

        # 3. Accumulate B0
        if self._first_task:
            self._B0 = {n: f.clone() for n, f in omega.items()}
        else:
            for n in self._B0:
                self._B0[n] = self._B0[n] + omega[n]

        # 4. SQN sampling
        S, Y = self._compute_SY(dataloader, omega)

        # 5 & 6. Store / reduce
        if self.qn_method == "sr1":
            b0_flat = self._flatten_dict(omega)
            Z_new = self._compute_Z_sr1(S, Y, b0_flat)
            self._accumulate_Z(Z_new)

        else:  # bfgs
            if self.reduce_strategy in ("ct", "mrt"):
                # Convert to ZZ^T form so we can reduce
                b0_flat = self._flatten_dict(omega)
                Z_new = self._compute_Z_bfgs(S, Y, b0_flat)
                self._accumulate_Z(Z_new)
            else:
                # Store raw S, Y per task (no reduction)
                if S.shape[1] > 0:
                    self._S_list.append(S.detach())
                    self._Y_list.append(Y.detach())

        if self._first_task:
            self._first_task = False

    def _accumulate_Z(self, Z_new):
        """Concatenate new Z columns and optionally reduce."""
        if Z_new is None or Z_new.shape[1] == 0:
            return

        Z_new = Z_new.detach()

        if self.reduce_strategy == "mrt":
            # Most-Recent-Task: only keep this task's low-rank factor
            self._Z = Z_new
            return

        if self._Z is None:
            self._Z = Z_new
        else:
            self._Z = torch.cat([self._Z, Z_new], dim=1)

        # CT: SVD-reduce to target column count after each task
        target = self.M if self.qn_method == "sr1" else 2 * self.M
        if self.reduce_strategy == "ct" and self._Z.shape[1] > target:
            self._Z = self._reduce_Z_svd(self._Z, target)

    @staticmethod
    def _reduce_Z_svd(Z, target_cols):
        """
        SVD-based column reduction with magnitude compensation (Section III-E3, CT).
        Multiplies by  √(cols_before / cols_after)  to preserve regularisation scale.
        """
        if Z.shape[1] <= target_cols:
            return Z

        cols_before = Z.shape[1]
        U, S_vals, _ = torch.linalg.svd(Z, full_matrices=False)
        Z_reduced = U[:, :target_cols] * S_vals[:target_cols].unsqueeze(0)
        scale = (cols_before / target_cols) ** 0.5
        return Z_reduced * scale

    # ------------------------------------------------------------------
    #  SQN Sampling  (Algorithm 1)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_SY(self, dataloader, omega):
        """
        Sample M perturbations around θ* and build the (S, Y) pair matrices.

        Σ = (Ω + ε·max(Ω)·I)^{-1}   (Eq. below Alg. 1 in the paper)
        s_i = θ* − x̃_i ,   y_i = ∇L(θ*) − ∇L(x̃_i)

        Returns S [N, M'], Y [N, M'] where M' ≤ M (pairs failing the
        curvature condition are discarded).
        """
        theta_flat = self._flatten_params()
        N = theta_flat.numel()
        device = theta_flat.device

        # Covariance for sampling
        omega_flat = self._flatten_dict(omega)
        max_omega = omega_flat.max().item()
        sigma = 1.0 / (omega_flat + self.csqn_epsilon * max_omega + 1e-30)
        sigma_sqrt = sigma.sqrt()

        # Gradient at θ*
        grad_star = self._compute_gradient_at(dataloader, theta_flat)

        S_cols, Y_cols = [], []

        for _ in tqdm(range(self.M), desc=f"SQN ({self.qn_method.upper()})", leave=True):
            noise = torch.randn(N, device=device) * sigma_sqrt
            x_tilde = theta_flat + noise
            s = -noise  # θ* − x̃

            grad_tilde = self._compute_gradient_at(dataloader, x_tilde)
            y = grad_star - grad_tilde

            # Curvature condition
            if self.qn_method == "bfgs":
                if s @ y > self.csqn_kappa * s.norm() ** 2:
                    S_cols.append(s)
                    Y_cols.append(y)
            else:  # sr1
                B0s = omega_flat * s
                if torch.abs(s @ (y - B0s)) >= self.csqn_kappa * s.norm() ** 2:
                    S_cols.append(s)
                    Y_cols.append(y)

        if not S_cols:
            return (
                torch.zeros(N, 0, device=device),
                torch.zeros(N, 0, device=device),
            )

        return torch.stack(S_cols, dim=1), torch.stack(Y_cols, dim=1)

    @torch.enable_grad()
    def _compute_gradient_at(self, dataloader, params_flat):
        """
        Compute ∇L^ce(params_flat) averaged over the full dataset.

        Uses functional_call to evaluate the network at an arbitrary
        parameter vector without mutating the stored parameters.
        """
        params_dict = self._unflatten_to_dict(params_flat, requires_grad=True)
        buffers = {n: b for n, b in self.named_buffers()}
        names = self._flat_names()

        was_training = self.training
        self.eval()

        grad_accum = {n: torch.zeros_like(params_dict[n]) for n in names}
        total_samples = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            output = functional_call(self, (params_dict, buffers), (inputs,))
            loss = self.loss_fn(output, targets)

            grads = torch.autograd.grad(
                loss, [params_dict[n] for n in names], create_graph=False
            )

            bs = inputs.size(0)
            for n, g in zip(names, grads):
                grad_accum[n].add_(g.detach(), alpha=bs)
            total_samples += bs

        if was_training:
            self.train()

        return (
            torch.cat([grad_accum[n].flatten() for n in names]) / total_samples
        )

    # ------------------------------------------------------------------
    #  Z computation  (Algorithm 2 – SR1)
    # ------------------------------------------------------------------

    def _compute_Z_sr1(self, S, Y, b0_flat):
        """
        Algorithm 2:  compute Z  s.t.  B^(t) − diag(Ω^(t)) = ZZ^T   for SR1.

        From Eq. 9:   B = B0 + X A^{-1} X^T
            where  X = Y − B0·S
                   A = D + L + L^T − S^T B0 S
                   D_ii = s_i^T y_i ,   L_ij = s_i^T y_j  (i > j)

        We decompose  A^{-1} = L_chol L_chol^T  (or clip negative eigenvalues)
        so that  Z = X L_chol.
        """
        if S.shape[1] == 0:
            return None

        m = S.shape[1]
        device = S.device

        X = Y - b0_flat.unsqueeze(1) * S  # [N, m]

        D = torch.einsum("ni,ni->i", S, Y)
        StY = S.t() @ Y
        L_mat = torch.tril(StY, diagonal=-1)
        B0S = b0_flat.unsqueeze(1) * S
        StB0S = S.t() @ B0S

        A = torch.diag(D) + L_mat + L_mat.t() - StB0S  # [m, m]

        try:
            A_inv = torch.linalg.inv(A)
        except Exception:
            return None

        # Eigendecomposition of A^{-1}
        eigvals, V = torch.linalg.eigh(A_inv)

        if eigvals.min() > 1e-12:
            # A^{-1} is PD → Cholesky
            try:
                L_chol = torch.linalg.cholesky(A_inv)
                return X @ L_chol
            except Exception:
                pass

        # Clip negative eigenvalues, then QR to get the factor
        eigvals_pos = torch.clamp(eigvals, min=0)
        VsqrtG = V * eigvals_pos.sqrt().unsqueeze(0)  # [m, m]
        _, R = torch.linalg.qr(VsqrtG.t())             # R is [m, m]
        return X @ R.t()

    # ------------------------------------------------------------------
    #  Z computation  (BFGS → ZZ^T form, for reduction strategies)
    # ------------------------------------------------------------------

    def _compute_Z_bfgs(self, S, Y, b0_flat):
        """
        Convert the BFGS compact correction to a ZZ^T factor.

        B − B0 = −C M^{-1} C^T   where C = [B0·S  Y].
        If −M^{-1} has positive eigenvalues, we extract Z from those.
        """
        if S.shape[1] == 0:
            return None

        m = S.shape[1]
        device = S.device

        B0S = b0_flat.unsqueeze(1) * S
        D = torch.einsum("ni,ni->i", S, Y)
        StY = S.t() @ Y
        L_mat = torch.tril(StY, diagonal=-1)
        StB0S = S.t() @ B0S

        M_mat = torch.zeros(2 * m, 2 * m, device=device)
        M_mat[:m, :m] = StB0S
        M_mat[:m, m:] = L_mat
        M_mat[m:, :m] = L_mat.t()
        M_mat[m:, m:] = -torch.diag(D)

        C = torch.cat([B0S, Y], dim=1)  # [N, 2m]

        try:
            M_inv = torch.linalg.inv(M_mat)
        except Exception:
            return None

        # We need  −M^{-1}  to have positive eigenvalues for the ZZ^T decomposition
        neg_M_inv = -M_inv
        eigvals, V = torch.linalg.eigh(neg_M_inv)

        pos = eigvals > 1e-12
        if not pos.any():
            return None

        # Z = C @ V_pos @ diag(sqrt(eigvals_pos))
        V_pos = V[:, pos]
        sqrt_eig = eigvals[pos].sqrt()
        return C @ (V_pos * sqrt_eig.unsqueeze(0))

    # ------------------------------------------------------------------
    #  Device management
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        device = next(self.parameters()).device

        for d in [self._B0, self._theta_star]:
            for key in d:
                d[key] = d[key].to(device)

        if self._Z is not None:
            self._Z = self._Z.to(device)

        self._S_list = [s.to(device) for s in self._S_list]
        self._Y_list = [y.to(device) for y in self._Y_list]

        return self

    # ------------------------------------------------------------------
    #  Monitoring
    # ------------------------------------------------------------------

    def get_csqn_stats(self):
        """Return storage / regularisation statistics for logging."""
        stats = {}
        if self._B0:
            stats["B0_norm"] = sum(
                torch.norm(f).item() ** 2 for f in self._B0.values()
            ) ** 0.5

        if self._Z is not None:
            stats["Z_cols"] = self._Z.shape[1]
            stats["Z_norm"] = torch.norm(self._Z).item()

        if self._S_list:
            total_cols = sum(s.shape[1] for s in self._S_list)
            stats["bfgs_total_cols"] = total_cols

        return stats