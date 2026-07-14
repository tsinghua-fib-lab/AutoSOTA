"""
GeneralEFCNetwork: Equilibrium Fisher Control for arbitrary architectures.

Key generalizations from the FC-only EFC:
  1. Blocks can be *any* nn.Module (Conv2d, BasicBlock, Linear, ...).
  2. Jacobians / VJPs are computed entirely through torch.autograd
     instead of manual matrix multiplications.
  3. Gamma (Fisher-modulated teaching signal) is computed via a
     finite-difference JVP through each block, avoiding the need
     to know the layer structure.
  4. Teaching-signal weight gradients are obtained by backpropagating
     through each block's computation graph.

Only Dynamical Inversion (DI) mode is supported.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.func import functional_call, vmap, grad


class GeneralEFCNetwork(nn.Module):
    """
    General-architecture EFC network.

    Accepts an ordered list of nn.Module *blocks*.  The last block must be a
    classifier (nn.Linear) whose output dimension equals the total number of
    classes across all tasks.
    """

    def __init__(self, blocks, config, name="GeneralEFC"):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.name = name

        # ---- loss ----
        self.loss_fn_name = config.loss_fn
        self.loss_fn = (
            nn.CrossEntropyLoss() if config.loss_fn == "ce" else nn.MSELoss()
        )

        # ---- device / training ----
        self.device = config.device
        self.lr = config.lr
        self.setting = config.setting

        # ---- DI hyper-parameters ----
        self.dt = float(config.dt_di)
        self.tau = float(config.time_constant_ratio)
        self.k_p = float(config.k_p)
        self.tmax = int(config.tmax_di)
        self.eps = float(config.eps)
        self.target_lr = float(config.target_lr)
        self.beta = float(config.beta_efc)

        # ---- continual learning ----
        self.num_tasks = config.num_tasks
        self.classes_per_task = config.classes_per_task
        self._setup_task_masks()

        # ---- Fisher storage ----
        self._fisher = {}
        self._theta_star = {}
        self._first_task = True
        self.task_id = 0

        # ---- loss-specific helpers ----
        if config.loss_fn == "ce":
            self._compute_error = self._error_ce
            self._set_targets = self._targets_ce
        else:
            self._compute_error = self._error_mse
            self._set_targets = self._targets_mse

    # ------------------------------------------------------------------
    # Task masks (identical logic to the FC code)
    # ------------------------------------------------------------------

    def _setup_task_masks(self):
        self.task_masks = {}
        self.task_masks_complement = {}
        s = self.setting.lower()
        for t in range(self.num_tasks):
            if "taskil" in s:
                start = t * self.classes_per_task
                end = (t + 1) * self.classes_per_task
                self.task_masks[t] = slice(start, end)
                self.task_masks_complement[t] = list(range(0, start)) + list(
                    range(end, self.num_tasks * self.classes_per_task)
                )
            elif "classil" in s:
                end = (t + 1) * self.classes_per_task
                self.task_masks[t] = slice(0, end)
                self.task_masks_complement[t] = list(
                    range(end, self.num_tasks * self.classes_per_task)
                )
            else:  # domainIL
                self.task_masks[t] = slice(None)
                self.task_masks_complement[t] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """Standard forward pass; caches activations at every block boundary."""
        self.input = x
        self.bzs = x.shape[0]
        self.block_activations = []
        for block in self.blocks:
            x = block(x)
            self.block_activations.append(x)
        x = x[:, self.task_masks[self.task_id]]
        self.y_hat = x
        return x

    def calculate_loss(self, y_hat, y):
        self.loss = self.loss_fn(y_hat, y)
        return self.loss

    # ------------------------------------------------------------------
    # Error / target helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error_mse(y_hat, y):
        return y - y_hat

    @staticmethod
    def _error_ce(y_hat, y):
        return y - F.softmax(y_hat, dim=1)

    def _targets_mse(self, y):
        self.targets = (
            (1 - 2 * self.target_lr) * self.y_hat + 2 * self.target_lr * y
        )
        self.output_size = self.targets.shape[1]

    def _targets_ce(self, y):
        sm = F.softmax(self.y_hat, dim=1)
        self.targets = sm - self.target_lr * (sm - y)
        self.output_size = self.targets.shape[1]

    # ==================================================================
    #  BACKWARD  =  DI  +  teaching-signal gradients
    # ==================================================================

    def backward(self, y):
        """EFC backward pass: run DI, then compute per-block gradients."""
        self._set_targets(y)
        self._dynamical_inversion()
        self._apply_teaching_gradients()

    # ------------------------------------------------------------------
    # Dynamical Inversion
    # ------------------------------------------------------------------

    def _dynamical_inversion(self):
        """
        Iteratively refine block activations so that the output moves
        towards the target.  Uses proportional control with Fisher-gamma
        modulation, generalised to arbitrary blocks via autograd.
        """
        L = len(self.blocks)
        device = self.input.device

        u_current = torch.zeros(self.bzs, self.output_size, device=device)
        converged = torch.zeros(self.bzs, dtype=torch.bool, device=device)

        # Start from the feed-forward activations
        activations = [a.detach().clone() for a in self.block_activations]

        # Fix BatchNorm during DI (use running stats, no stat updates)
        bn_state = self._save_bn_training_state()
        self._set_bn_eval()

        t = 0
        while converged.float().mean().item() <= 0.99 and t < self.tmax:
            print(t)
            t += 1
            if converged.all():
                break

            # 1. Re-compute feed-forward outputs from current activations
            block_ffs = self._recompute_ffs(activations)

            # 2. Output error & proportional control
            output = block_ffs[-1][:, self.task_masks[self.task_id]]
            error = self._compute_error(output, self.targets)
            u_next = self.k_p * error

            # 3. Teaching signals via chained autograd VJPs
            psis = self._compute_psis(u_next, activations)

            # 4. Update every block's activation
            for i in range(L):
                r_ff = block_ffs[i]
                psi = psis[i]
                gamma = self._compute_gamma_block(
                    i,
                    activations[i - 1] if i > 0 else self.input,
                    r_ff,
                )

                if i == L - 1:
                    # Classifier: only update task-active outputs
                    ts = self.task_masks[self.task_id]
                    psi_t = psi[:, ts]
                    gamma_t = gamma[:, ts] if isinstance(gamma, torch.Tensor) else gamma
                    e = torch.tanh(psi_t + gamma_t) + 1
                    activations[i][:, ts] = activations[i][:, ts] + (
                        self.dt / self.tau * (e * r_ff[:, ts] - activations[i][:, ts])
                    )
                else:
                    e = torch.tanh(psi + gamma) + 1
                    activations[i] = activations[i] + self.dt / self.tau * (
                        e * r_ff - activations[i]
                    )

            # 5. Convergence check
            converged = converged | (
                torch.norm(u_next - u_current, dim=1) < self.eps
            )
            u_current = u_next

        # Reset non-converged samples to feed-forward
        mask = ~converged
        if mask.any():
            for i in range(L):
                activations[i][mask] = self.block_activations[i][mask].detach()

        self._target_activations = activations
        self._restore_bn_training_state(bn_state)

    # ------------------------------------------------------------------
    # Autograd-based psi (teaching signal) computation
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def _compute_psis(self, u, activations):
        """
        Compute per-block teaching signals by chaining block-local VJPs.

        For block *i* the teaching signal is
            psi_i  =  (d output / d r_i)^T  u
        where r_i is the output of block i.

        We chain VJPs backward through the blocks.  Each block's local
        graph is built from the *current* (DI-modified) activations so
        that the Jacobians reflect the latest operating point.
        """
        L = len(self.blocks)

        # --- forward: build one local graph per block ---
        local_graphs = []
        for i, block in enumerate(self.blocks):
            r_in = (
                activations[i - 1].detach() if i > 0 else self.input.detach()
            ).requires_grad_(True)
            r_out = block(r_in)
            local_graphs.append((r_in, r_out))

        # --- backward: chain VJPs from the output to the first block ---
        psis = [None] * L
        v = u.detach()

        for i in range(L - 1, -1, -1):
            r_in, r_out = local_graphs[i]

            if i == L - 1:
                # Last block (classifier): pad u with zeros for inactive outputs
                full_v = torch.zeros_like(r_out)
                full_v[:, self.task_masks[self.task_id]] = v
                psis[i] = full_v.detach()
                v_prev = torch.autograd.grad(
                    r_out, r_in, grad_outputs=full_v, retain_graph=False
                )[0]
            else:
                psis[i] = v.detach()
                v_prev = torch.autograd.grad(
                    r_out, r_in, grad_outputs=v, retain_graph=False
                )[0]

            v = v_prev.detach()

        return psis

    # ------------------------------------------------------------------
    # Gamma (Fisher-modulated signal) via finite-difference JVP
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_gamma_block(self, block_idx, r_prev, r_ff):
        """
        Compute the Fisher-modulated gamma for one block.

        gamma = -beta * JVP(block, theta, F * Delta_theta) / ||F||

        The JVP is approximated by a central finite difference to avoid
        dependence on torch.func.jvp (which can be fragile with BN / buffers).
        """
        if self._first_task:
            return 0.0

        block = self.blocks[block_idx]

        # Collect original state (parameters + buffers)
        orig_state = {}
        for name, buf in block.named_buffers():
            orig_state[name] = buf
        for name, param in block.named_parameters():
            orig_state[name] = param.data

        # Build perturbed state: theta + eps * F * Delta_theta
        has_fisher = False
        eps_fd = 1e-5
        perturbed_state = dict(orig_state)
        for name, param in block.named_parameters():
            full_name = f"blocks.{block_idx}.{name}"
            if full_name in self._fisher:
                has_fisher = True
                tangent = self._fisher[full_name] * (
                    param.data - self._theta_star[full_name]
                )
                perturbed_state[name] = param.data + eps_fd * tangent

        if not has_fisher:
            return 0.0

        r_ff_plus = functional_call(block, perturbed_state, (r_prev,))
        gamma_unnorm = (r_ff_plus - r_ff) / eps_fd

        # Normalise by per-element Fisher magnitude
        fisher_norm_sq = sum(
            (self._fisher[f"blocks.{block_idx}.{n}"] ** 2).sum()
            for n, _ in block.named_parameters()
            if f"blocks.{block_idx}.{n}" in self._fisher
        )
        num_out = max(r_ff[0].numel(), 1)
        fisher_norm = torch.sqrt(fisher_norm_sq / num_out + 1e-8)

        return -self.beta * gamma_unnorm / fisher_norm

    # ------------------------------------------------------------------
    # Feed-forward recomputation helper (used inside DI)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _recompute_ffs(self, activations):
        """Re-run each block from the current (modified) activations."""
        ffs = []
        for i, block in enumerate(self.blocks):
            r_prev = activations[i - 1] if i > 0 else self.input
            ffs.append(block(r_prev))
        return ffs

    # ------------------------------------------------------------------
    # Teaching-signal weight gradients via autograd
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def _apply_teaching_gradients(self):
        """
        For every block, compute

            grad_theta  =  -2/N  *  (r_target - r_ff)  .  d r_ff / d theta

        by running block(r_prev) with autograd and backpropagating the
        teaching signal.  Since r_prev is detached, only the current
        block's parameters receive gradients.
        """
        # Zero all parameter gradients
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Use eval-mode BN for consistency with DI
        bn_state = self._save_bn_training_state()
        self._set_bn_eval()

        for i in range(len(self.blocks)):
            # Skip blocks with no trainable parameters (e.g. pooling)
            if not any(p.requires_grad for p in self.blocks[i].parameters()):
                continue

            r_prev = (
                self._target_activations[i - 1].detach()
                if i > 0
                else self.input.detach()
            )
            r_target = self._target_activations[i].detach()

            # Forward through block (creates local graph for this block only)
            r_ff = self.blocks[i](r_prev)

            # Teaching signal gradient:
            # d/d_theta (1/2N)||r*-r_ff||^2  =  -1/N (r*-r_ff) . d r_ff/d_theta
            teaching_signal = (r_target - r_ff).detach()
            r_ff.backward(gradient=-2.0 / self.bzs * teaching_signal)

        self._restore_bn_training_state(bn_state)

    # ==================================================================
    #  FISHER  computation  &  continual-learning bookkeeping
    # ==================================================================

    def _calculate_fisher(self, dataloader):
        """Diagonal Fisher via vmap + per-sample gradients."""
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.named_parameters()
            if p.requires_grad
        }
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        buffers = {n: b for n, b in self.named_buffers()}

        def compute_ll(params, buffers, x, y):
            out = functional_call(self, (params, buffers), (x.unsqueeze(0),))
            return (F.log_softmax(out, dim=1) * y.unsqueeze(0)).sum()

        grad_fn = vmap(grad(compute_ll), in_dims=(None, None, 0, 0))

        self.eval()
        total = 0
        pbar = tqdm(total=len(dataloader), desc="Fisher", leave=True)
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            bs = inputs.size(0)

            per_sample_grads = grad_fn(params, buffers, inputs, targets)

            for n in fisher:
                fisher[n] += (per_sample_grads[n] ** 2).sum(dim=0)
            total += bs
            pbar.update(1)
        pbar.close()

        for n in fisher:
            fisher[n] /= total
        return fisher

    def start_task(self, dataloader):
        pass

    def complete_task(self, dataloader):
        """Compute Fisher, store reference params, accumulate across tasks."""
        current_fisher = self._calculate_fisher(dataloader)
        self._theta_star = {
            n: p.data.clone()
            for n, p in self.named_parameters()
            if p.requires_grad
        }

        if self._first_task:
            self._fisher = current_fisher
            self._first_task = False
        else:
            # Rescale so that accumulated Fisher keeps balanced magnitudes
            old_norm = sum(
                torch.norm(f).item() ** 2 for f in self._fisher.values()
            ) ** 0.5
            new_norm = sum(
                torch.norm(f).item() ** 2 for f in current_fisher.values()
            ) ** 0.5
            scale = old_norm / (new_norm + 1e-8)
            for n in self._fisher:
                self._fisher[n] = self._fisher[n] + scale * current_fisher[n]

    # ------------------------------------------------------------------
    # BatchNorm helpers
    # ------------------------------------------------------------------

    def _save_bn_training_state(self):
        return {
            name: m.training
            for name, m in self.named_modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
        }

    def _set_bn_eval(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()

    def _restore_bn_training_state(self, state):
        for name, m in self.named_modules():
            if name in state:
                m.training = state[name]
