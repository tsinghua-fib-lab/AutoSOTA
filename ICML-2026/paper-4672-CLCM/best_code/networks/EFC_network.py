import torch

from networks.network_interface import *
from networks.layers import *
from networks.activation_function import *


class EFC_network(Network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="EFC_network"):
        Network.__init__(self, DFC_layer, ReLU, Linear, config, name)
        JacobianInterface.__init__(self, config)
        FisherInterface.__init__(self)

        self.beta = config.beta_efc

    @torch.no_grad()
    def _non_dynamical_inversion(self):
        # Calculate Jacobians for each layer
        Js = self._calculate_layerwise_jacobians()

        # Compute J_eff and gamma_eff with Q = J^T
        J_eff, gamma_eff, J_i, gamma_i = self._calculate_jeff_and_gammaeff()

        # Compute the output error
        delta_L_minus = self._compute_error(self.y_hat, self.targets)

        # Solve for u_star: (alpha I + J_eff) u_star = delta_L_minus - gamma_eff
        u_star = torch.linalg.solve(
            J_eff + self.alpha_I * torch.eye(J_eff.shape[1], device=self.device), delta_L_minus - gamma_eff
        )

        # Compute the control signal for each layer
        Qu_i = [
            torch.bmm(J_i[i].transpose(1, 2), u_star.unsqueeze(-1)).squeeze(-1)
            for i in range(len(Js))
        ]

        # Compute delta_r = (I - J_{i,i-1})^-1 (Qu*_i + gamma_i) for each layer and update r^* = r^- + delta_r
        delta_r_prev = torch.zeros_like(self.input)
        for i, layer in enumerate(self.layers):
            # (Qu*_i + γ_i) ⊙ r^-_i + J_{i,i-1} * Δr_{i-1} where J_{i,i-1} = φ'(W_i * r^-_{i-1}) ⊙ W_i
            # This is equivalent to φ'(pre_activation) ⊙ (W_i @ Δr_{i-1})
            if i == len(self.layers) - 1:  # Only update current task's activations
                delta_r_i = torch.zeros_like(layer.r_ff)
                task_slice = self.task_masks[self.task_id]
                delta_r_i[:, task_slice] = (Qu_i[i] + gamma_i[i]) * layer.r_ff[
                    :, task_slice
                ] + torch.matmul(Js[i], delta_r_prev.unsqueeze(-1)).squeeze(-1)
            else:
                delta_r_i = (Qu_i[i] + gamma_i[i]) * layer.r_ff + torch.matmul(
                    Js[i], delta_r_prev.unsqueeze(-1)
                ).squeeze(-1)

            delta_r_prev = delta_r_i

            layer.r = layer.r_ff + delta_r_i

    @torch.no_grad()
    def _dynamical_inversion(self):
        u_current = torch.zeros((self.bzs, self.output_size), device=self.device)
        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool, device=self.device)

        t = 0

        while converged_mask.float().mean().item() <= 0.99 and t < self.tmax:
            t = t + 1
            # Stop if converged
            if converged_mask.all():
                break
            error = self._compute_error(
                self.layers[-1].r[:, self.task_masks[self.task_id]], self.targets
            )

            # Proportional control
            u_next = self.k_p * error
            psis = self._calculate_psis(u_next)

            # Iterate over layers with control signal
            for i, layer in enumerate(self.layers):
                layer.r_prev = self.layers[i - 1].r if i != 0 else self.input
                layer.r_ff = layer.forward(layer.r_prev)

                # Apical with teaching signal and Fisher modulation
                psi = psis[i]
                gamma = self._compute_gamma(layer, i)

                # Soma with modulation
                if (
                    i == len(self.layers) - 1
                ):  # For final layer, only update current task neurons
                    task_slice = self.task_masks[self.task_id]
                    psi_task = psi[:, task_slice]
                    e_psi_gamma = torch.tanh(psi_task + gamma) + 1

                    delta_r = (
                        self.dt
                        / self.time_constant_ratio
                        * (
                            e_psi_gamma * layer.r_ff[:, task_slice]
                            - layer.r[:, task_slice]
                        )
                    )
                    layer.r[:, task_slice] = layer.r[:, task_slice] + delta_r
                else:
                    e_psi_gamma = torch.tanh(psi + gamma) + 1
                    layer.r = layer.r + self.dt / self.time_constant_ratio * (
                        e_psi_gamma * layer.r_ff - layer.r
                    )

            # Compute convergence check
            converged_mask |= torch.norm(u_next - u_current, dim=1) < self.eps
            u_current = u_next

        mask = ~converged_mask
        if mask.any():
            for i, layer in enumerate(self.layers):
                layer.r[mask] = layer.r_ff[mask]

    # In EFC_network class
    def _calculate_full_fisher(self, dataloader):
        """Override to use gamma-modulated Fisher computation"""
        return self._calculate_full_fisher_with_gamma(dataloader)

    def _calculate_full_fisher_with_gamma(self, dataloader):
        """Compute full Fisher Information Matrix with EFC gamma modulation (vectorized)"""
        params_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}

        param_count = sum(p.numel() for p in params_dict.values())
        fisher = torch.zeros(param_count, param_count, dtype=torch.float32, device=self.device)

        # Capture EFC-specific quantities in closure
        fisher_diag = self._fisher
        theta_star = self._theta_star
        beta = self.beta
        n_layers = len(self.layers)
        task_mask = self.task_masks[0]  # Task A mask (classes 0-4)

        def compute_loss_single_with_gamma(params, x, y):
            """Compute log likelihood for a single sample with gamma modulation"""
            h = x.flatten()

            for layer_idx in range(n_layers):
                # Get params for this layer
                W = params[f"layers.{layer_idx}._weights"]
                b = params[f"layers.{layer_idx}._bias"]

                # Compute gamma
                F_weights = fisher_diag[f"layers.{layer_idx}._weights"]
                F_bias = fisher_diag[f"layers.{layer_idx}._bias"]

                weight_diff = W - theta_star[f"layers.{layer_idx}._weights"]
                bias_diff = b - theta_star[f"layers.{layer_idx}._bias"]

                gamma = (h @ (F_weights * weight_diff).T) + (F_bias * bias_diff)

                if layer_idx == n_layers - 1:
                    fisher_norm = torch.sqrt(
                        torch.sum(F_weights[task_mask, :] ** 2, dim=1)
                        + F_bias[task_mask] ** 2
                        + 1e-8
                    )
                    gamma = gamma[task_mask]
                else:
                    fisher_norm = torch.sqrt(
                        torch.sum(F_weights**2, dim=1) + F_bias**2 + 1e-8
                    )

                gamma = -beta * gamma / fisher_norm

                # Forward through layer
                z = F.linear(h, W, b)

                # Apply Softplus activation and gamma modulation (tanh + 1)
                if layer_idx < n_layers - 1:
                    r_ff = F.softplus(z)
                    h = r_ff * (torch.tanh(gamma) + 1)
                else:
                    # Final layer: apply modulation only to task-relevant outputs
                    h = z.clone()
                    modulation = torch.tanh(gamma) + 1
                    h[task_mask] = z[task_mask] * modulation

            # Only use Task A outputs for Fisher
            output = h[task_mask].unsqueeze(0)
            log_probs = F.log_softmax(output, dim=1)

            # Target should also be restricted to Task A classes
            y_task = y[task_mask]
            log_likelihood = (log_probs * y_task.unsqueeze(0)).sum()

            return log_likelihood

        # Create gradient function and vectorize over batch dimension
        grad_fn = grad(compute_loss_single_with_gamma)
        grad_fn_vmap = vmap(grad_fn, in_dims=(None, 0, 0))

        self.eval()
        total_samples = 0
        pbar = tqdm(total=len(dataloader), desc="Fisher (EFC+γ)", leave=True)

        for inputs, targets in dataloader:
            batch_size = inputs.size(0)

            # Compute per-sample gradients (parallelized across batch)
            per_sample_grads = grad_fn_vmap(params_dict, inputs, targets)

            # Flatten gradients for each sample
            grads_flat = torch.stack(
                [
                    torch.cat(
                        [per_sample_grads[n][i].flatten() for n in params_dict.keys()]
                    )
                    for i in range(batch_size)
                ]
            )  # Shape: [batch_size, param_count]

            # Accumulate outer products
            fisher += torch.einsum("bi,bj->ij", grads_flat, grads_flat)

            total_samples += batch_size
            pbar.update(1)

        pbar.close()
        fisher /= total_samples

        return fisher
