import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.func import functional_call, vmap, grad

from networks.layers import *



class Network(nn.Module):
    def __init__(self, layer_class, activation_fn, out_activation_fn, config, name):
        super().__init__()
        self.create_network(layer_class, activation_fn, out_activation_fn, config)
        self.loss_fn = (
            nn.MSELoss() if config.loss_fn == "mse" else nn.CrossEntropyLoss()
        )
        self.loss_fn_name = config.loss_fn
        self.device = config.device
        self.lr = config.lr
        self.name = name
        self.setting = config.setting


        # Setup task masks for continual learning
        self.num_tasks = getattr(config, "num_tasks", 5)
        self.classes_per_task = getattr(config, "classes_per_task", 2)
        self._setup_task_masks()

    def _setup_task_masks(self):
        """Setup masks based on continual learning setting."""
        self.task_masks = {}
        self.task_masks_complement = {}

        setting_lower = self.setting.lower()
        for task_id in range(self.num_tasks):
            if "taskil" in setting_lower:
                # Task IL: only current task's outputs
                start_idx = task_id * self.classes_per_task
                end_idx = (task_id + 1) * self.classes_per_task
                self.task_masks[task_id] = slice(start_idx, end_idx)


                # Complement: all other tasks
                complement_indices = list(range(0, start_idx)) + list(
                    range(end_idx, self.num_tasks * self.classes_per_task)
                )
                self.task_masks_complement[task_id] = complement_indices

            elif "classil" in setting_lower:
                # Class IL: all classes up to current task
                end_idx = (task_id + 1) * self.classes_per_task
                self.task_masks[task_id] = slice(0, end_idx)


                # Complement: future classes only
                complement_indices = list(
                    range(end_idx, self.num_tasks * self.classes_per_task)
                )
                self.task_masks_complement[task_id] = complement_indices


            else:  # domainIL
                # Domain IL: all outputs (no masking)
                self.task_masks[task_id] = slice(None)
                self.task_masks_complement[task_id] = []

    @property
    def layer_sizes(self):
        return [layer.out_features for layer in self.layers]

    @property
    def activations(self):
        return [layer.r for layer in self.layers]

    @property
    def linear_activations(self):
        return [layer.v_ff for layer in self.layers]

    def forward(self, x):
        self.input = x
        self.bzs = x.shape[0]
        for layer in self.layers:
            x = layer(x)

        x = x[:, self.task_masks[self.task_id]]

        self.y_hat = x
        return x

    def create_network(self, layer_class, activation_fn, out_activation_fn, config):
        _layers = config.layers
        self.layers = nn.ModuleList()
        for i in range(len(_layers) - 2):
            self.layers.append(
                layer_class(
                    _layers[i],
                    _layers[i + 1],
                    activation_fn=activation_fn(),
                )
            )
        self.layers.append(
            layer_class(
                _layers[-2],
                _layers[-1],
                activation_fn=out_activation_fn(),
            )
        )

    def calculate_loss(self, y_hat, y):
        self.loss = self.loss_fn(y_hat, y)
        return self.loss

    def _calculate_full_fisher(self, dataloader):
        """Compute full Fisher Information Matrix"""
        # Get flattened parameters
        params_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
        buffers = {n: b for n, b in self.named_buffers()}

        # Count total parameters
        param_count = sum(p.numel() for p in params_dict.values())
        fisher = torch.zeros(
            param_count, param_count, dtype=torch.float32, device=self.device
        )

        def compute_loss_single(params, buffers, x, y):
            """Compute log likelihood for a single sample"""
            output = functional_call(self, (params, buffers), (x.unsqueeze(0),))
            log_probs = F.log_softmax(output, dim=1)
            log_likelihood = (log_probs * y.unsqueeze(0)).sum()
            return log_likelihood

        # Create gradient function and vectorize it
        grad_fn = grad(compute_loss_single)
        grad_fn_vmap = vmap(grad_fn, in_dims=(None, None, 0, 0))

        self.eval()
        total_samples = 0
        pbar = tqdm(total=len(dataloader), desc="Hessian", leave=True)

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size = inputs.size(0)

            # Compute per-sample gradients (parallelized across batch)
            per_sample_grads = grad_fn_vmap(params_dict, buffers, inputs, targets)

            # Flatten gradients for each sample
            # per_sample_grads is a dict with shape [batch_size, ...] for each param
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

        # Normalize
        fisher /= total_samples

        return fisher



class JacobianInterface:
    def __init__(self, config):
        if config.mode == "ndi":
            self._inversion = self._non_dynamical_inversion
        else:
            self._inversion = self._dynamical_inversion
            self.dt = float(config.dt_di)
            self.apical_time_constant = self.dt
            self.time_constant_ratio = config.time_constant_ratio
            self.k_p = config.k_p
            self.tmax = config.tmax_di
            self.eps = float(config.eps)

            assert self.k_p > 0
            assert self.eps > 0

        if config.loss_fn == "mse":
            self._compute_error = self._compute_error_mse
            self._set_targets = self._set_targets_mse
        else:
            self._compute_error = self._compute_error_ce
            self._set_targets = self._set_targets_ce
            self._softmax = nn.Softmax(dim=1)

        self.tau = config.tau
        self.target_lr = float(config.target_lr)
        self.alpha = float(config.alpha_di)
        self.alpha_I = float(config.alpha_I)

        assert self.alpha > 0

    def backward(self, y):
        self._set_targets(y)
        self._inversion()

        for layer in self.layers:
            layer.backward()

    def _compute_error_mse(self, y_hat, y):
        return y - y_hat

    def _compute_error_ce(self, y_hat, y):
        return y - self._softmax(y_hat)

    def _set_targets_mse(self, y):
        """MSE loss solution"""
        self.targets = (1 - 2 * self.target_lr) * self.y_hat + 2 * self.target_lr * y
        self.output_size = self.targets.shape[1]

    def _set_targets_ce(self, y):
        """CE loss solution"""
        self.targets = self._softmax(self.y_hat) - self.target_lr * (
            self._softmax(self.y_hat) - y
        )
        self.output_size = self.targets.shape[1]

    def _calculate_layerwise_jacobians(self):
        """
        Compute the Jacobian J_{i,i-1} for each layer using DFC_layer's method.
        """
        Js = []
        for layer in self.layers:
            J = layer.compute_layerwise_jacobian()
            Js.append(J)

        Js[-1] = Js[-1][:, self.task_masks[self.task_id], :]

        return Js

    @torch.enable_grad()
    def _calculate_jeff_and_gammaeff(self):
        # Recompute forward with requires_grad and retain_grad on modulatable activations
        x = self.input.detach().requires_grad_(True)
        activations_with_grad = []


        for layer in self.layers:
            x = layer.forward(x)
            x.retain_grad()
            activations_with_grad.append(x)


        y = activations_with_grad[-1]
        y = y[:, self.task_masks[self.task_id]]

        out_dim = y.shape[1]

        J_eff = torch.zeros(self.bzs, out_dim, out_dim, device=self.device)
        gamma_eff = torch.zeros(self.bzs, out_dim, device=self.device)

        # Collect rows of cumulative Jacobians for each layer
        ji_rows_per_layer = [[] for _ in activations_with_grad]
        for k in range(out_dim):
            # Zero previous grads
            for r_i in activations_with_grad:
                if r_i.grad is not None:
                    r_i.grad.zero_()

            grad_outputs = torch.zeros_like(y)
            grad_outputs[:, k] = 1.0

            y.backward(gradient=grad_outputs, retain_graph=True)

            # Collect the k-th row for each layer
            for l, r_i in enumerate(activations_with_grad):
                if l == len(activations_with_grad) - 1:
                    grad_flat = (
                        r_i.grad[:, self.task_masks[self.task_id]]
                        .view(self.bzs, -1)
                        .clone()
                    )
                else:
                    grad_flat = r_i.grad.view(self.bzs, -1).clone()
                ji_rows_per_layer[l].append(grad_flat)

        # Now process per layer
        J_list = []
        gamma_list = []
        for l in range(len(activations_with_grad)):
            # Stack rows to form Ji_flat (bzs, out_dim, flat_dim)
            Ji_flat = torch.stack(ji_rows_per_layer[l], dim=1)
            J_list.append(Ji_flat)

            r_ff_flat = activations_with_grad[l].detach().view(self.bzs, -1)
            if l == len(self.layers) - 1:
                r_ff_flat = r_ff_flat[:, self.task_masks[self.task_id]]

            gamma_i = self._compute_gamma(self.layers[l], l)
            gamma_list.append(gamma_i)
            gamma_flat = gamma_i.view(self.bzs, -1) if not self._first_task else 0.0

            # Compute contribution to J_eff: Ji @ diag(r) @ Ji^T = (Ji_flat * r_ff_flat.unsqueeze(1)) @ Ji_flat.transpose(1, 2)
            J_eff += torch.bmm(
                Ji_flat * r_ff_flat.unsqueeze(1), Ji_flat.transpose(1, 2)
            )

            # Compute contribution to gamma_eff: Ji @ (gamma ⊙ r)
            gamma_r_flat = (gamma_flat * r_ff_flat).unsqueeze(-1)
            gamma_eff += torch.bmm(Ji_flat, gamma_r_flat).squeeze(-1)

        return J_eff, gamma_eff, J_list, gamma_list

    def _calculate_full_jacobian(self):
        Js = [None] * len(self.layers)
        output_size = self.layer_sizes[-1]

        activations_derivatives = [
            layer.activation_derivative(layer.v_ff) for layer in self.layers
        ]

        # Last layer
        Js[-1] = activations_derivatives[-1].view(self.bzs, output_size, 1) * torch.eye(output_size, device=self.device)

        # Rest of the layers
        for i in range(len(self.layers) - 2, -1, -1):
            Js[i] = activations_derivatives[i].unsqueeze(1) * torch.matmul(
                Js[i + 1], self.layers[i + 1].weights
            )

        return torch.cat(Js, dim=2), Js


    @torch.no_grad()
    def _calculate_psis(self, u):
        L = len(self.layers)
        psi_list = [None] * L

        # Derivatives per layer
        activations_derivatives = [
            layer.activation_derivative(layer.v_ff) for layer in self.layers
        ]

        # Last layer - might have to expand u to full size
        full_u = torch.zeros_like(activations_derivatives[-1])
        full_u[:, self.task_masks[self.task_id]] = u
        psi = full_u * activations_derivatives[-1]
        psi_list[-1] = psi


        # Backward from second-to-last to first
        for i in range(L - 2, -1, -1):
            psi = (psi @ self.layers[i + 1].weights) * activations_derivatives[i]
            psi_list[i] = psi


        return psi_list


class FisherInterface:
    def __init__(self):
        self._fisher = {}  # Accumulated Fisher matrix
        self._theta_star = {}  # Latest parameter optima (theta_T^*)
        self._first_task = True


    def _calculate_fisher(self, dataloader):
        """Compute diagonal Fisher Information Matrix"""
        fisher = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p)

        # Get parameters as a dictionary for functional_call
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        buffers = {n: b for n, b in self.named_buffers()}


        def compute_loss_single(params, buffers, x, y):
            """Compute log likelihood for a single sample"""
            output = functional_call(self, (params, buffers), (x.unsqueeze(0),))
            log_probs = F.log_softmax(output, dim=1)
            log_likelihood = (log_probs * y.unsqueeze(0)).sum()
            return log_likelihood


        # Create gradient function and vectorize it
        grad_fn = grad(compute_loss_single)
        grad_fn_vmap = vmap(grad_fn, in_dims=(None, None, 0, 0))


        self.eval()
        total_samples = 0
        pbar = tqdm(total=len(dataloader), desc="Fisher", leave=True)


        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size = inputs.size(0)


            # Compute per-sample gradients (parallelized across batch)
            per_sample_grads = grad_fn_vmap(params, buffers, inputs, targets)


            # Accumulate squared gradients
            for n in fisher.keys():
                fisher[n].data += (per_sample_grads[n] ** 2).sum(dim=0)


            total_samples += batch_size
            pbar.update(1)


        pbar.close()


        # Normalize
        for n in fisher.keys():
            fisher[n] /= total_samples


        return fisher

    def start_task(self, dataloader):
        pass

    def complete_task(self, dataloader):
        """Update Fisher with rescaling to match magnitudes across tasks."""
        current_fisher = self._calculate_fisher(dataloader)
        self._theta_star = {n: p.data.clone() for n, p in self.named_parameters() if p.requires_grad}
        
        if self._first_task:
            self._fisher = current_fisher
            self._first_task = False
        else:
            old_norm = sum(torch.norm(f).item()**2 for f in self._fisher.values())**0.5
            new_norm = sum(torch.norm(f).item()**2 for f in current_fisher.values())**0.5
            scale = old_norm / (new_norm + 1e-8)
            
            for n in self._fisher:
                self._fisher[n] += scale * current_fisher[n]


    def _compute_gamma(self, layer, i):
        if self._first_task:
            return 0.0

        F_weights = self._fisher[f"layers.{i}._weights"]
        F_bias = self._fisher[f"layers.{i}._bias"]

        weight_diff = layer._weights - self._theta_star[f"layers.{i}._weights"]
        bias_diff = layer._bias - self._theta_star[f"layers.{i}._bias"]

        gamma = (layer.r_prev @ (F_weights * weight_diff).T) + (F_bias * bias_diff)

        if i == len(self.layers) - 1:
            fisher_norm = (
                torch.sum(F_weights[self.task_masks[self.task_id], :] ** 2, dim=1)
                + (F_bias[self.task_masks[self.task_id]] ** 2)
                + 1e-8
            )
            gamma = gamma[:, self.task_masks[self.task_id]]
        else:
            fisher_norm = torch.sum(F_weights**2, dim=1) + (F_bias**2) + 1e-8

        fisher_norm = torch.sqrt(fisher_norm)
        gamma = -self.beta * gamma / fisher_norm

        return gamma
