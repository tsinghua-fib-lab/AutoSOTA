import torch

from networks.network_interface import *
from networks.layers import DFC_layer
from networks.activation_function import *

check_nan = lambda x: torch.isnan(x).any().item()


class DFC_network(Network, JacobianInterface):
    def __init__(self, config, name="DFC_network") -> None:
        Network.__init__(self, DFC_layer, ReLU, Linear, config, name)
        JacobianInterface.__init__(self, config)

    @torch.no_grad()
    def _non_dynamical_inversion(self):
        J, _ = self._calculate_full_jacobian()
        J_T = J.transpose(1, 2)

        error = self._compute_error(self.y_hat, self.targets)
        error = error.unsqueeze(2)

        u = torch.linalg.solve(
            torch.bmm(J, J_T) + self.alpha * torch.eye(J.shape[1]), error
        )

        delta_v = torch.bmm(J_T, u).squeeze(-1)
        delta_vs = torch.split(delta_v, self.layer_sizes, dim=1)

        rs = [self.input]

        for i, layer in enumerate(self.layers):
            v_ff = torch.bmm(rs[i], layer.weights.t())
            v_ff += layer.bias.unsqueeze(0).expand_as(v_ff)
            v = v_ff + delta_vs[i]

            r_ff = layer.activation_fn(v_ff)
            r = layer.activation_fn(v)
            rs.append(r)

            layer.v_ff = v_ff
            layer.v = v
            layer.delta_v = delta_vs[i]

            layer.r = r
            layer.r_ff = r_ff
            layer.r_prev = rs[i]

    @torch.no_grad()
    def _dynamical_inversion(self):
        # Setup
        layer_out_dims = [layer.weights.shape[0] for layer in self.layers]

        v_fb_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        v_ff_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        v_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        r_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        u_current = torch.zeros((self.bzs, self.output_size))
        u_int_current = torch.zeros((self.bzs, self.output_size))

        for i, layer in enumerate(self.layers):
            v_ff_current[i] = layer.v_ff
            v_current[i] = layer.v_ff
            r_current[i] = layer.r

        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool)

        # Simulate tmax timesteps
        for _ in range(self.tmax - 1):
            # Stop if converged
            if converged_mask.all():
                break

            error = self._compute_error(
                r_current[-1][:, self.task_masks[self.task_id]], self.targets
            )

            # Proportional and integral (PI) control.
            u_int_next = u_int_current + self.dt * (error - self.alpha * u_current)
            u_next = u_int_next + self.k_p * error

            # Compute convergence check
            converged_mask |= torch.norm(u_next - u_current, dim=1) < self.eps

            _, Js = self._calculate_full_jacobian()

            # Iterate over layers with control signal
            for i, layer in enumerate(self.layers):
                r_prev = r_current[i - 1] if i != 0 else self.input

                # Basal and apical
                v_ff_current[i] = r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0)
                v_fb_current[i] = torch.bmm(
                    Js[i][:, self.task_masks[self.task_id], :].transpose(1, 2),
                    u_next.unsqueeze(2),
                ).squeeze(2)

                # Soma with apical
                tau = self.dt / self.time_constant_ratio
                if i == len(self.layers) - 1:
                    task_slice = self.task_masks[self.task_id]
                    v_update = torch.zeros_like(v_current[i])
                    v_update[:, task_slice] = tau * (
                        v_fb_current[i][:, task_slice]
                        + v_ff_current[i][:, task_slice]
                        - v_current[i][:, task_slice]
                    )
                    v_current[i] += v_update
                else:
                    v_current[i] += tau * (
                        v_fb_current[i] + v_ff_current[i] - v_current[i]
                    )

                r_current[i] = layer.activation_fn(v_current[i])

                layer.v_ff = v_current[i]
                layer.r = r_current[i]

            u_int_current = u_int_next
            u_current = u_next

        # Steady-state values per layer
        rs = [self.input]

        for i, layer in enumerate(self.layers):
            layer.r = r_current[i]
            layer.r_ff = layer.activation_fn(v_ff_current[i])
            layer.r_prev = rs[i]
            rs.append(r_current[i])


class DFC_Mult_network(Network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="DFC_Mult_network"):
        Network.__init__(self, DFC_layer, ReLU, Linear, config, name)
        JacobianInterface.__init__(self, config)

    @torch.no_grad()
    def _dynamical_inversion(self):
        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool)
        u_current = torch.zeros((self.bzs, self.output_size))

        for _ in range(1, self.tmax):
            error = self._compute_error(self.layers[-1].r, self.targets)

            # Proportional control
            u_next = self.k_p * error
            psis = self._calculate_psis(u_next)

            # Forward pass
            for i, layer in enumerate(self.layers):
                layer.r_prev = self.layers[i - 1].r if i != 0 else self.input
                layer.v_ff = layer.r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(
                    0
                )
                layer.r_ff = layer.activation_fn(layer.v_ff)

                psi = psis[i]
                e_psi = torch.tanh(psi) + 1

                layer.r = layer.r + self.dt / self.time_constant_ratio * (
                    e_psi * layer.r_ff - layer.r
                )

            # Compute convergence check
            converged_mask |= torch.norm(u_next - u_current, dim=1) < self.eps
            if converged_mask.all():
                break
            u_current = u_next
