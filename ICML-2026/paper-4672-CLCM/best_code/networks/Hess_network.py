import torch
from networks.network_interface import *
from networks.layers import BP_layer
from networks.activation_function import ReLU, Linear, Softplus
from tqdm import tqdm
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad


class Hess_network(Network, FisherInterface):
    def __init__(self, config, name="Hess_network"):
        Network.__init__(self, BP_layer, Softplus, Linear, config, name)
        FisherInterface.__init__(self)
        self.importance = config.importance_ewc
        self._theta_star = None
        self._hessian = None  # Full p x p Hessian

    def ehc_loss(self):
        if self._first_task or self._hessian is None:
            return torch.tensor(0.0, device=self.device)

        v = torch.cat(
            [
                (p - self._theta_star[n]).flatten()
                for n, p in self.named_parameters()
                if p.requires_grad
            ]
        )

        # Detach Hessian and compute manually to avoid autograd overhead
        with torch.no_grad():
            Hv = self._hessian @ v

        # Compute the quadratic form
        quad = 0.5 * (v @ Hv)

        return self.importance * quad

    def backward(self, y):
        loss = self.loss_fn(self.y_hat, y)
        if not self._first_task:
            loss += self.ehc_loss()
        loss.backward()

    def complete_task(self, dataloader):
        # Use your _calculate_hessian from FisherInterface
        current_hessian = self._calculate_full_fisher(dataloader).detach()
        self._theta_star = {
            n: p.data.clone() for n, p in self.named_parameters() if p.requires_grad
        }

        if self._first_task:
            self._hessian = current_hessian
            self._first_task = False
        else:
            for n in self._hessian:
                self._hessian[n] += current_hessian[n]
