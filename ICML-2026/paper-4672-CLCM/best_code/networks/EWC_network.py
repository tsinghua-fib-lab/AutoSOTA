import torch

from networks.network_interface import *
from networks.layers import BP_layer
from networks.activation_function import ReLU, Linear, Softplus


class EWC_network(Network, FisherInterface):
    def __init__(self, config, name="EWC_network"):
        Network.__init__(self, BP_layer, Softplus, Linear, config, name)
        FisherInterface.__init__(self)

        self.importance = config.importance_ewc

    def backward(self, y):
        loss = self.loss_fn(self.y_hat, y)
        if not self._first_task:
            loss += self.ewc_loss()
        loss.backward()

    def ewc_loss(self):
        """Compute EWC regularization loss"""
        loss = 0.0

        for n, p in self.named_parameters():
            if n in self._theta_star and n in self._fisher:
                loss += torch.sum(self._fisher[n] * (p - self._theta_star[n]) ** 2)
        return self.importance * loss
