import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_basic_settings import *


ACTIVATION_REGISTRY = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}


class EvoNet(nn.Module):
    def __init__(self, config):
        super(EvoNet, self).__init__()

        self.n_neurons = config["n_neurons"]

        self.w1 = torch.randn(self.n_neurons[0][1], self.n_neurons[0][0]).to(
            DEVICE
        )  # F.linear uses (out_features, in_features) weights
        self.b1 = torch.randn(self.n_neurons[0][1]).to(DEVICE)
        self.w2 = torch.randn(self.n_neurons[1][1], self.n_neurons[1][0]).to(DEVICE)
        self.b2 = torch.randn(self.n_neurons[1][1]).to(DEVICE)

        self.n_w1 = self.w1.numel()
        self.n_b1 = self.b1.numel()
        self.n_w2 = self.w2.numel()
        self.n_b2 = self.b2.numel()

        activation_name = config["activation"]
        if activation_name not in ACTIVATION_REGISTRY:
            raise ValueError(f"Unsupported activation: {activation_name}")
        self.activation = ACTIVATION_REGISTRY[activation_name]

    def set_params(self, params):
        """
        Args:
            params: flattened parameter vector, shape (d,)
        """
        self.w1 = (
            params[: self.n_w1].view(self.n_neurons[0][1], -1).to(DEVICE).to(DTYPE)
        )
        self.b1 = (
            params[self.n_w1 : self.n_w1 + self.n_b1].view(-1).to(DEVICE).to(DTYPE)
        )
        self.w2 = (
            params[self.n_w1 + self.n_b1 : self.n_w1 + self.n_b1 + self.n_w2]
            .view(self.n_neurons[1][1], -1)
            .to(DEVICE)
            .to(DTYPE)
        )
        self.b2 = (
            params[self.n_w1 + self.n_b1 + self.n_w2 :].view(-1).to(DEVICE).to(DTYPE)
        )

    def get_dim(self):
        dim = int(
            self.n_neurons[0][0] * self.n_neurons[0][1]
            + self.n_neurons[0][1]
            + self.n_neurons[1][0] * self.n_neurons[1][1]
            + self.n_neurons[1][1]
        )

        return dim

    def forward(self, x):
        x = F.linear(x, weight=self.w1, bias=self.b1)
        x = self.activation(x)
        x = F.linear(x, weight=self.w2, bias=self.b2)

        return x


 
