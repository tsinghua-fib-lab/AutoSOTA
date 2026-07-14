import torch
import torch.nn as nn


class Layer(nn.Module):
    def __init__(self, in_features, out_features, activation_fn, name):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.name = name

        self._create_init_layer()

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def shape(self):
        return self._weights.shape

    @property
    def activation_derivative(self):
        return self.activation_fn.derivative

    def _create_init_layer(self):
        self.feedforward = nn.Sequential(
            nn.Linear(self.in_features, self.out_features), self.activation_fn
        )

        nn.init.kaiming_normal_(self.feedforward[0].weight)
        self._weights = self.feedforward[0].weight
        self._bias = self.feedforward[0].bias

    def forward(self, x):
        self.r_prev = x
        a = torch.matmul(x, self.weights.t())
        a += self.bias.unsqueeze(0).expand_as(a)
        self.r = self.activation_fn(a)
        self.r_ff = self.r
        self.v_ff = a

        return self.r
