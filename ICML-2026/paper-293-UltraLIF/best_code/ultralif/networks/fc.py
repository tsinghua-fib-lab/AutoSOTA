# -*- coding: utf-8 -*-
"""
Fully-connected spiking neural network architectures.

Classes:
    SNN:       Single hidden layer.
    DeepSNN:   Two hidden layers with optional residual connections and BatchNorm.
    TripleSNN: Three hidden layers with optional residual connections and BatchNorm.
"""

import torch
import torch.nn as nn

from ultralif.datasets.encoding import rate_encode


class SNN(nn.Module):
    """
    Single-layer Spiking Neural Network.

    Architecture:
        Input -> [rate encode] -> fc1 -> neuron -> fc2 -> output (averaged over T)

    For static inputs, pixel values are Poisson rate-encoded to spike trains.
    For neuromorphic inputs, the event frames are used directly.

    Args:
        neuron: Instantiated spiking neuron module.
        in_dim: Input feature dimension.
        hid_dim: Hidden layer width.
        out_dim: Number of output classes.
        timesteps: Number of time steps (used for rate encoding).
        neuromorphic: If True, input is already a spike train [B, T, features].

    Example:
        >>> from ultralif.neurons import UltraLIF
        >>> neuron = UltraLIF(dim=256)
        >>> model = SNN(neuron, in_dim=784, hid_dim=256, out_dim=10)
    """

    def __init__(
        self,
        neuron: nn.Module,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        timesteps: int = 30,
        neuromorphic: bool = False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.neuron = neuron
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.T = timesteps
        self.neuromorphic = neuromorphic
        self.last_spike_rate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        device, dtype = x.device, x.dtype
        if self.neuromorphic:
            if x.dim() > 3:
                x = x.view(batch, x.shape[1], -1)
            spikes_in = x
            T = spikes_in.shape[1]
        else:
            x = x.view(batch, -1)
            spikes_in = rate_encode(x, self.T, gain=0.5)
            T = self.T

        self.neuron.reset(batch, device)
        out_sum = torch.zeros(batch, self.fc2.out_features, device=device, dtype=dtype)
        spike_sum = 0.0

        for t in range(T):
            h = self.fc1(spikes_in[:, t, :])
            spike = self.neuron(h)
            spike_sum = spike_sum + spike.mean()
            out_sum = out_sum + self.fc2(spike)

        self.last_spike_rate = spike_sum / T
        return out_sum / T


class DeepSNN(nn.Module):
    """
    Two-layer Spiking Neural Network with optional BatchNorm and residual connections.

    Architecture:
        Input -> fc1 -> [BN] -> neuron1 -> fc2 -> [BN] -> neuron2 -> [+skip] -> fc3 -> output

    The residual skip connects neuron1's output to neuron2's output
    (hidden-to-hidden, not input-to-output).

    Args:
        neuron1: First hidden layer neuron.
        neuron2: Second hidden layer neuron.
        in_dim: Input dimension.
        hid_dim: Hidden layer width (same for both layers).
        out_dim: Number of output classes.
        timesteps: Time steps for rate encoding.
        neuromorphic: If True, use event frames directly.
        use_res: Add residual connection between hidden layers.
        use_bn: Apply BatchNorm1d before each neuron.

    Example:
        >>> from ultralif.neurons import UltraLIF
        >>> n1, n2 = UltraLIF(64), UltraLIF(64)
        >>> model = DeepSNN(n1, n2, in_dim=784, hid_dim=64, out_dim=10)
    """

    def __init__(
        self,
        neuron1: nn.Module,
        neuron2: nn.Module,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        timesteps: int = 30,
        neuromorphic: bool = False,
        use_res: bool = False,
        use_bn: bool = False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.neuron1 = neuron1
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.neuron2 = neuron2
        self.fc3 = nn.Linear(hid_dim, out_dim)
        self.T = timesteps
        self.neuromorphic = neuromorphic
        self.use_res = use_res
        self.use_bn = use_bn
        self.last_spike_rate = None
        if use_bn:
            self.bn1 = nn.BatchNorm1d(hid_dim)
            self.bn2 = nn.BatchNorm1d(hid_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        device, dtype = x.device, x.dtype
        if self.neuromorphic:
            if x.dim() > 3:
                x = x.view(batch, x.shape[1], -1)
            spikes_in = x
            T = spikes_in.shape[1]
        else:
            x = x.view(batch, -1)
            spikes_in = rate_encode(x, self.T, gain=0.5)
            T = self.T

        self.neuron1.reset(batch, device)
        self.neuron2.reset(batch, device)
        out_sum = torch.zeros(batch, self.fc3.out_features, device=device, dtype=dtype)
        spike_sum = 0.0

        for t in range(T):
            h1 = self.fc1(spikes_in[:, t, :])
            if self.use_bn:
                h1 = self.bn1(h1)
            spike1 = self.neuron1(h1)

            h2 = self.fc2(spike1)
            if self.use_bn:
                h2 = self.bn2(h2)
            spike2 = self.neuron2(h2)
            if self.use_res:
                spike2 = spike2 + spike1

            spike_sum = spike_sum + (spike1.mean() + spike2.mean()) / 2
            out_sum = out_sum + self.fc3(spike2)

        self.last_spike_rate = spike_sum / T
        return out_sum / T


class TripleSNN(nn.Module):
    """
    Three-layer Spiking Neural Network with optional BatchNorm and residual connections.

    Architecture:
        Input -> fc1 -> [BN] -> n1 -> fc2 -> [BN] -> n2 -> [+skip] -> fc3 -> [BN] -> n3 -> [+skip] -> fc4 -> output

    Args:
        neuron1: First hidden layer neuron.
        neuron2: Second hidden layer neuron.
        neuron3: Third hidden layer neuron.
        in_dim: Input dimension.
        hid_dim: Hidden layer width (same for all layers).
        out_dim: Number of output classes.
        timesteps: Time steps for rate encoding.
        neuromorphic: If True, use event frames directly.
        use_res: Add residual connections between hidden layers.
        use_bn: Apply BatchNorm1d before each neuron.

    Example:
        >>> from ultralif.neurons import UltraLIF
        >>> n1, n2, n3 = UltraLIF(64), UltraLIF(64), UltraLIF(64)
        >>> model = TripleSNN(n1, n2, n3, in_dim=784, hid_dim=64, out_dim=10)
    """

    def __init__(
        self,
        neuron1: nn.Module,
        neuron2: nn.Module,
        neuron3: nn.Module,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        timesteps: int = 30,
        neuromorphic: bool = False,
        use_res: bool = False,
        use_bn: bool = False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.neuron1 = neuron1
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.neuron2 = neuron2
        self.fc3 = nn.Linear(hid_dim, hid_dim)
        self.neuron3 = neuron3
        self.fc4 = nn.Linear(hid_dim, out_dim)
        self.T = timesteps
        self.neuromorphic = neuromorphic
        self.use_res = use_res
        self.use_bn = use_bn
        self.last_spike_rate = None
        if use_bn:
            self.bn1 = nn.BatchNorm1d(hid_dim)
            self.bn2 = nn.BatchNorm1d(hid_dim)
            self.bn3 = nn.BatchNorm1d(hid_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        device, dtype = x.device, x.dtype
        if self.neuromorphic:
            if x.dim() > 3:
                x = x.view(batch, x.shape[1], -1)
            spikes_in = x
            T = spikes_in.shape[1]
        else:
            x = x.view(batch, -1)
            spikes_in = rate_encode(x, self.T, gain=0.5)
            T = self.T

        self.neuron1.reset(batch, device)
        self.neuron2.reset(batch, device)
        self.neuron3.reset(batch, device)
        out_sum = torch.zeros(batch, self.fc4.out_features, device=device, dtype=dtype)
        spike_sum = 0.0

        for t in range(T):
            h1 = self.fc1(spikes_in[:, t, :])
            if self.use_bn:
                h1 = self.bn1(h1)
            spike1 = self.neuron1(h1)

            h2 = self.fc2(spike1)
            if self.use_bn:
                h2 = self.bn2(h2)
            spike2 = self.neuron2(h2)
            if self.use_res:
                spike2 = spike2 + spike1

            h3 = self.fc3(spike2)
            if self.use_bn:
                h3 = self.bn3(h3)
            spike3 = self.neuron3(h3)
            if self.use_res:
                spike3 = spike3 + spike2

            spike_sum = spike_sum + (spike1.mean() + spike2.mean() + spike3.mean()) / 3
            out_sum = out_sum + self.fc4(spike3)

        self.last_spike_rate = spike_sum / T
        return out_sum / T
