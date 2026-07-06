# -*- coding: utf-8 -*-
"""
Convolutional spiking neural network architectures.

Classes:
    ConvSNN:     2-layer Conv + spiking neurons.
    DeepConvSNN: 4-layer Conv + spiking neurons.
"""

import torch
import torch.nn as nn


class ConvSNN(nn.Module):
    """
    2-layer Convolutional Spiking Neural Network.

    Architecture (32x32 input):
        Conv(in,32,3x3) -> [BN] -> Pool(2) -> neuron1
        -> Conv(32,64,3x3) -> [BN] -> Pool(2) -> neuron2
        -> FC -> output

    Neuron dimensions are sized to the post-pool feature map (flattened).

    Args:
        neuron_cls: Neuron class (called with dim as sole argument).
        in_channels: Number of input channels (1 for MNIST, 3 for CIFAR-10).
        out_dim: Number of output classes.
        timesteps: Time steps; static images are repeated T times.
        input_size: Spatial input size (28 for MNIST, 32 for CIFAR-10).
        stateless: If True, reset membrane state each timestep (ablation).
        use_bn: Apply BatchNorm2d before each neuron.

    Example:
        >>> from ultralif.neurons import UltraLIF
        >>> model = ConvSNN(UltraLIF, in_channels=1, out_dim=10, input_size=28)
    """

    def __init__(
        self,
        neuron_cls,
        in_channels: int,
        out_dim: int,
        timesteps: int = 30,
        input_size: int = 32,
        stateless: bool = False,
        use_bn: bool = False,
    ):
        super().__init__()
        self.T = timesteps
        self.input_size = input_size
        self.stateless = stateless
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool2d(2)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)

        if input_size == 32:
            fc_in = 64 * 8 * 8
        elif input_size == 28:
            fc_in = 64 * 7 * 7
        else:
            fc_in = 64 * (input_size // 4) ** 2

        self.fc = nn.Linear(fc_in, out_dim)
        self.neuron1 = neuron_cls(32 * (input_size // 2) ** 2)
        self.neuron2 = neuron_cls(fc_in)
        self.last_spike_rate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        device = x.device
        if x.dim() == 4:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1)
        T = x.shape[1]

        self.neuron1.reset(batch, device)
        self.neuron2.reset(batch, device)
        out_sum = torch.zeros(batch, self.fc.out_features, device=device, dtype=x.dtype)
        spike_sum = 0.0

        for t in range(T):
            xt = x[:, t]
            if self.stateless:
                self.neuron1.reset(batch, device)
                self.neuron2.reset(batch, device)

            h1 = self.conv1(xt)
            if self.use_bn:
                h1 = self.bn1(h1)
            h1 = self.pool1(h1)
            h1_flat = h1.view(batch, -1)
            spike1 = self.neuron1(h1_flat)
            spike1 = spike1.view(batch, 32, self.input_size // 2, self.input_size // 2)

            h2 = self.conv2(spike1)
            if self.use_bn:
                h2 = self.bn2(h2)
            h2 = self.pool2(h2)
            h2_flat = h2.view(batch, -1)
            spike2 = self.neuron2(h2_flat)

            spike_sum = spike_sum + (spike1.mean() + spike2.mean()) / 2
            out_sum = out_sum + self.fc(spike2)

        self.last_spike_rate = spike_sum / T
        return out_sum / T


class DeepConvSNN(nn.Module):
    """
    4-layer Convolutional Spiking Neural Network.

    Architecture (32x32 input):
        Conv(in,32,3x3) -> [BN] -> Pool(2) -> neuron1
        Conv(32,64,3x3) -> [BN] -> Pool(2) -> neuron2
        Conv(64,128,3x3) -> [BN] -> Pool(2) -> neuron3
        Conv(128,256,3x3) -> [BN] -> Pool(2) -> neuron4
        -> FC(1024, n_classes)

    Args:
        neuron_cls: Neuron class.
        in_channels: Number of input channels.
        out_dim: Number of output classes.
        timesteps: Time steps.
        input_size: Spatial input size (must be divisible by 16).
        use_bn: Apply BatchNorm2d before each neuron.
    """

    def __init__(
        self,
        neuron_cls,
        in_channels: int,
        out_dim: int,
        timesteps: int = 30,
        input_size: int = 32,
        use_bn: bool = False,
    ):
        super().__init__()
        self.T = timesteps
        self.input_size = input_size
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.AvgPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.AvgPool2d(2)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(256)

        s1 = input_size // 2
        s2 = input_size // 4
        s3 = input_size // 8
        s4 = input_size // 16
        self.neuron1 = neuron_cls(32 * s1 * s1)
        self.neuron2 = neuron_cls(64 * s2 * s2)
        self.neuron3 = neuron_cls(128 * s3 * s3)
        self.neuron4 = neuron_cls(256 * s4 * s4)
        self.fc = nn.Linear(256 * s4 * s4, out_dim)
        self.last_spike_rate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        device = x.device
        if x.dim() == 4:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1)
        T = x.shape[1]
        s = self.input_size
        s1, s2, s3, s4 = s // 2, s // 4, s // 8, s // 16

        self.neuron1.reset(batch, device)
        self.neuron2.reset(batch, device)
        self.neuron3.reset(batch, device)
        self.neuron4.reset(batch, device)
        out_sum = torch.zeros(batch, self.fc.out_features, device=device, dtype=x.dtype)
        spike_sum = 0.0

        for t in range(T):
            xt = x[:, t]

            h1 = self.conv1(xt)
            if self.use_bn:
                h1 = self.bn1(h1)
            h1 = self.pool1(h1)
            spike1 = self.neuron1(h1.view(batch, -1))
            spike1 = spike1.view(batch, 32, s1, s1)

            h2 = self.conv2(spike1)
            if self.use_bn:
                h2 = self.bn2(h2)
            h2 = self.pool2(h2)
            spike2 = self.neuron2(h2.view(batch, -1))
            spike2 = spike2.view(batch, 64, s2, s2)

            h3 = self.conv3(spike2)
            if self.use_bn:
                h3 = self.bn3(h3)
            h3 = self.pool3(h3)
            spike3 = self.neuron3(h3.view(batch, -1))
            spike3 = spike3.view(batch, 128, s3, s3)

            h4 = self.conv4(spike3)
            if self.use_bn:
                h4 = self.bn4(h4)
            h4 = self.pool4(h4)
            spike4 = self.neuron4(h4.view(batch, -1))

            spike_sum = spike_sum + (spike1.mean() + spike2.mean() + spike3.mean() + spike4.mean()) / 4
            out_sum = out_sum + self.fc(spike4)

        self.last_spike_rate = spike_sum / T
        return out_sum / T
