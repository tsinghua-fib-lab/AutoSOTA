# -*- coding: utf-8 -*-
"""
Fully spiking ResNet-18 architecture.

Every ReLU is replaced with a spiking neuron. BatchNorm is always applied
(Conv -> BN -> Neuron pattern throughout), following He et al. 2016
CIFAR-variant ResNet-20/56.

Classes:
    SpikeBasicBlock:  ResNet BasicBlock with spiking neurons.
    SpikingResNet18:  Full 17-layer spiking ResNet18.
"""

import torch
import torch.nn as nn


class SpikeBasicBlock(nn.Module):
    """
    ResNet BasicBlock with spiking neurons replacing ReLU.

    Neurons are applied channel-wise (neuron_cls(C)) to keep state
    dimensions manageable and avoid large logsumexp tensors for Ultra variants.

    Args:
        neuron_cls: Neuron class (called with number of channels).
        in_ch: Input channel count.
        out_ch: Output channel count.
        in_h: Input spatial height.
        in_w: Input spatial width.
        stride: Conv stride (use 2 for downsampling blocks).
    """

    def __init__(self, neuron_cls, in_ch: int, out_ch: int, in_h: int, in_w: int, stride: int = 1):
        super().__init__()
        out_h = (in_h - 1) // stride + 1
        out_w = (in_w - 1) // stride + 1
        self.out_ch, self.out_h, self.out_w = out_ch, out_h, out_w

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.neuron1 = neuron_cls(out_ch)
        self.neuron2 = neuron_cls(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = None

    @staticmethod
    def _cw(h: torch.Tensor, neuron: nn.Module) -> torch.Tensor:
        """Apply neuron channel-wise: (B,C,H,W) -> neuron over C dim -> (B,C,H,W)."""
        B, C, H, W = h.shape
        flat = h.permute(0, 2, 3, 1).reshape(B * H * W, C)
        return neuron(flat).reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.bn1(self.conv1(x))
        s1 = self._cw(h1, self.neuron1)
        h2 = self.bn2(self.conv2(s1))
        residual = self.shortcut(x) if self.shortcut is not None else x
        return self._cw(h2 + residual, self.neuron2)

    def reset(self, batch: int, device):
        hw = self.out_h * self.out_w
        self.neuron1.reset(batch * hw, device)
        self.neuron2.reset(batch * hw, device)


class SpikingResNet18(nn.Module):
    """
    Fully spiking ResNet-18 with channel-wise spiking neurons.

    Follows the CIFAR-10 variant (3x3 stem conv, stride=1, no maxpool).
    Architecture: stem + 4 layer groups of 2 blocks each (64->128->256->512 channels).

    Args:
        neuron_cls: Neuron class (called with channel count).
        in_channels: Input channels (1 for MNIST, 3 for CIFAR-10, 2 for DVS).
        out_dim: Number of output classes.
        timesteps: Number of time steps.
        input_size: Spatial input size.

    Example:
        >>> from ultralif.neurons import UltraLIF
        >>> model = SpikingResNet18(UltraLIF, in_channels=3, out_dim=10)
    """

    def __init__(
        self,
        neuron_cls,
        in_channels: int,
        out_dim: int,
        timesteps: int = 1,
        input_size: int = 32,
    ):
        super().__init__()
        self.T = timesteps
        H = W = input_size

        self.stem_conv = nn.Conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        self.stem_neuron = neuron_cls(64)
        self._stem_hw = H * W

        H2, W2 = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        H3, W3 = (H2 - 1) // 2 + 1, (W2 - 1) // 2 + 1
        H4, W4 = (H3 - 1) // 2 + 1, (W3 - 1) // 2 + 1

        self.blocks = nn.ModuleList([
            SpikeBasicBlock(neuron_cls, 64, 64, H, W, stride=1),
            SpikeBasicBlock(neuron_cls, 64, 64, H, W, stride=1),
            SpikeBasicBlock(neuron_cls, 64, 128, H, W, stride=2),
            SpikeBasicBlock(neuron_cls, 128, 128, H2, W2, stride=1),
            SpikeBasicBlock(neuron_cls, 128, 256, H2, W2, stride=2),
            SpikeBasicBlock(neuron_cls, 256, 256, H3, W3, stride=1),
            SpikeBasicBlock(neuron_cls, 256, 512, H3, W3, stride=2),
            SpikeBasicBlock(neuron_cls, 512, 512, H4, W4, stride=1),
        ])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, out_dim)
        self.last_spike_rate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device = x.device

        if x.dim() == 4:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1)
        T = x.shape[1]

        self.stem_neuron.reset(B * self._stem_hw, device)
        for blk in self.blocks:
            blk.reset(B, device)

        out_sum = torch.zeros(B, self.fc.out_features, device=device, dtype=x.dtype)
        spike_sum = 0.0
        n_layers = 1 + len(self.blocks)

        for t in range(T):
            xt = x[:, t]
            h = self.stem_bn(self.stem_conv(xt))
            B2, C, H, W = h.shape
            feat = (
                self.stem_neuron(h.permute(0, 2, 3, 1).reshape(B2 * H * W, C))
                .reshape(B2, H, W, C)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            total = feat.mean()

            for blk in self.blocks:
                feat = blk(feat)
                total = total + feat.mean()

            spike_sum = spike_sum + total / n_layers
            out_sum = out_sum + self.fc(self.avgpool(feat).view(B, -1))

        self.last_spike_rate = spike_sum / T
        return out_sum / T
