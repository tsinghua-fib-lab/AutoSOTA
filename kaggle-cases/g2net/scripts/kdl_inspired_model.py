"""New reconstruction inspired by KDL's public winner writeup.

The winner did not publish its complete neural model. This implementation is
therefore an experiment, not an exact reproduction or recovered winner model.
Separate multi-scale detector encoders feed an unpretrained ResNet34; synthetic
pretraining can additionally regress chirp mass, mass ratio and network SNR.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import butter, sosfreqz
from torch import nn
from torchvision.models import resnet34


class Highpass(nn.Module):
    def __init__(self):
        super().__init__()
        sos = butter(5, 20.43, 'highpass', fs=2048, output='sos')
        _, response = sosfreqz(sos, worN=np.fft.rfftfreq(8192, d=1/2048), fs=2048)
        self.register_buffer('response', torch.tensor(np.abs(response)**2, dtype=torch.float32))

    def forward(self, wave):
        # Squared Butterworth response represents forward/backward filtering.
        # Odd reflection reduces short-record boundary effects. Its boundary
        # handling differs from scipy.sosfiltfilt and is recorded in metadata.
        wave = wave.float()
        left = 2*wave[..., :1]-wave[..., 1:2049].flip(-1)
        right = 2*wave[..., -1:]-wave[..., -2049:-1].flip(-1)
        extended = torch.cat([left, wave, right], -1)
        return torch.fft.irfft(torch.fft.rfft(extended)*self.response, n=8192)[..., 2048:6144]


class ConvBlock(nn.Module):
    def __init__(self, incoming, outgoing, kernel):
        super().__init__()
        self.padding = ((kernel-1)//2, kernel//2)
        self.layers = nn.Sequential(nn.Conv1d(incoming, outgoing, kernel, bias=False),
                                    nn.BatchNorm1d(outgoing), nn.SiLU())

    def forward(self, x):
        return self.layers(F.pad(x, self.padding))


class MultiScale(nn.Module):
    def __init__(self, incoming, outgoing, base):
        super().__init__()
        self.branches = nn.ModuleList([ConvBlock(incoming, outgoing, k)
                                      for k in [base//4, base//2, base, base*2, base*4]])
        self.fuse = ConvBlock(outgoing*5+incoming, outgoing, 1)

    def forward(self, x):
        return self.fuse(torch.cat([layer(x) for layer in self.branches]+[x], 1))


class SplitMultiScaleResNet(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.width = width
        self.highpass = Highpass()
        self.encoders = nn.ModuleList([
            nn.Sequential(MultiScale(1, width, 64), nn.MaxPool1d(4),
                          MultiScale(width, width*2, 32), nn.MaxPool1d(4),
                          MultiScale(width*2, width*4, 16), nn.MaxPool1d(4))
            for _ in range(3)])
        self.fusion = resnet34(weights=None)
        self.fusion.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.fusion.maxpool = nn.Identity()
        self.fusion.fc = nn.Identity()
        self.detection = nn.Linear(512, 1)
        self.parameters_head = nn.Linear(512, 3)

    def forward(self, wave, auxiliary=False):
        filtered = self.highpass(wave)
        encoded = torch.stack([encoder(filtered[:, channel:channel+1])
                               for channel, encoder in enumerate(self.encoders)], 1)
        features = self.fusion(encoded)
        logit = self.detection(features).squeeze(-1)
        return (logit, self.parameters_head(features)) if auxiliary else logit
