import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class ChannelPool(nn.Module):

    def __init__(self, dim=1, concat=True):
        super().__init__()
        self.dim = dim
        self.concat = concat

    def forward(self, x):
        max_out = torch.max(x, self.dim)[0].unsqueeze(1)
        avg_out = torch.mean(x, self.dim).unsqueeze(1)
        if self.concat:
            return torch.cat((max_out, avg_out), dim=self.dim)
        else:
            return max_out, avg_out


class AdaptiveConcatPool3d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1, 1)
        self.ap = nn.AdaptiveAvgPool3d(sz)
        self.mp = nn.AdaptiveMaxPool3d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 
            (x.size(-2), x.size(-1))).pow(1./self.p)

    def __repr__(self):
        return f'GeM(p={self.p}, eps={self.eps})'


class AdaptiveGeM(nn.Module):
    def __init__(self, size=(1, 1), p=3, eps=1e-6):
        super().__init__()
        self.size = size
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), self.size).pow(1./self.p)

    def __repr__(self):
        return f'AdaptiveGeM(size={self.size}, p={self.p}, eps={self.eps})'
