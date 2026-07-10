import pywt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_wavelets.dwt.lowlevel as lowlevel

class DWT1D(nn.Module):
    def __init__(self, J=1, wave='db1', learnable=True):
        super().__init__()
        self.J = J
        self.mode = 'zero'

        hac = pywt.Wavelet(wave).dec_lo
        self.N = len(hac)
        hac = torch.tensor(hac)
        self.hac = nn.Parameter(hac, requires_grad=learnable)

    def forward(self, x, Inverse=0):
        hac =  self.hac
        hdc, miu = hac.flip(-1), torch.ones_like(hac, device=hac.device)
        miu[::2] = -1
        hdc *= miu

        if Inverse == 0:
            x0, highs = x, []
            for j in range(self.J):
                x0, x1 = self.AFB1D(x0, hac.view(1, 1, -1), hdc.view(1, 1, -1))
                highs.append(x1)
            return x0, highs
        else:
            x0, highs = x
            for x1 in highs[::-1]:
                if x1 is None:
                    x1 = torch.zeros_like(x0)
                if x0.shape[-1] > x1.shape[-1]:
                    x0 = x0[..., :-1]
                x0 = self.SFB1D(x0, x1, hac.view(1, 1, -1), hdc.view(1, 1, -1))
            return x0

    def AFB1D(self, x, h0, h1):
        x = x[:, :, None, :]
        h0 = h0[:, :, None, :]
        h1 = h1[:, :, None, :]

        lohi = afb1d(x, h0, h1, mode=self.mode, dim=3)
        x0 = lohi[:, ::2, 0].contiguous()
        x1 = lohi[:, 1::2, 0].contiguous()
        return x0, x1

    def SFB1D(self, low, high, g0, g1):
        low = low[:, :, None, :]
        high = high[:, :, None, :]
        g0 = g0[:, :, None, :]
        g1 = g1[:, :, None, :]

        return sfb1d(low, high, g0, g1, mode=self.mode, dim=3)[:, :, 0]

def afb1d(x, h0, h1, mode='zero', dim=-1):
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    N = x.shape[d]

    L = h0.numel()
    shape = [1,1,1,1]
    shape[d] = L

    h = torch.cat([h0, h1] * x.shape[1], dim=0)
    outsize = pywt.dwt_coeff_len(N, L, mode=mode)

    p = 2 * (outsize - 1) - N + L
    if p % 2 == 1:
        pad = (0, 0, 0, 1) if d == 2 else (0, 1, 0, 0)
        x = F.pad(x, pad)
    pad = (p//2, 0) if d == 2 else (0, p//2)
    lohi = F.conv2d(x, h, padding=pad, stride=s, groups=x.shape[1])

    return lohi

def sfb1d(lo, hi, g0, g1, mode='zero', dim=-1):
    d = dim % 4
    L = g0.numel()
    shape = [1,1,1,1]
    shape[d] = L

    s = (2, 1) if d == 2 else (1,2)
    g0 = torch.cat([g0]*lo.shape[1],dim=0)
    g1 = torch.cat([g1]*lo.shape[1],dim=0)
    pad = (L-2, 0) if d == 2 else (0, L-2)
    y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=lo.shape[1]) + \
        F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=lo.shape[1])

    return y

def roll(x, n, dim, make_even=False):
    if n < 0:
        n = x.shape[dim] + n

    if make_even and x.shape[dim] % 2 == 1:
        end = 1
    else:
        end = 0

    if dim == 0:
        return torch.cat((x[-n:], x[:-n+end]), dim=0)
    elif dim == 1:
        return torch.cat((x[:,-n:], x[:,:-n+end]), dim=1)
    elif dim == 2 or dim == -2:
        return torch.cat((x[:,:,-n:], x[:,:,:-n+end]), dim=2)
    elif dim == 3 or dim == -1:
        return torch.cat((x[:,:,:,-n:], x[:,:,:,:-n+end]), dim=3)