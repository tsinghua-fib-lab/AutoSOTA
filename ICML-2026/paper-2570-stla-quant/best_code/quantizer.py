import torch
import torch.nn as nn

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class MinMaxQuantizer(nn.Module):
    def __init__(self, shape=1):
        super(MinMaxQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))
        self.register_buffer('zeta', torch.zeros(shape))

    def configure(self, bits, per_channel=False, sym=True, mse=False):
        self.nbits = bits
        self.maxq = torch.tensor(2 ** bits - 1)
        self.per_channel = per_channel
        self.sym = sym
        self.mse = mse

    def find_params(self, x):
        dev = x.device
        org_x = x
        d_in = org_x.flatten(1).shape[-1]
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.per_channel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        scale = (xmax - xmin) / self.maxq
        if self.sym:
            zero = torch.full_like(scale, (self.maxq + 1) / 2)
        else:
            zero = torch.round(-xmin / scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(.8 * 100)):
                p = 1 - i / 100
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                zero1 = zero1.to(dev)
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(2)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        if not self.per_channel:
            tmp = shape[0]
            scale = scale.repeat(tmp)
            zero = zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        scale = scale.reshape(shape)
        zero = zero.reshape(shape)
        zeta = torch.ones((1, d_in), device=dev)

        if len(shape) == 4:
            scale = scale.reshape((1, -1, 1, 1))
            zero = zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            scale = scale.reshape((1, 1, -1))
            zero = zero.reshape((1, 1, -1)) 
        
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.zero = nn.Parameter(zero, requires_grad=False)
        self.zeta = nn.Parameter(zeta, requires_grad=False)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)