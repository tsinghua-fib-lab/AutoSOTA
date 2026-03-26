import torch
import torch.nn as nn

import pdb

CLIPMIN = 1e-4



def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        group_size=None,
        weight=None,
    ):
        super().__init__()
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.group_size = group_size if group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % group_size == 0
        self.enable = True
        
        # init scale and zero point through Max-Min quantization
        with torch.no_grad():
            if weight is not None:
                x = weight.reshape(-1,self.group_size)
                xmin = x.amin([-1], keepdim=True)
                xmax =  x.amax([-1], keepdim=True)
                range = xmax - xmin
                scale = range / (2**self.n_bits-1)
                scale = scale.clamp(min=1e-4, max=1e4)
                zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4) 
                self.scale = nn.Parameter(scale)
                self.zero_point = nn.Parameter(zero_point.round())
            

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = int(2 ** (n_bits) - 1)
        
    def fake_quant(self, x):
        scale = clamp_ste(self.scale,1e-4, 1e4)
        round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)
        
        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        return x_dequant
    

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x

        x_dequant = self.fake_quant(x)
        return x_dequant

        

    