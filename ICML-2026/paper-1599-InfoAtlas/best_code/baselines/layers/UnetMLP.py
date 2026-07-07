
from functools import partial
import torch
import torch.nn.functional as F 
from torch import Tensor   
from torch import nn


# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        # nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Linear(dim, default(dim_out, dim))
    )


def Downsample(dim, dim_out=None):
    return nn.Linear(dim, default(dim_out, dim))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, shift_scale=True):
        super().__init__()
        # self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.proj = nn.Linear(dim, dim_out)
        #self.act = nn.SiLU()
        self.act = SiLU()
        # self.act = nn.Relu()
        self.norm = nn.GroupNorm(groups, dim)
        # self.norm = nn.BatchNorm1d( dim)
        self.shift_scale = shift_scale

    def forward(self, x, t=None):
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)

        if exists(t):
            if self.shift_scale:
                scale, shift = t
                x = x * (scale.squeeze() + 1) + shift.squeeze()
            else:
                x = x + t

        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=32, shift_scale=False):
        super().__init__()
        self.shift_scale = shift_scale
        self.mlp = nn.Sequential(
            #nn.SiLU(),
            SiLU(),
            # nn.Linear(time_emb_dim, dim_out * 2)
            nn.Linear(time_emb_dim, dim_out*2 if shift_scale else dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups,
                            shift_scale=shift_scale)
        self.block2 = Block(dim_out, dim_out, groups=groups,
                            shift_scale=shift_scale)
        # self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.lin_layer = nn.Linear(
            dim, dim_out) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):

            time_emb = self.mlp(time_emb)
            scale_shift = time_emb

        h = self.block1(x, t=scale_shift)

        h = self.block2(h)

        return h + self.lin_layer(x)


class UnetMLP_simple(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=128,
        dim_mults=(1, 1),
        resnet_block_groups=8,
        time_dim=128,
        nb_var=1,
    ):
        super().__init__()

        # determine dimensions
        self.nb_var = nb_var
        init_dim = default(init_dim, dim)
        if init_dim == None:
            init_dim = dim * dim_mults[0]

        dim_in = dim
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        self.init_lin = nn.Linear(dim, init_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(nb_var, time_dim),
            #nn.GELU(),
            GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            module = nn.ModuleList([block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                                    #        block_klass(dim_in, dim_in, time_emb_dim = time_dim)
                                    ])

            # module.append( Downsample(dim_in, dim_out) if not is_last else nn.Linear(dim_in, dim_out))
            self.downs.append(module)

        mid_dim = dims[-1]
        joint_dim = mid_dim
       # joint_dim = 24
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # self.mid_block2 = block_klass(joint_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            module = nn.ModuleList([block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                                    #       block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim)
                                    ])
            # module.append( Upsample(dim_out, dim_in) if not is_last else  nn.Linear(dim_out, dim_in))
            self.ups.append(module)

        # default_out_dim = channels * (1 if not learned_variance else 2)

        self.out_dim = dim_in

        self.final_res_block = block_klass(
            init_dim * 2, init_dim, time_emb_dim=time_dim)

        self.proj = nn.Linear(init_dim, dim)

        self.proj.weight.data.fill_(0.0)
        self.proj.bias.data.fill_(0.0)

        self.final_lin = nn.Sequential(
            nn.GroupNorm(resnet_block_groups, init_dim),
            #nn.SiLU(),
            SiLU(),
            self.proj
        )

    def forward(self, x, t=None, std=None):
        t = t.reshape(t.size(0), self.nb_var)

        x = self.init_lin(x.float())

        r = x.clone()

        t = self.time_mlp(t).squeeze()

        h = []

        for blocks in self.downs:

            block1 = blocks[0]

            x = block1(x, t)

            h.append(x)
       #     x = downsample(x)

        # x = self.mid_block1(x, t)

        # x = self.mid_block2(x, t)

        for blocks in self.ups:

            block1 = blocks[0]
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            # x = torch.cat((x, h.pop()), dim = 1)
            # x = block2(x, t)

           # x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)

        if std != None:
            return self.final_lin(x) / std
        else:
            return self.final_lin(x)
        
        
        
        
        
        
class GELU(nn.Module):
    
    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.gelu(x.float()).type_as(x)
        

        
        
class SiLU(nn.Module):
    
    def forward(self, x: Tensor) -> Tensor:
        return x*torch.nn.functional.sigmoid(x.float()).type_as(x)
        
        
        
