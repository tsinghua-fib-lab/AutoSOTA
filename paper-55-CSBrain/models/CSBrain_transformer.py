import torch
from torch import nn, einsum,Tensor
from torch.nn import functional as Fun

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["OMP_NUM_THREADS"] = "1"       
from threadpoolctl import threadpool_limits
from collections import OrderedDict
import numpy as np
import copy
from typing import Optional, Any, Union, Callable




def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:

        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class CSBrain_TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src: Tensor,
            area_config: dict,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:

        output = src # [128, 19, 30, 200]
        for mod in self.layers:
            output = mod(output, area_config, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output




class TemEmbedEEGLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_sizes,
        stride=1
    ):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=dim_in, out_channels=dim_scale, kernel_size=(kt, 1),
                      stride=(stride, 1), padding=((kt - 1) // 2, 0))
            for (kt,), dim_scale in zip(kernel_sizes, dim_scales)
        ])

    def forward(self, x):
        batch, chans, time, d_model = x.shape

        x = x.view(batch * chans, d_model, time, 1)

        fmaps = [conv(x) for conv in self.convs]

        assert all(f.shape[2] == time for f in fmaps), "Time dimension mismatch after convolutions!"

        x = torch.cat(fmaps, dim=1)

        x = x.view(batch, chans, time, -1)

        return x


class BrainEmbedEEGLayer(nn.Module):
    def __init__(self, dim_in=200, dim_out=200, total_regions=5):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.total_regions = total_regions

        kernel_sizes = [1, 3, 5]

        dim_scales = [dim_out // (2 ** (i + 1)) for i in range(len(kernel_sizes) - 1)]
        dim_scales.append(dim_out - sum(dim_scales))

        self.region_blocks = nn.ModuleDict({
            f"region_{i}": nn.ModuleList([
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=dim_scale,
                    kernel_size=(k, 1),
                    padding=(0, 0),
                    groups=1
                ) for k, dim_scale in zip(kernel_sizes, dim_scales)
            ])
            for i in range(total_regions)
        })

    def forward(self, x, area_config):
        batch, chans, T, F = x.shape
        device = x.device

        output = torch.zeros((batch, chans, T, self.dim_out), device=device)

        for region_key, region_info in area_config.items():
            if region_key not in self.region_blocks:
                continue

            channel_slice = region_info['slice']
            n_electrodes = region_info['channels']

            x_region = x[:, channel_slice, :, :]

            x_trans = x_region.permute(0, 2, 1, 3).reshape(-1, n_electrodes, F)
            x_trans = x_trans.permute(0, 2, 1).unsqueeze(-1)

            fmap_outputs = []
            for conv, k in zip(self.region_blocks[region_key], [1, 3, 5]):
                pad_size = (k - 1) // 2

                if n_electrodes == 1:
                    x_padded = Fun.pad(x_trans, (0, 0, pad_size, pad_size), mode='constant', value=0)
                else:
                    x_padded = Fun.pad(x_trans, (0, 0, pad_size, pad_size), mode='circular')

                fmap_outputs.append(conv(x_padded))

            fmap_cat = torch.cat(fmap_outputs, dim=1)
            fmap_out = fmap_cat.squeeze(-1).permute(0, 2, 1).reshape(batch, T, n_electrodes, self.dim_out)
            fmap_out = fmap_out.permute(0, 2, 1, 3)

            output[:, channel_slice, :, :] = fmap_out

        return output


class BrainAreaConv(nn.Module):
    def __init__(self, area_config):
        super().__init__()
        self.conv_layers = nn.ModuleDict({
            name: nn.Conv2d(
                in_channels=cfg['channels'],
                out_channels=cfg['channels'],
                kernel_size=(1, 1),  
                padding=(0, 0)       
            ) for name, cfg in area_config.items()
        })
        self.area_config = area_config
        
    def forward(self, x):
        outputs = []
        for name, cfg in self.area_config.items():
            x_area = x[:, cfg['slice'], :, :]
            conv = self.conv_layers[name]
            outputs.append(conv(x_area))
        return torch.cat(outputs, dim=1)


# transformer classes

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv2d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv2d(dim * mult, dim, 1)
    )



def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return Fun.relu
    elif activation == "gelu":
        return Fun.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    r"""Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


if __name__ == '__main__':
    encoder_layer = TransformerEncoderLayer(
        d_model=256, nhead=4, dim_feedforward=1024, batch_first=True, norm_first=True,
        activation=Fun.gelu
    )
    encoder = TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
    encoder = encoder.cuda()

    a = torch.randn((4, 19, 30, 256)).cuda()
    b = encoder(a)
    print(a.shape, b.shape)

    

if __name__ == '__main__':
    pass