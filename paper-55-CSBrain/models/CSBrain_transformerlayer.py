import copy
from typing import Optional, Any, Union, Callable

import torch
import torch.nn as nn
import warnings
from torch import Tensor
from torch.nn import functional as F
import numpy as np


class CSBrain_TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, bias: bool = True,
                 area_config: dict = {}, sorted_indices: list = []):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

        self.inter_region_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                       bias=bias, batch_first=batch_first)

        self.inter_window_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                       bias=bias, batch_first=batch_first)

        self.global_fc = nn.Linear(d_model, d_model, bias=bias)

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = getattr(F, activation, F.relu)
        self.activation = activation

        self.area_config = area_config
        self.mask_builder = None
        self.region_attn_mask = None
        self.region_indices_dict = None

        if area_config is not None:
            total_channels = sum(len(range(info['slice'].start or 0, info['slice'].stop, info['slice'].step or 1))
                                 if isinstance(info['slice'], slice) else len(info['slice'])
                                 for info in area_config.values())

            self.mask_builder = RegionAttentionMaskBuilder(total_channels, area_config)
            self.region_attn_mask = self.mask_builder.get_mask()
            self.region_indices_dict = self.mask_builder.get_region_indices()

    def forward(
        self,
        src: torch.Tensor,
        area_config: Optional[dict] = None,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = src
        x = x + self._inter_window_attention(self.norm1(x), src_mask, src_key_padding_mask)

        if self.region_attn_mask is None and area_config is not None:
            x = x + self._inter_region_attention_dynamic(self.norm2(x), area_config, src_mask, src_key_padding_mask)
        else:
            x = x + self._inter_region_attention_static(self.norm2(x), src_mask, src_key_padding_mask)

        x = x + self._ff_block(self.norm3(x))
        return x

    def _inter_region_attention_static(self, x: torch.Tensor,
                                       attn_mask: Optional[torch.Tensor] = None,
                                       key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.region_attn_mask is None or self.region_indices_dict is None:
            raise ValueError("no initialized region attention mask or region indices dictionary")

        batch, chans, T, F = x.shape

        x_reshaped = x.permute(0, 2, 1, 3)
        x_flat = x_reshaped.reshape(batch * T, chans, F)

        region_global_features = {}
        for region_name, region_indices in self.region_indices_dict.items():
            region_x = x[:, region_indices, :, :]
            region_global = region_x.mean(dim=1, keepdim=True)
            region_global_features[region_name] = region_global

        global_features = torch.zeros_like(x_flat)

        for region_name, region_indices in self.region_indices_dict.items():
            region_global = region_global_features[region_name]
            region_global = region_global.permute(0, 2, 1, 3)
            region_global = region_global.reshape(batch * T, 1, F)

            for idx in region_indices:
                global_features[:, idx:idx + 1, :] = region_global

        global_features = self.global_fc(global_features)
        x_enhanced = x_flat + global_features

        region_attn_mask = self.region_attn_mask.to(x.device)

        attn_output = self.inter_region_attn(
            x_enhanced, x_enhanced, x_enhanced,
            attn_mask=region_attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]

        attn_output = attn_output.reshape(batch, T, chans, F).permute(0, 2, 1, 3)

        return self.dropout1(attn_output)

    def _inter_window_attention(self, x: torch.Tensor,
                                attn_mask: Optional[torch.Tensor] = None,
                                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, chans, T, Fea = x.shape
        window_size = min(T, 5)

        num_windows = T // window_size
        original_T = T

        if T % window_size != 0:
            pad_length = window_size - (T % window_size)
            x = F.pad(x, (0, 0, 0, pad_length))
            T = T + pad_length
            num_windows = T // window_size

        x = x.view(batch, chans, num_windows, window_size, Fea)

        x = x.permute(0, 3, 1, 2, 4)
        x = x.reshape(batch * window_size * chans, num_windows, Fea)

        temporal_attn_mask = None
        if attn_mask is not None:
            if isinstance(attn_mask, torch.Tensor) and attn_mask.dim() == 2:
                temporal_attn_mask = torch.triu(
                    torch.ones(num_windows, num_windows, device=x.device) * float('-inf'),
                    diagonal=1
                )

        x = self.inter_window_attn(
            x, x, x,
            attn_mask=temporal_attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]

        x = x.reshape(batch, window_size, chans, num_windows, Fea)
        x = x.permute(0, 2, 3, 1, 4)

        x = x.reshape(batch, chans, T, Fea)
        if T != original_T:
            x = x[:, :, :original_T, :]

        return self.dropout2(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, F = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * T, C, F)

        x_ff = self.linear2(self.dropout(self.activation(self.linear1(x_reshaped))))

        x_ff = x_ff.reshape(B, T, C, F).permute(0, 2, 1, 3)

        return self.dropout3(x_ff)


class RegionAttentionMaskBuilder:
    def __init__(self, num_channels: int, area_config: dict, device=None):
        self.num_channels = num_channels
        self.area_config = area_config
        self.device = device

        self.region_indices_dict = self._process_region_indices()

        self.attention_mask = self._build_attention_mask()

    def _process_region_indices(self):
        region_indices_dict = {}

        for region_name, region_info in self.area_config.items():
            region_slice = region_info['slice']
            if isinstance(region_slice, slice):
                start = region_slice.start or 0
                stop = region_slice.stop
                step = region_slice.step or 1
                region_indices = list(range(start, stop, step))
            else:
                region_indices = list(region_slice)

            region_indices_dict[region_name] = region_indices

        return region_indices_dict

    def _build_attention_mask(self):
        device = self.device if self.device is not None else torch.device('cpu')
        region_attn_mask = torch.ones(self.num_channels, self.num_channels, device=device) * float('-inf')

        num_groups = max(len(indices) for indices in self.region_indices_dict.values())

        groups = [[] for _ in range(num_groups)]

        for g in range(num_groups):
            for region_name, region_indices in self.region_indices_dict.items():
                n_electrodes = len(region_indices)
                if n_electrodes == 0:
                    continue

                electrode_idx = region_indices[g % n_electrodes]
                groups[g].append(electrode_idx)

        for g, group_electrodes in enumerate(groups):
            for idx1 in group_electrodes:
                for idx2 in group_electrodes:
                    region_attn_mask[idx1, idx2] = 0

        return region_attn_mask

    def get_mask(self):
        return self.attention_mask

    def get_region_indices(self):
        return self.region_indices_dict


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

def _get_clones(module, N):
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
            return src_size[0]
        else:
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


if __name__ == '__main__':
    pass