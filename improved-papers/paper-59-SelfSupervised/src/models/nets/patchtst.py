# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from src.models.nets.components.patchtst.patchtst_backbone import Backbone
from src.models.modules.decomp import SeriesDecomp


class Model(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_channels: int,
        with_revin: bool,
        num_layers: int,
        num_heads: int,
        dim: int,
        d_ff: int,
        dropout: float,
        fc_dropout: float,
        head_dropout: float,
        patch_len: int,
        stride: int,
        max_seq_len: Optional[int] = 1024,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm: str = "LayerNorm",
        attn_dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: bool = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type="flatten",
        verbose: bool = False,
        if_snr: bool = False,
        num_models: int = 1,
        **kwargs
    ):

        super().__init__()

        # load parameters
        c_in = num_channels
        context_window = input_size
        target_window = output_size

        n_layers = num_layers
        n_heads = num_heads
        d_model = dim
        d_ff = d_ff
        dropout = dropout
        fc_dropout = fc_dropout
        head_dropout = head_dropout

        individual = False

        patch_len = patch_len
        stride = stride
        padding_patch = None

        revin = with_revin
        affine = False
        subtract_last = False

        decomposition = False
        kernel_size = None

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = SeriesDecomp(kernel_size)
            self.model_trend = Backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                if_snr=if_snr,
                **kwargs
            )
            self.model_res = Backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                if_snr=if_snr,
                **kwargs
            )
        else:
            self.model = Backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                if_snr=if_snr,
                **kwargs
            )

    def init_params(self):
        pass

    def predict(self, x, x_mark=None, *args, **kwargs):
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init, trend_init
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            y_hat = res + trend
        else:
            y_hat = self.model(x)
        return y_hat

    def forward(self, x, y, x_mark=None, y_mark=None, mode="train"):
        result = {}
        y_hat = self.predict(x)
        result["y_hat"] = y_hat
        result["loss_tar"] = F.mse_loss(y_hat, y)
        result["loss_total"] = result["loss_tar"]

        return result
