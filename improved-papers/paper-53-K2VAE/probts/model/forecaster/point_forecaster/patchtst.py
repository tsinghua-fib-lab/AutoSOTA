# ---------------------------------------------------------------------------------
# Portions of this file are derived from PatchTST
# - Source: https://github.com/yuqinie98/PatchTST/tree/main
# - Paper: PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
# - License: Apache-2.0

# We thank the authors for their contributions.
# -----
# ----------------------------------------------------------------------------


import torch.nn as nn
from torch import Tensor
from typing import Optional

from probts.model.forecaster import Forecaster
from probts.model.nn.arch.PatchTSTModule.PatchTST_backbone import PatchTST_backbone
from probts.model.nn.arch.PatchTSTModule.PatchTST_layers import series_decomp
from torch.distributions import Normal
class PatchTST(Forecaster):
    def __init__(
        self,
        stride: int,
        patch_len: int,
        padding_patch: str = None,
        max_seq_len: int = 1024,
        n_layers:int = 3,
        n_heads = 16,
        d_k: int = None,
        d_v: int = None,
        d_ff: int = 256,
        attn_dropout: float = 0.,
        dropout: float = 0.,
        act: str = "gelu", 
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = 'zeros',
        learn_pe: bool = True,
        attn_mask: Optional[Tensor] = None,
        individual: bool = False,
        head_type: str = 'flatten',
        padding_var: Optional[int] = None, 
        revin: bool = True,
        key_padding_mask: str = 'auto',
        affine: bool = False,
        subtract_last: bool = False,
        decomposition: bool = False,
        kernel_size: int = 3,
        fc_dropout: float = 0.,
        head_dropout: float = 0.,
        f_hidden_size: int = 40,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if self.input_size != self.target_dim:
            self.enc_linear = nn.Linear(
                in_features=self.input_size, out_features=self.target_dim
            )
        else:
            self.enc_linear = nn.Identity()

        # Load parameters
        c_in = self.input_size
        context_window = self.context_length
        target_window = self.prediction_length

        # Model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=f_hidden_size,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=False, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=f_hidden_size,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=False, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=f_hidden_size,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=False, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last)
        # self.loss_fn = nn.MSELoss(reduction='none')
        self.loss_fn = lambda dist,targets: -dist.log_prob(targets)

    
    def forward(self, x):
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x_mu,x_sigma = self.model(x)
            x_mu = x_mu.permute(0,2,1)    # x: [Batch, Input length, Channel]
            x_sigma = x_sigma.permute(0,2,1)    # x: [Batch, Input length, Channel]
            return Normal(loc=x_mu, scale=x_sigma)
        return x

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'encode')
        inputs = self.enc_linear(inputs)
        dist = self(inputs)
        
        targets = batch_data.future_target_cdf
        loss = self.loss_fn(dist, targets)

        # loss = self.get_weighted_loss(batch_data, loss)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'encode')
        inputs = self.enc_linear(inputs)
        dist = self(inputs)
        outputs = dist.rsample(sample_shape=(num_samples,))
        return outputs
