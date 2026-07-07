from typing import Dict, Text, Callable, List, Optional
from collections import defaultdict

import numpy as np
import torch, math
import torch.nn.functional as F
from torch import nn

from .hook import HookManager


class MultiheadAttention(nn.Module):
    """
    There are variety of ways to look at multihead attention. Because of that I implemented a few so it will be easy to compare.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
        hook: Optional[HookManager] = None,
    ):
        super().__init__()
        self.hook = hook or HookManager()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim)))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim)))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

    def forward_direct(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.hook(
            "in_proj_bias.post",
            ret=self.hook("in_proj.post", ret=x @ self.in_proj_weight.T)
            + self.in_proj_bias,
        )
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        k = self.hook("k", ret=k)
        q = self.hook("q", ret=q)
        v = self.hook("v", ret=v)
        dk = q.size()[-1]
        q = q / math.sqrt(dk)
        q = self.hook("q_norm", ret=q)
        attn = q @ k.transpose(-2, -1)  # [B, H, N, N]
        attn = self.hook("pre_mask", ret=attn)
        if attn_mask is not None:
            attn += attn_mask
        attn = self.hook("post_mask", ret=attn)
        attn = attn.softmax(dim=-1)
        attn = self.hook("post_softmax", ret=attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.hook("attn_v", ret=x)
        x = self.hook(
            "out_proj_bias.post",
            ret=self.hook("out_proj.post", ret=x @ self.out_proj.weight.T)
            + self.out_proj.bias,
        )
        return x

    def _split_qkv_weight(self):
        q_weight, k_weight, v_weight = (
            self.in_proj_weight[: self.embed_dim].reshape(
                self.num_heads, self.head_dim, -1
            ),
            self.in_proj_weight[self.embed_dim : self.embed_dim * 2].reshape(
                self.num_heads, self.head_dim, -1
            ),
            self.in_proj_weight[self.embed_dim * 2 :].reshape(
                self.num_heads, self.head_dim, -1
            ),
        )
        return q_weight, k_weight, v_weight

    def _split_qkv_bias(self):
        q_bias, k_bias, v_bias = (
            self.in_proj_bias[: self.embed_dim].reshape(
                1, self.num_heads, 1, self.head_dim
            ),
            self.in_proj_bias[self.embed_dim : self.embed_dim * 2].reshape(
                1, self.num_heads, 1, self.head_dim
            ),
            self.in_proj_bias[self.embed_dim * 2 :].reshape(
                1, self.num_heads, 1, self.head_dim
            ),
        )
        return q_bias, k_bias, v_bias

    def forward_qkv(self, x, attn_mask=None):
        B, N, C = x.shape
        q_weight, k_weight, v_weight = (
            self.in_proj_weight[: self.embed_dim],
            self.in_proj_weight[self.embed_dim : self.embed_dim * 2],
            self.in_proj_weight[self.embed_dim * 2 :],
        )
        q_bias, k_bias, v_bias = (
            self.in_proj_bias[: self.embed_dim],
            self.in_proj_bias[self.embed_dim : self.embed_dim * 2],
            self.in_proj_bias[self.embed_dim * 2 :],
        )
        q = (
            self.hook(
                "in_q_bias.post",
                ret=self.hook("in_q.post", ret=x @ q_weight.T) + q_bias,
            )
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.hook(
                "in_k_bias.post",
                ret=self.hook("in_k.post", ret=x @ k_weight.T) + k_bias,
            )
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.hook(
                "in_v_bias.post",
                ret=self.hook("in_v.post", ret=x @ v_weight.T) + v_bias,
            )
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        dk = q.size()[-1]
        q = q / math.sqrt(dk)
        q = self.hook("q_norm", ret=q)
        attn = q @ k.transpose(-2, -1)
        attn = self.hook("attention.pre_mask", ret=attn)
        if attn_mask is not None:
            attn += attn_mask
        attn = self.hook("attention.post_mask", ret=attn)
        attn = attn.softmax(dim=-1)
        attn = self.hook("attention.post_softmax", ret=attn)  # [B, H, N, N]
        x = torch.einsum("bhnm,bhmc->bhnmc", attn, v)
        x = self.hook("extended_attn_v", ret=x)
        x = x.sum(axis=3).transpose(1, 2).reshape(B, N, C)
        x = self.hook("attn_v", ret=x)
        x = self.hook(
            "out.post_bias",
            ret=self.hook("out.post", ret=x @ self.out_proj.weight.T)
            + self.out_proj.bias,
        )
        return x

    def forward_per_head_no_spatial(self, x, attn_mask=None):
        B, N, C = x.shape
        q_weight, k_weight, v_weight = self._split_qkv_weight()
        q_bias, k_bias, v_bias = self._split_qkv_bias()
        q = self.hook(
            "in_q_bias.post",
            ret=self.hook("in_q.post", ret=torch.einsum("bnc,hdc->bhnd", x, q_weight))
            + q_bias,
        )
        k = self.hook(
            "in_k_bias.post",
            ret=self.hook("in_k.post", ret=torch.einsum("bnc,hdc->bhnd", x, k_weight))
            + k_bias,
        )
        v = self.hook(
            "in_v_bias.post",
            ret=self.hook("in_v.post", ret=torch.einsum("bnc,hdc->bhnd", x, v_weight))
            + v_bias,
        )  # (B, self.num_heads, N, self.head_dim)
        dk = q.size()[-1]
        q = q / math.sqrt(dk)
        q = self.hook("q_norm", ret=q)
        attn = q @ k.transpose(-2, -1)
        attn = self.hook("attention.pre_mask", ret=attn)
        if attn_mask is not None:
            attn += attn_mask
        attn = self.hook("attention.post_mask", ret=attn)
        attn = attn.softmax(dim=-1)
        attn = self.hook("attention.post_softmax", ret=attn)  # [B, H, N, N]
        x = torch.einsum(
            "bhnm,bhmc->bnhc", attn, v
        )  # We also switch here back from head-first to n-first
        x = self.hook("attn_v", ret=x)
        x = self.hook(
            "out.post",
            ret=torch.einsum(
                "bnhc,dhc->bnhd",
                x,
                self.out_proj.weight.reshape(
                    self.embed_dim, self.num_heads, self.head_dim
                ),
            ),
        )
        x = self.hook("out.post_collapse", ret=x.sum(axis=2))
        x = self.hook("out.post_bias", ret=x + self.out_proj.bias)
        return x


    def forward_per_head(self, x, attn_mask=None):
        B, N, C = x.shape
        q_weight, k_weight, v_weight = self._split_qkv_weight()
        q_bias, k_bias, v_bias = self._split_qkv_bias()
        q = self.hook(
            "in_q_bias.post",
            ret=self.hook("in_q.post", ret=torch.einsum("bnc,hdc->bhnd", x, q_weight))
            + q_bias,
        )
        k = self.hook(
            "in_k_bias.post",
            ret=self.hook("in_k.post", ret=torch.einsum("bnc,hdc->bhnd", x, k_weight))
            + k_bias,
        )
        v = self.hook(
            "in_v_bias.post",
            ret=self.hook("in_v.post", ret=torch.einsum("bnc,hdc->bhnd", x, v_weight))
            + v_bias,
        )  # (B, self.num_heads, N, self.head_dim)
        dk = q.size()[-1]
        q = q / math.sqrt(dk)
        q = self.hook("q_norm", ret=q)
        attn = q @ k.transpose(-2, -1)
        attn = self.hook("attention.pre_mask", ret=attn)
        if attn_mask is not None:
            attn += attn_mask
        attn = self.hook("attention.post_mask", ret=attn)
        attn = attn.softmax(dim=-1)
        attn = self.hook("attention.post_softmax", ret=attn)  # [B, H, N, N]
        x = torch.einsum(
            "bhnm,bhmc->bnmhc", attn, v
        )  # We also switch here back from head-first to n-first
        x = self.hook("extended_attn_v", ret=x)
        x = self.hook(
            "out.post",
            ret=torch.einsum(
                "bnmhc,dhc->bnmhd",
                x,
                self.out_proj.weight.reshape(
                    self.embed_dim, self.num_heads, self.head_dim
                ),
            ),
        )
        x = self.hook("out.post_collapse", ret=x.sum(axis=[2, 3]))
        x = self.hook("out.post_bias", ret=x + self.out_proj.bias)
        return x

    def _get_ov_circuit(
        self,
    ):
        reshaped_o = self.out_proj.weight.reshape(
            self.embed_dim, self.num_heads, self.head_dim
        )
        _, _, v_weight = self._split_qkv_weight()  # num_heads, head_dim, embed_dim
        _, _, v_bias = self._split_qkv_bias()  # 1, num_heads, 1, head_dim
        ov_circuit = torch.einsum("onh,nhi->oni", reshaped_o, v_weight)
        ov_bias_circuit = torch.einsum(
            "onh,bnxh->bnxo", reshaped_o, v_bias
        )  # [1, num_heads, 1, embed_dim]
        return ov_circuit, ov_bias_circuit

    def forward_ov_circuit(self, x, attn_mask=None):
        B, N, C = x.shape
        q_weight, k_weight, _ = self._split_qkv_weight()
        q_bias, k_bias, _ = self._split_qkv_bias()
        q = self.hook(
            "in_q_bias.post",
            ret=self.hook("in_q.post", ret=torch.einsum("bnc,hdc->bhnd", x, q_weight))
            + q_bias,
        )
        k = self.hook(
            "in_k_bias.post",
            ret=self.hook("in_k.post", ret=torch.einsum("bnc,hdc->bhnd", x, k_weight))
            + k_bias,
        )
        ov, ov_bias = self._get_ov_circuit()
        ov = self.hook("ov", ret=ov)
        ov_bias = self.hook("ov_bias", ret=ov_bias)
        v = self.hook(
            "ov_bias.post",
            ret=self.hook("ov.post", ret=torch.einsum("bnc,dhc->bhnd", x, ov))
            + ov_bias,
        )

        dk = q.size()[-1]
        q = q / math.sqrt(dk)
        q = self.hook("q_norm", ret=q)
        attn = q @ k.transpose(-2, -1)
        attn = self.hook("attention.pre_mask", ret=attn)
        if attn_mask is not None:
            attn += attn_mask
        attn = self.hook("attention.post_mask", ret=attn)
        attn = attn.softmax(dim=-1)
        attn = self.hook("attention.post_softmax", ret=attn)  # [B, H, N, N]
        x = torch.einsum(
            "bhnm,bhmc->bnmhc", attn, v
        )  # We also switch here back from head-first to n-first
        x = self.hook("extended_attn_ov", ret=x)
        x = self.hook("out.post_collapse", ret=x.sum(axis=[2, 3]))
        x = self.hook("out.post_bias", ret=x + self.out_proj.bias)
        return x

    def forward(self, x, attn_mask=None, method: Text = "direct"):
        if not self.batch_first:
            x = x.transpose(0, 1)
        if method == "direct":
            x = self.forward_direct(x, attn_mask=attn_mask)
        elif method == "qkv":
            x = self.forward_qkv(x, attn_mask=attn_mask)
        elif method == "head":
            x = self.forward_per_head(x, attn_mask=attn_mask)
        elif method == "head_no_spatial":
            x = self.forward_per_head_no_spatial(x, attn_mask=attn_mask)
        elif method == "ov_circuit":
            x = self.forward_ov_circuit(x, attn_mask=attn_mask)
        else:
            raise NotImplementedError('Unknown attention method')
        self.hook.finalize()
        if not self.batch_first:
            x = x.transpose(0, 1)
        return x