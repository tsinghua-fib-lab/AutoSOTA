import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union, Callable, List, Tuple
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

import platform

# Custom multi_head_attention_forward
from .functional import multi_head_attention_forward


__all__ = [
    'TransformerEncoderLayer',
    'TransformerEncoder',
    'MultiheadAttention'
]


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Custom Transformer Encoder Block with optional access to the attention weights.

    This subclass extends PyTorch's TransformerEncoderLayer to optionally expose
    the self-attention weights via the ``self_attn_weights`` attribute.

    Attributes:
        self_attn (nn.Module): The multi-head self-attention module used in the layer.
        linear1 (nn.Linear): First feedforward layer.
        dropout (nn.Dropout): Dropout layer after the first feedforward layer.
        linear2 (nn.Linear): Second feedforward layer.
        norm1 (nn.LayerNorm): Layer normalization for the input to the first FFN.
        norm2 (nn.LayerNorm): Layer normalization for the output of the first FFN.
        activation (callable or str): Activation function used between FFN layers.
        return_attn (bool): If True, the forward pass stores attention weights.
        _self_attn_weights (Tensor): Cached attention weights when return_attn is True.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, return_attn: bool=False, device=None, dtype=None):
        super(TransformerEncoderLayer, self).__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, 
            norm_first=norm_first, bias=bias, device=device, dtype=dtype
        )
        """Initialize the custom TransformerEncoderLayer.

        Args:
            d_model (int): Number of expected features in the input.
            nhead (int): Number of heads in the multiheadattention model.
            dim_feedforward (int, optional): Dimension of the feedforward network. Default: 2048.
            dropout (float, optional): Dropout probability. Default: 0.1.
            activation (str or callable, optional): Activation function to use. Default: ReLU.
            layer_norm_eps (float, optional): Epsilon value for LayerNorm. Default: 1e-5.
            batch_first (bool, optional): If True, inputs are provided as (batch, seq, feature). Default: False.
            norm_first (bool, optional): If True, layer norm is applied before attention/FFN. Default: False.
            bias (bool, optional): If True, add bias to linear layers. Default: True.
            return_attn (bool, optional): If True, store attention weights for inspection. Default: False.
            device (torch.device, optional): Target device for parameters. Default: None.
            dtype (torch.dtype, optional): Target data type for parameters. Default: None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        
        #choose = False
        if platform.system() == 'Linux' and False:
            print(f"Using custom {MultiheadAttention}")
            # use custom torch extension on linux
            self.self_attn = MultiheadAttention(
                d_model,
                nhead,
                dropout=dropout,
                bias=bias,
                batch_first=batch_first,
                **factory_kwargs,
            )
        else:
            print(f"Using custom torch {nn.MultiheadAttention}")
            # stick to torch implementation for other OS
            self.self_attn = nn.MultiheadAttention(
                d_model,
                nhead,
                dropout=dropout,
                bias=bias,
                batch_first=batch_first,
                **factory_kwargs,
            )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation
        self.return_attn = return_attn
        self._self_attn_weights = None
    
    # self-attention block
    def _sa_block(self, 
        x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], 
        is_causal: bool = False,) -> Tensor | Tuple[Tensor]:
        """Self-attention block with optional attention weight caching.

        Args:
            x (Tensor): Input features.
            attn_mask (Tensor, optional): Attention mask.
            key_padding_mask (Tensor, optional): Padding mask for keys.
            is_causal (bool): If True, apply causality mask.

        Returns:
            Tensor or tuple: Either only the attention values or the attention weights and values 
                are returned. 
                attention values (Tensor): Output features after self-attention and dropout.
                attention weights (Tensor, optional): Output attention weights.
        """
        if self.return_attn:
            x, attn_weights = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=True, # changed form False
                            average_attn_weights=True,
                            is_causal=is_causal)
            # cache attention
            #print('attn_weights_encoderblock', attn_weights.size())
            self._self_attn_weights = attn_weights # [bs, seq_len, seq_len] .mean(dim=0) # [bs, seq_len, seq_len] -> [seq_len, seq_len]
        
        else:
            x = self.self_attn(x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False, # changed form False
                average_attn_weights=True,
                is_causal=is_causal)[0]

        return self.dropout1(x)

    @property
    def self_attn_weights(self,):
        """Get cached self-attention weights if computed."""
        return self._self_attn_weights


class TransformerEncoder(nn.Module):
    """A configurable Transformer encoder composed of multiple encoder layers.

    This module builds a stack of the TransformerEncoderLayer modules and an optional
    final normalization layer. It supports returning attention weights from each layer.

    Attributes:
        return_attn (bool): If True, store and return attention weights from each layer.
        transformer_encoder (nn.TransformerEncoder): The underlying PyTorch Transformer encoder.
    """
    def __init__(self, 
        # encoder layer
        d_model: int, 
        nhead: int, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5, 
        batch_first: bool = False, 
        norm_first: bool = False,
        bias: bool = True, 
        device=None, 
        dtype=None,
        # encoder
        num_layers: int=None,
        norm: Optional[nn.Module] = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
        return_attn: bool = True,
        *args, **kwargs) -> None:
        super(TransformerEncoder, self).__init__()
        """Initialize the TransformerEncoder.

        Args:
            d_model (int): Number of expected features in the input.
            nhead (int): Number of heads in the multi-head attention.
            dim_feedforward (int, optional): Dimension of the feedforward network. Default: 2048.
            dropout (float, optional): Dropout probability. Default: 0.1.
            activation (str or callable, optional): Activation function. Default: ReLU.
            layer_norm_eps (float, optional): Epsilon for LayerNorm. Default: 1e-5.
            batch_first (bool, optional): If True, inputs are (batch, seq, feature). Default: False.
            norm_first (bool, optional): If True, apply layer norms before other ops. Default: False.
            bias (bool, optional): If True, include bias in linear layers. Default: True.
            device (torch.device, optional): Target device for parameters. Default: None.
            dtype (torch.dtype, optional): Target dtype for parameters. Default: None.
            num_layers (int, optional): Number of encoder layers. Default: 1.
            norm (nn.Module, optional): Optional normalization module. If None, uses LayerNorm.
            enable_nested_tensor (bool, optional): Enable nested tensor support. Default: True.
            mask_check (bool, optional): Enable mask checks. Default: True.
            return_attn (bool, optional): If True, store attention weights. Default: True.
            *args, **kwargs: Additional arguments for base class.
        """
        self.return_attn = return_attn
        num_layers = num_layers if num_layers is not None else 1

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, 
            norm_first=norm_first, bias=bias, return_attn=return_attn, device=device, dtype=dtype, *args, **kwargs
        )
        encoder_norm = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **kwargs) if norm is None else norm
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm,
            enable_nested_tensor=enable_nested_tensor,
            mask_check=mask_check,
        )
        

    def forward(self, 
        src: Tensor, 
        N: Union[List[int], Tensor],
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
        aggregate_attn_per_batch: bool=False
        ) -> Tuple[Tensor, List[Tensor]]:
        """Forward pass through the Transformer encoder.

        Args:
            src (Tensor): Input sequence tensor.
            N (list or Tensor): Sequence lengths for attention aggregation.
            mask (Tensor, optional): Optional attention mask. Default: None.
            src_key_padding_mask (Tensor, optional): Padding mask for source keys. Default: None.
            is_causal (bool, optional): If True, apply causal masking. Default: None.
            aggregate_attn_per_batch (bool, optional): If True, aggregate attention per batch when returning. Default: False.

        Returns:
            tuple: (output, attn_weights) where
                - output (Tensor): Encoder output.
                - attn_weights (list[Tensor] or None): List of attention weights per layer if return_attn is True; otherwise None.
        """
        attn_weights = None
        if self.return_attn:
            out = self.transformer_encoder.forward(src=src, mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
            attn_weights = [
                attn_batch_mean_adjusted_for_padding(l.self_attn_weights, N=N) if aggregate_attn_per_batch else l.self_attn_weights
                for l in self.transformer_encoder.layers
            ]
        else:
            out = self.transformer_encoder.forward(src=src, mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
    
        return out, attn_weights


def attn_batch_mean_adjusted_for_padding(x: Tensor, N: Union[List[int], Tensor]):
    """Compute mean attention weights per batch, adjusted for padding lengths.

    Args:
        x (Tensor): Attention weights for a batch, shape [bs, N, N].
        N (list or Tensor): Sequence lengths for each item in the batch.

    Returns:
        mean (Tensor): Mean attention weights per batch, accounting for padding.
    """
    if any(elem != 25 for elem in N):
        # found at least one element that is padded 
        
        # Create a dynamic mask
        mask = torch.zeros_like(x, dtype=torch.bool)
        for i, length in enumerate(N):
            mask[i, length:, :] = True

        # Calculate the mean, ignoring the padding
        x_mean = torch.sum(x * (~mask), dim=(0)) / len(N)
    else:
        x_mean = x.mean(dim=0)
    return x_mean 


class MultiheadAttention(nn.Module):
    r"""
    Adjusted from Pytorch Source Code

    Allows the model to jointly attend to information from different representation subspaces.

    Method described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    ``nn.MultiHeadAttention`` will use the optimized implementations of
    ``scaled_dot_product_attention()`` when possible.

    In addition to support for the new ``scaled_dot_product_attention()``
    function, for speeding up Inference, MHA will use
    fastpath inference with support for Nested Tensors, iff:

    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor).
    - inputs are batched (3D) with ``batch_first==True``
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
      nor ``attn_mask`` is passed
    - autocast is disabled

    If the optimized inference fastpath implementation is in use, a
    `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
    ``query``/``key``/``value`` to represent padding more efficiently than using a
    padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
    will be returned, and an additional speedup proportional to the fraction of the input
    that is padding can be expected.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """

    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super().__setstate__(state)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Compute attention outputs using query, key, and value embeddings.

        Supports optional parameters for padding, masks and attention weights.

    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and float masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Set ``need_weights=False`` to use the optimized ``scaled_dot_product_attention``
            and achieve the best performance for MHA.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
            If both attn_mask and key_padding_mask are supplied, their types should match.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)
        is_causal: If specified, applies a causal mask as attention mask.
            Default: ``False``.
            Warning:
            ``is_causal`` provides a hint that ``attn_mask`` is the
            causal mask. Providing incorrect hints can result in
            incorrect execution, including forward and backward
            compatibility.

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """
        why_not_fast_path = ''
        if ((attn_mask is not None and torch.is_floating_point(attn_mask))
           or (key_padding_mask is not None) and torch.is_floating_point(key_padding_mask)):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif _is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(_check_arg_device(x) for x in tensor_args):
                why_not_fast_path = ("some Tensor argument's device is neither one of "
                                     f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}")
            elif torch.is_grad_enabled() and any(_arg_requires_grad(x) for x in tensor_args):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        self.in_proj_weight,
                        self.in_proj_bias,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            #attn_output, attn_output_weights = F.multi_head_attention_forward(
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        else:
            #attn_output, attn_output_weights = F.multi_head_attention_forward(
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def merge_masks(self, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                    query: Tensor) -> Tuple[Optional[Tensor], Optional[int]]:
        r"""Determine mask type and combine masks if necessary.

        If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2

            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type


def _is_make_fx_tracing():
    if not torch.jit.is_scripting():
        torch_dispatch_mode_stack = torch.utils._python_dispatch._get_current_dispatch_mode_stack()
        return any(type(x) == torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode for x in torch_dispatch_mode_stack)
    else:
        return False

def _check_arg_device(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.device.type in ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
    return True


def _arg_requires_grad(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.requires_grad
    return False


