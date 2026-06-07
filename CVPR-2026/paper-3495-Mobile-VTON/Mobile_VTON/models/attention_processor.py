import inspect
import math
from typing import Callable, List, Optional, Tuple, Union
import logging
import sys
import os.path as osp
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate, is_torch_xla_available
from diffusers.utils.import_utils import is_torch_npu_available, is_torch_xla_version, is_xformers_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import (
    AttnProcessor,
    SpatialNorm,
    AttnProcessor2_0,
    XLAFlashAttnProcessor2_0,
    AttnProcessorNPU,
    CustomDiffusionAttnProcessor,
    CustomDiffusionXFormersAttnProcessor,
    CustomDiffusionAttnProcessor2_0,
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
    XFormersAttnAddedKVProcessor,
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0,
    IPAdapterXFormersAttnProcessor,
    JointAttnProcessor2_0,
    XFormersJointAttnProcessor,
    XFormersAttnProcessor,
    SlicedAttnProcessor,
    AttentionProcessor,
)
from diffusers.models.embeddings import apply_rotary_emb

if is_torch_npu_available():
    import torch_npu

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

if is_torch_xla_available():
    # flash attention pallas kernel is introduced in the torch_xla 2.3 release.
    if is_torch_xla_version(">", "2.2"):
        from torch_xla.experimental.custom_kernel import flash_attention
        from torch_xla.runtime import is_spmd
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

# Setup working directory
WORK_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
logger.debug(f"Working directory: {WORK_DIR}")
if WORK_DIR not in sys.path:
    logger.warning(f"Working directory ({WORK_DIR}) is not in sys.path. Adding it.")
    sys.path.append(WORK_DIR)


@maybe_allow_in_graph
class Attention(nn.Module):
    r"""Modified from Attention in diffusers.models.attention_processor:
            from diffusers.models.attention_processor import Attention
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
        out_context_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
        is_causal: bool = False,
    ):
        super().__init__()

        # To prevent circular import.
        from diffusers.models.normalization import FP32LayerNorm, LpNorm, RMSNorm

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.is_causal = is_causal

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(f_channels=query_dim, zq_channels=spatial_norm_dim)
        else:
            self.spatial_norm = None

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        elif qk_norm == "fp32_layer_norm":
            self.norm_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
            self.norm_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
        elif qk_norm == "layer_norm_across_heads":
            # Lumina applies qk norm across all heads
            self.norm_q = nn.LayerNorm(dim_head * heads, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head * kv_heads, eps=eps)
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim_head * heads, eps=eps)
            self.norm_k = RMSNorm(dim_head * kv_heads, eps=eps)
        elif qk_norm == "l2":
            self.norm_q = LpNorm(p=2, dim=-1, eps=eps)
            self.norm_k = LpNorm(p=2, dim=-1, eps=eps)
        else:
            raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None,'layer_norm','fp32_layer_norm','rms_norm'")

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
            self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        self.added_proj_bias = added_proj_bias
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            if self.context_pre_only is not None:
                self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
        else:
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(nn.Dropout(dropout))
        else:
            self.to_out = None

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, self.out_context_dim, bias=out_bias)
        else:
            self.to_add_out = None

        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "fp32_layer_norm":
                self.norm_added_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
                self.norm_added_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
            elif qk_norm == "rms_norm":
                self.norm_added_q = RMSNorm(dim_head, eps=eps)
                self.norm_added_k = RMSNorm(dim_head, eps=eps)
            else:
                raise ValueError(
                    f"unknown qk_norm: {qk_norm}. Should be one of `None,'layer_norm','fp32_layer_norm','rms_norm'`"
                )
        else:
            self.norm_added_q = None
            self.norm_added_k = None

        # set attention processor
        # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
        # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
        # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
        logger.debug(f"Setting attention processor to {processor}")
        if processor is None:
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
        self.set_processor(processor)

    def set_use_xla_flash_attention(
        self, use_xla_flash_attention: bool, partition_spec: Optional[Tuple[Optional[str], ...]] = None
    ) -> None:
        r"""
        Set whether to use xla flash attention from `torch_xla` or not.

        Args:
            use_xla_flash_attention (`bool`):
                Whether to use pallas flash attention kernel from `torch_xla` or not.
            partition_spec (`Tuple[]`, *optional*):
                Specify the partition specification if using SPMD. Otherwise None.
        """
        if use_xla_flash_attention:
            if not is_torch_xla_available:
                raise "torch_xla is not available"
            elif is_torch_xla_version("<", "2.3"):
                raise "flash attention pallas kernel is supported from torch_xla version 2.3"
            elif is_spmd() and is_torch_xla_version("<", "2.4"):
                raise "flash attention pallas kernel using SPMD is supported from torch_xla version 2.4"
            else:
                processor = XLAFlashAttnProcessor2_0(partition_spec)
        else:
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
        self.set_processor(processor)

    def set_use_npu_flash_attention(self, use_npu_flash_attention: bool) -> None:
        r"""
        Set whether to use npu flash attention from `torch_npu` or not.

        """
        if use_npu_flash_attention:
            processor = AttnProcessorNPU()
        else:
            # set attention processor
            # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
            # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
            # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
        self.set_processor(processor)

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ) -> None:
        r"""
        Set whether to use memory efficient attention from `xformers` or not.

        Args:
            use_memory_efficient_attention_xformers (`bool`):
                Whether to use memory efficient attention from `xformers` or not.
            attention_op (`Callable`, *optional*):
                The attention operation to use. Defaults to `None` which uses the default attention operation from
                `xformers`.
        """
        is_custom_diffusion = hasattr(self, "processor") and isinstance(
            self.processor,
            (CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor, CustomDiffusionAttnProcessor2_0),
        )
        is_added_kv_processor = hasattr(self, "processor") and isinstance(
            self.processor,
            (
                AttnAddedKVProcessor,
                AttnAddedKVProcessor2_0,
                SlicedAttnAddedKVProcessor,
                XFormersAttnAddedKVProcessor,
            ),
        )
        is_ip_adapter = hasattr(self, "processor") and isinstance(
            self.processor,
            (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0, IPAdapterXFormersAttnProcessor),
        )
        is_joint_processor = hasattr(self, "processor") and isinstance(
            self.processor,
            (
                JointAttnProcessor2_0,
                XFormersJointAttnProcessor,
            ),
        )

        if use_memory_efficient_attention_xformers:
            if is_added_kv_processor and is_custom_diffusion:
                raise NotImplementedError(
                    f"Memory efficient attention is currently not supported for custom diffusion for attention processor type {self.processor}"
                )
            if not is_xformers_available():
                raise ModuleNotFoundError(
                    (
                        "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                        " xformers"
                    ),
                    name="xformers",
                )
            elif not torch.cuda.is_available():
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                    " only available for GPU "
                )
            else:
                try:
                    # Make sure we can run the memory efficient attention
                    _ = xformers.ops.memory_efficient_attention(
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                    )
                except Exception as e:
                    raise e

            if is_custom_diffusion:
                processor = CustomDiffusionXFormersAttnProcessor(
                    train_kv=self.processor.train_kv,
                    train_q_out=self.processor.train_q_out,
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    attention_op=attention_op,
                )
                processor.load_state_dict(self.processor.state_dict())
                if hasattr(self.processor, "to_k_custom_diffusion"):
                    processor.to(self.processor.to_k_custom_diffusion.weight.device)
            elif is_added_kv_processor:
                # TODO(Patrick, Suraj, William) - currently xformers doesn't work for UnCLIP
                # which uses this type of cross attention ONLY because the attention mask of format
                # [0, ..., -10.000, ..., 0, ...,] is not supported
                # throw warning
                logger.warning(
                    "Memory efficient attention with `xformers` might currently not work correctly if an attention mask is required for the attention operation."
                )
                processor = XFormersAttnAddedKVProcessor(attention_op=attention_op)
            elif is_ip_adapter:
                processor = IPAdapterXFormersAttnProcessor(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    num_tokens=self.processor.num_tokens,
                    scale=self.processor.scale,
                    attention_op=attention_op,
                )
                processor.load_state_dict(self.processor.state_dict())
                if hasattr(self.processor, "to_k_ip"):
                    processor.to(
                        device=self.processor.to_k_ip[0].weight.device, dtype=self.processor.to_k_ip[0].weight.dtype
                    )
            elif is_joint_processor:
                processor = XFormersJointAttnProcessor(attention_op=attention_op)
            else:
                processor = XFormersAttnProcessor(attention_op=attention_op)
        else:
            if is_custom_diffusion:
                attn_processor_class = (
                    CustomDiffusionAttnProcessor2_0
                    if hasattr(F, "scaled_dot_product_attention")
                    else CustomDiffusionAttnProcessor
                )
                processor = attn_processor_class(
                    train_kv=self.processor.train_kv,
                    train_q_out=self.processor.train_q_out,
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                )
                processor.load_state_dict(self.processor.state_dict())
                if hasattr(self.processor, "to_k_custom_diffusion"):
                    processor.to(self.processor.to_k_custom_diffusion.weight.device)
            elif is_ip_adapter:
                processor = IPAdapterAttnProcessor2_0(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    num_tokens=self.processor.num_tokens,
                    scale=self.processor.scale,
                )
                processor.load_state_dict(self.processor.state_dict())
                if hasattr(self.processor, "to_k_ip"):
                    processor.to(
                        device=self.processor.to_k_ip[0].weight.device, dtype=self.processor.to_k_ip[0].weight.dtype
                    )
            else:
                # set attention processor
                # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
                # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
                # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
                processor = (
                    AttnProcessor2_0()
                    if hasattr(F, "scaled_dot_product_attention") and self.scale_qk
                    else AttnProcessor()
                )

        self.set_processor(processor)

    def set_attention_slice(self, slice_size: int) -> None:
        r"""
        Set the slice size for attention computation.

        Args:
            slice_size (`int`):
                The slice size for attention computation.
        """
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        if slice_size is not None and self.added_kv_proj_dim is not None:
            processor = SlicedAttnAddedKVProcessor(slice_size)
        elif slice_size is not None:
            processor = SlicedAttnProcessor(slice_size)
        elif self.added_kv_proj_dim is not None:
            processor = AttnAddedKVProcessor()
        else:
            # set attention processor
            # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
            # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
            # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )

        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor") -> None:
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
        """
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.warning(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._modules.pop("processor")

        self.processor = processor

    def get_processor(self, return_deprecated_lora: bool = False) -> "AttentionProcessor":
        r"""
        Get the attention processor in use.

        Args:
            return_deprecated_lora (`bool`, *optional*, defaults to `False`):
                Set to `True` to return the deprecated LoRA attention processor.

        Returns:
            "AttentionProcessor": The attention processor in use.
        """
        if not return_deprecated_lora:
            return self.processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        if tensor.ndim == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
        else:
            batch_size, extra_dim, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

        return tensor

    def get_attention_scores(
        self, query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    def norm_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`torch.Tensor`): Hidden states of the encoder.

        Returns:
            `torch.Tensor`: The normalized encoder hidden states.
        """
        assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        else:
            assert False

        return encoder_hidden_states

    @torch.no_grad()
    def fuse_projections(self, fuse=True):
        device = self.to_q.weight.data.device
        dtype = self.to_q.weight.data.dtype

        if not self.is_cross_attention:
            # fetch weight matrices.
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            # create a new single projection layer and copy over the weights.
            self.to_qkv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
            self.to_qkv.weight.copy_(concatenated_weights)
            if self.use_bias:
                concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
                self.to_qkv.bias.copy_(concatenated_bias)

        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            self.to_kv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
            self.to_kv.weight.copy_(concatenated_weights)
            if self.use_bias:
                concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
                self.to_kv.bias.copy_(concatenated_bias)

        # handle added projections for SD3 and others.
        if (
            getattr(self, "add_q_proj", None) is not None
            and getattr(self, "add_k_proj", None) is not None
            and getattr(self, "add_v_proj", None) is not None
        ):
            concatenated_weights = torch.cat(
                [self.add_q_proj.weight.data, self.add_k_proj.weight.data, self.add_v_proj.weight.data]
            )
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            self.to_added_qkv = nn.Linear(
                in_features, out_features, bias=self.added_proj_bias, device=device, dtype=dtype
            )
            self.to_added_qkv.weight.copy_(concatenated_weights)
            if self.added_proj_bias:
                concatenated_bias = torch.cat(
                    [self.add_q_proj.bias.data, self.add_k_proj.bias.data, self.add_v_proj.bias.data]
                )
                self.to_added_qkv.bias.copy_(concatenated_bias)

        self.fused_projections = fuse


class AttnProcessor2_0:
    r"""Modified from AttnProcessor2_0 in diffusers.models.attention_processor:
            from diffusers.models.attention_processor import AttnProcessor2_0
        Modified from HunyuanAttnProcessor2_0 in diffusers.models.attention_processor:
            from diffusers.models.attention_processor import HunyuanAttnProcessor2_0
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        logger.debug(f"MQA/GQA before query.shape: {query.shape}")
        logger.debug(f"MQA/GQA before key.shape: {key.shape}")
        logger.debug(f"MQA/GQA before value.shape: {value.shape}")


        # * MQA or GQA
        # ! https://cloud.tencent.com/developer/ask/sof/106627779
        head_dim = int(query.shape[-1]) // attn.heads
        kv_heads = int(key.shape[-1]) // head_dim
        logger.debug(f"query.shape[-1]: {query.shape[-1]} int(query.shape[-1]): {int(query.shape[-1])}")
        logger.debug(f"key.shape[-1]: {key.shape[-1]} int(key.shape[-1]): {int(key.shape[-1])}")
        logger.debug(f"kv_heads: {kv_heads} head_dim: {head_dim}")

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # * MQA or GQA
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        logger.debug(f"MQA/GQA after query.shape: {query.shape}")
        logger.debug(f"MQA/GQA after key.shape: {key.shape}")
        logger.debug(f"MQA/GQA after value.shape: {value.shape}")

        if kv_heads != attn.heads:
            logger.debug(f"kv_heads != attn.heads: {kv_heads} != {attn.heads}")
            logger.debug("MQA or GQA")
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            # ! Better not to use repeat_interleave.
            # ! https://github.com/pytorch/pytorch/issues/128505
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
            value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
            logger.debug(f"Norm Query: {query.shape}, {attn.norm_q}")
        if attn.norm_k is not None:
            key = attn.norm_k(key)
            logger.debug(f"Norm Key: {key.shape}, {attn.norm_k}")
        
        if image_rotary_emb is not None:
            S_rope = image_rotary_emb[0].shape[0]
            S = query.shape[2]
            if S_rope < S:
                q_prefix = apply_rotary_emb(query[:, :, :S_rope, :], image_rotary_emb)
                q_suffix = apply_rotary_emb(query[:, :, S_rope:, :], image_rotary_emb)
                query = torch.cat([q_prefix, q_suffix], dim=2)
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                if S_rope < S:
                    k_prefix = apply_rotary_emb(key[:, :, :S_rope, :], image_rotary_emb)
                    k_suffix = apply_rotary_emb(key[:, :, S_rope:, :], image_rotary_emb)
                    key = torch.cat([k_prefix, k_suffix], dim=2)
                else:
                    key = apply_rotary_emb(key, image_rotary_emb)
        
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class XFormersAttnProcessor:
    r"""Modified from XFormersAttnProcessor in diffusers.models.attention_processor:
            from diffusers.models.attention_processor import XFormersAttnProcessor
    """

    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        logger.debug(f"hidden_states dtype: {hidden_states.dtype}")
        logger.debug(f"encoder_hidden_states dtype: {encoder_hidden_states.dtype if encoder_hidden_states is not None else None}")
        logger.debug(f"temb dtype: {temb.dtype if temb is not None else None}")
        logger.debug(f"image_rotary_emb[0] dtype: {image_rotary_emb[0].dtype if image_rotary_emb is not None else None}")
        logger.debug(f"image_rotary_emb[1] dtype: {image_rotary_emb[1].dtype if image_rotary_emb is not None else None}")

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        logger.debug(f"MQA/GQA before query.shape: {query.shape}")
        logger.debug(f"MQA/GQA before key.shape: {key.shape}")
        logger.debug(f"MQA/GQA before value.shape: {value.shape}")

        # * MQA or GQA
        # ! https://cloud.tencent.com/developer/ask/sof/106627779
        head_dim = int(query.shape[-1]) // attn.heads
        kv_heads = int(key.shape[-1]) // head_dim
        logger.debug(f"query.shape[-1]: {query.shape[-1]} int(query.shape[-1]): {int(query.shape[-1])}")
        logger.debug(f"key.shape[-1]: {key.shape[-1]} int(key.shape[-1]): {int(key.shape[-1])}")
        logger.debug(f"kv_heads: {kv_heads} head_dim: {head_dim}")

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # * MQA or GQA
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        logger.debug(f"MQA/GQA after query.shape: {query.shape}")
        logger.debug(f"MQA/GQA after key.shape: {key.shape}")
        logger.debug(f"MQA/GQA after value.shape: {value.shape}")

        if kv_heads != attn.heads:
            logger.debug(f"kv_heads != attn.heads: {kv_heads} != {attn.heads}")
            logger.debug("MQA or GQA")
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            # ! Better not to use repeat_interleave.
            # ! https://github.com/pytorch/pytorch/issues/128505
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
            value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
            logger.debug(f"Norm Query: {query.shape}, {attn.norm_q}")
        if attn.norm_k is not None:
            key = attn.norm_k(key)
            logger.debug(f"Norm Key: {key.shape}, {attn.norm_k}")

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            logger.debug("RoPE applied")
            logger.debug(f"len(image_rotary_emb): {len(image_rotary_emb)}")
            logger.debug(f"image_rotary_emb[0].shape: {image_rotary_emb[0].shape}")
            logger.debug(f"image_rotary_emb[1].shape: {image_rotary_emb[1].shape}")
            logger.debug(f"Rotary Query: {query.shape}")
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)
                logger.debug(f"Rotary Key: {key.shape}")
        else:
            logger.debug(f"No Rotary: {query.shape}, {key.shape}")

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


ATTENTION_MODULES = {
    "attention": Attention,
}


def get_attention_module(attention_module: str, *args, **kwargs) -> nn.Module:
    """
    Get the attention module class based on the attention module name.

    Args:
        attention_module (`str`):
            The name of the attention module to use.
        *args: Positional arguments to pass to the attention module.
        **kwargs: Keyword arguments to pass to the attention module.

    Returns:
        `nn.Module`: The attention module class.
    """
    attention_module = attention_module.lower()
    if attention_module in ATTENTION_MODULES:
        return ATTENTION_MODULES[attention_module](*args, **kwargs)
    else:
        raise ValueError(f"Unknown attention module '{attention_module}'. Supported modules are {list(ATTENTION_MODULES.keys())}.")


ATTENTION_PROCESSORS = {
    "attnprocessor2_0": AttnProcessor2_0(),
}


def get_attention_processor(processor_type: Optional[str]) -> Optional[AttentionProcessor]:
    """
    Get the attention processor class based on the processor name.

    Args:
        processor (`str`, *optional*):
            The name of the attention processor to use. If `None`, defaults to `AttnProcessor2_0`.

    Returns:
        `AttentionProcessor`: The attention processor class.
    """
    processor_type = processor_type.lower()
    if processor_type in ATTENTION_PROCESSORS:
        return ATTENTION_PROCESSORS[processor_type]
    else:
        raise ValueError(f"Unknown attention processor '{processor_type}'. Supported processors are {list(ATTENTION_PROCESSORS.keys())}.")
