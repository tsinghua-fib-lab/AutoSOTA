from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import sys
import os.path as osp
import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin, UNet2DConditionLoadersMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.utils import USE_PEFT_BACKEND, deprecate, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    FusedAttnProcessor2_0,
)
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    GLIGENTextBoundingboxProjection,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
    get_2d_rotary_pos_embed,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

# Setup logger
logger = logging.getLogger(__name__)

# Setup working directory
WORK_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../../../.."))
logger.info(f"Working directory: {WORK_DIR}")
if WORK_DIR not in sys.path:
    logger.warning(f"Working directory ({WORK_DIR}) is not in sys.path. Adding it.")
    sys.path.append(WORK_DIR)


from Mobile_VTON.models.unets.unet_2d_blocks_garment import (
    get_down_block,
    get_mid_block,
    get_up_block,
)
from Mobile_VTON.models.activations import get_activation
from Mobile_VTON.models.convolutions import get_convolution_module
from Mobile_VTON.models.embeddings import get_time_text_embedding_module, get_time_embedding_module, get_text_projection_module

import torch.nn.functional as F



def resize_tensor(x, size=(1024, 1024), align_corners=False):
    """
    Resize a tensor to the given size using bilinear interpolation.
    The input tensor is expected to be in NCHW format.
    """
    x_dim = x.dim()
    if x_dim == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif x_dim == 3:
        x = x.unsqueeze(0)  # [1, C, H, W]
    elif x_dim != 4:
        raise ValueError("Input tensor must be 2D, 3D or 4D.")

    x_resized = F.interpolate(x, size=size, mode='bicubic', align_corners=align_corners)
    
    if x_dim == 2:
        x_resized = x_resized.squeeze(0).squeeze(0)  # [1024, 1024]
    elif x_dim == 3:
        x_resized = x_resized.squeeze(0)  # [C, 1024, 1024]
    elif x_dim != 4:
        raise ValueError("Input tensor must be 2D, 3D or 4D.")

    return x_resized

class UNet2DConditionModel(
    ModelMixin, ConfigMixin, FromOriginalModelMixin, UNet2DConditionLoadersMixin, PeftAdapterMixin
):
    r"""Modified from UNet2DConditionModel in diffusers.models.unets.unet_2d_condition:
            from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
        # ! SD3.5Large Timestep Embedding Style
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D", "CrossAttnUpBlock2D", "Transformer2DModel", "ResnetBlock2D", "CrossAttnUpBlock2D"]

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        height: Optional[Union[int, Tuple[int, int]]] = None,
        width: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads: int = 64,
        #! ConvIn
        conv_in_module: str = "Conv2d",
        conv_in_dw_bias: bool = True,
        conv_in_pw_bias: bool = False,
        #! TimeTextEmbedding
        time_text_embedding_mode: str = "default",
        time_text_embedding_module: str = "CombinedTimestepTextProjEmbeddings",
        time_text_embedding_time_embed_dim: int = 896,
        time_text_embedding_pooled_projection_dim: int = 2048,
        time_text_embedding_act_fn: Optional[str] = "hardswish",
        #! ContextEmbedding
        context_embedding_text_embedding_dim: int = 4096,
        context_embedding_caption_projection_dim: int = 2048,
        #! Iime Embedding
        time_embedding_module: str = "TimestepEmbedding",
        #! ConvOut
        conv_out_module: str = "Conv2d",
        conv_out_dw_bias: bool = True,
        conv_out_pw_bias: bool = False,
        #! Shared
        #! ResNet
        resnet_module: Optional[str] = "ResnetBlock2D",
        resnet_conv_module: Optional[str] = "SepConv2d",
        #! Transformer
        transformer2d_model_type: Optional[str] = "Transformer2DModel",
        transformer_block_type: Optional[str] = "BasicTransformerBlock",
        attn_module: Optional[str] = "Attention",
        attn_processor_type: Optional[str] = "AttnProcessor2_0",
        #! Downsample
        downsample_module: Optional[str] = "Downsample2D",
        downsample_conv_module: Optional[str] = "SepConv2d",
        #! Upsample
        upsample_module: Optional[str] = "Upsample2D",
        upsample_conv_module: Optional[str] = "SepConv2d",
        #! Down
        down_block_resnet_middle_expansion_type: Optional[str] = "max",  # "input", "max"
        down_block_resnet_middle_expansion: Optional[Union[int, Tuple[int]]] = None,
        down_block_kv_heads: Optional[Union[int, Tuple[int]]] = None,
        down_block_qk_norm: Optional[Union[str, Tuple[str]]] = None,
        down_block_ff_mult: Optional[Union[int, Tuple[int]]] = 4,
        down_block_use_self_attention: Optional[Union[bool, Tuple[bool]]] = False,
        down_block_resnet_dw_bias: Optional[bool] = True,
        down_block_resnet_pw_bias: Optional[bool] = False,
        #! Mid
        mid_block_resnet_middle_expansion_type: Optional[str] = "max",  # "input", "max"
        mid_block_resnet_middle_expansion: Optional[Union[int, Tuple[int]]] = None,
        mid_block_kv_heads: Optional[Union[int, Tuple[int]]] = None,
        mid_block_qk_norm: Optional[Union[str, Tuple[str]]] = None,
        mid_block_ff_mult: Optional[Union[int, Tuple[int]]] = 4,
        mid_block_use_self_attention: Optional[Union[bool, Tuple[bool]]] = True,
        mid_block_resnet_dw_bias: Optional[bool] = True,
        mid_block_resnet_pw_bias: Optional[bool] = False,
        mid_block_use_additional_resnet: Optional[bool] = True,
        #! Up
        up_block_resnet_middle_expansion_type: Optional[str] = "max",  # "input", "max"
        up_block_resnet_middle_expansion: Optional[Union[int, Tuple[int]]] = None,
        up_block_kv_heads: Optional[Union[int, Tuple[int]]] = None,
        up_block_qk_norm: Optional[Union[str, Tuple[str]]] = None,
        up_block_ff_mult: Optional[Union[int, Tuple[int]]] = 4,
        up_block_use_self_attention: Optional[Union[bool, Tuple[bool]]] = False,
        up_block_resnet_dw_bias: Optional[bool] = True,
        up_block_resnet_pw_bias: Optional[bool] = False,
        up_block_receive_additional_residuals: Optional[Union[bool, Tuple[bool]]] = True,
        #! RoPE
        use_rope: bool = False,
        #! Pooled Projection
        use_pooled_projection: bool = False,
        pooled_projection_dim: int = 2048,  # clip+clip
    ):
        super().__init__()
        if num_attention_heads is not None:
            raise ValueError(
                "At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19."
            )

        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = num_attention_heads or attention_head_dim

        # Check inputs
        self._check_config(
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
        )

        # input
        conv_in_padding = (conv_in_kernel - 1) // 2
        conv_in_module = conv_in_module.lower()
        conv_in_kwargs = dict(
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            kernel_size=conv_in_kernel,
            stride=1,
            padding=conv_in_padding,
        )
        if conv_in_module == "conv2d":
            pass
        elif conv_in_module == "sepconv2d":
            conv_in_kwargs.update(dict(
                dw_groups=in_channels,
                dw_bias=conv_in_dw_bias,
                pw_bias=conv_in_pw_bias,
            ))
        else:
            raise NotImplementedError(f"`conv_in_module` {conv_in_module} not implemented. Please use `Conv2d` or `SepConv2d`.")

        self.conv_in = get_convolution_module(conv_in_module, **conv_in_kwargs)
            
        # time
        self.use_pooled_projection = use_pooled_projection
        time_text_embedding_mode = time_text_embedding_mode.lower()
        self.time_text_embedding_mode = time_text_embedding_mode
        if time_text_embedding_mode == "default":
            # * SDXL Style
            logger.warning(f"UNet2DConditionModel::__init__ - when set time_text_embedding_mode to {time_text_embedding_mode}, the following parameters are ignored: time_text_embedding_module({time_text_embedding_module}), time_text_embedding_time_embed_dim({time_text_embedding_time_embed_dim}), time_text_embedding_pooled_projection_dim({time_text_embedding_pooled_projection_dim}), time_text_embedding_act_fn({time_text_embedding_act_fn}), context_embedding_text_embedding_dim({context_embedding_text_embedding_dim}), context_embedding_caption_projection_dim({context_embedding_caption_projection_dim})")

            self.time_text_embed = None
            self.context_embedder = None

            time_embed_dim, timestep_input_dim = self._set_time_proj(
                time_embedding_type,
                block_out_channels=block_out_channels,
                flip_sin_to_cos=flip_sin_to_cos,
                freq_shift=freq_shift,
                time_embedding_dim=time_embedding_dim,
            )

            time_embedding_module = time_embedding_module.lower()
            time_embedding_kwargs = dict(
                in_channels=timestep_input_dim,
                time_embed_dim=time_embed_dim,
                act_fn=act_fn,
                post_act_fn=timestep_post_act,
                cond_proj_dim=time_cond_proj_dim,
            )
            self.time_embedding = get_time_embedding_module(time_embedding_module, **time_embedding_kwargs)

            self._set_encoder_hid_proj(
                encoder_hid_dim_type,
                cross_attention_dim=cross_attention_dim,
                encoder_hid_dim=encoder_hid_dim,
            )

            # class embedding
            self._set_class_embedding(
                class_embed_type,
                act_fn=act_fn,
                num_class_embeds=num_class_embeds,
                projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
                time_embed_dim=time_embed_dim,
                timestep_input_dim=timestep_input_dim,
            )

            self._set_add_embedding(
                addition_embed_type,
                addition_embed_type_num_heads=addition_embed_type_num_heads,
                addition_time_embed_dim=addition_time_embed_dim,
                cross_attention_dim=cross_attention_dim,
                encoder_hid_dim=encoder_hid_dim,
                flip_sin_to_cos=flip_sin_to_cos,
                freq_shift=freq_shift,
                projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
                time_embed_dim=time_embed_dim,
            )

            if time_embedding_act_fn is None:
                self.time_embed_act = None
            else:
                self.time_embed_act = get_activation(time_embedding_act_fn)

            if use_pooled_projection:
                assert pooled_projection_dim is not None, "pooled_projection_dim must be specified when use_pooled_projection is True"
                text_projection_module = "PixArtAlphaTextProjection"
                text_projection_module = text_projection_module.lower()
                text_projection_kwargs = dict(
                    in_features=pooled_projection_dim,
                    hidden_size=time_embed_dim,
                    act_fn=act_fn,
                )
                self.text_projection = get_text_projection_module(text_projection_module, **text_projection_kwargs)
            else:
                self.text_projection = None

        elif time_text_embedding_mode == "combined":
            # * SD3.5 Large Style

            time_text_embedding_module = time_text_embedding_module.lower()
            time_and_text_embedding_kwargs = dict(
                embedding_dim=time_text_embedding_time_embed_dim,
                pooled_projection_dim=time_text_embedding_pooled_projection_dim,
            )
            if time_text_embedding_module == "combinedtimesteptextprojembeddings":
                time_and_text_embedding_kwargs.update(dict(
                    act_fn=time_text_embedding_act_fn,
                    use_pooled_projection=use_pooled_projection,
                ))
            self.time_text_embed = get_time_text_embedding_module(time_text_embedding_module, **time_and_text_embedding_kwargs)

            self.context_embedder = nn.Linear(
                context_embedding_text_embedding_dim,
                context_embedding_caption_projection_dim
            )

            assert context_embedding_caption_projection_dim == cross_attention_dim, "context embedding caption projection dim must match cross attention dim"

            logger.warning(f"UNet2DConditionModel::__init__ - when set time_text_embedding_mode to {time_text_embedding_mode}, the following parameters are ignored: time_embedding_type({time_embedding_type}), encoder_hid_dim_type({encoder_hid_dim_type}), class_embed_type({class_embed_type}), addition_embed_type({addition_embed_type}), time_embedding_act_fn({time_embedding_act_fn})")

            self.time_embedding = None
            self.encoder_hid_proj = None
            self.class_embedding = None
            self.add_embedding = None
            self.add_time_proj = None
            self.time_embed_act = None

            time_embed_dim = time_text_embedding_time_embed_dim

            if class_embeddings_concat:
                logger.warning(f"UNet2DConditionModel::__init__ - when set time_text_embedding_mode to {time_text_embedding_mode}, class_embeddings_concat({class_embeddings_concat}) should be set to False. Setting to False.")
                class_embeddings_concat = False
        else:
            raise ValueError(f"Unsupported `time_text_embedding_mode`: {time_text_embedding_mode}. Supported modes are: `default`, `combined`.")

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention

            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        if class_embeddings_concat:
            # The time embeddings are concatenated with the class embeddings. The dimension of the
            # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
            # regular time embeddings
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        # down
        if isinstance(down_block_resnet_middle_expansion_type, str):
            down_block_resnet_middle_expansion_type = [down_block_resnet_middle_expansion_type] * len(down_block_types)
        assert len(down_block_resnet_middle_expansion_type) == len(down_block_types), \
            f"down_resnet_middle_expansion_type should be of length {len(down_block_types)}, but is {len(down_block_resnet_middle_expansion_type)}"

        if isinstance(down_block_resnet_middle_expansion, int) or down_block_resnet_middle_expansion is None:
            down_block_resnet_middle_expansion = [down_block_resnet_middle_expansion] * len(down_block_types)
        assert len(down_block_resnet_middle_expansion) == len(down_block_types), \
            f"down_block_resnet_middle_expansion should be of length {len(down_block_types)}, but is {len(down_block_resnet_middle_expansion)}"

        if isinstance(down_block_kv_heads, int) or down_block_kv_heads is None:
            down_block_kv_heads = [down_block_kv_heads] * len(down_block_types)
        assert len(down_block_kv_heads) == len(down_block_types), \
            f"down_kv_heads should be of length {len(down_block_types)}, but is {len(down_block_kv_heads)}"

        if isinstance(down_block_qk_norm, str) or down_block_qk_norm is None:
            down_block_qk_norm = [down_block_qk_norm] * len(down_block_types)
        assert len(down_block_qk_norm) == len(down_block_types), \
            f"down_qk_norm should be of length {len(down_block_types)}, but is {len(down_block_qk_norm)}"

        if isinstance(down_block_ff_mult, int) or down_block_ff_mult is None:
            down_block_ff_mult = [down_block_ff_mult] * len(down_block_types)
        assert len(down_block_ff_mult) == len(down_block_types), \
            f"down_ff_mult should be of length {len(down_block_types)}, but is {len(down_block_ff_mult)}"

        if isinstance(down_block_use_self_attention, bool):
            down_block_use_self_attention = [down_block_use_self_attention] * len(down_block_types)
        assert len(down_block_use_self_attention) == len(down_block_types), \
            f"down_use_self_attention should be of length {len(down_block_types)}, but is {len(down_block_use_self_attention)}"

        if isinstance(down_block_resnet_dw_bias, bool):
            down_block_resnet_dw_bias = [down_block_resnet_dw_bias] * len(down_block_types)
        assert len(down_block_resnet_dw_bias) == len(down_block_types), \
            f"down_resnet_dw_bias should be of length {len(down_block_types)}, but is {len(down_block_resnet_dw_bias)}"

        if isinstance(down_block_resnet_pw_bias, bool):
            down_block_resnet_pw_bias = [down_block_resnet_pw_bias] * len(down_block_types)
        assert len(down_block_resnet_pw_bias) == len(down_block_types), \
            f"down_resnet_pw_bias should be of length {len(down_block_types)}, but is {len(down_block_resnet_pw_bias)}"

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
                resnet_middle_expansion_type=down_block_resnet_middle_expansion_type[i],
                resnet_middle_expansion=down_block_resnet_middle_expansion[i],
                resnet_module=resnet_module,
                resnet_conv_module=resnet_conv_module,
                downsample_module=downsample_module,
                downsample_conv_module=downsample_conv_module,
                transformer2d_model_type=transformer2d_model_type,
                transformer_block_type=transformer_block_type,
                attn_module=attn_module,
                attn_processor_type=attn_processor_type,
                kv_heads=down_block_kv_heads[i],
                qk_norm=down_block_qk_norm[i],
                ff_mult=down_block_ff_mult[i],
                use_self_attention=down_block_use_self_attention[i],
                resnet_dw_bias=down_block_resnet_dw_bias[i],
                resnet_pw_bias=down_block_resnet_pw_bias[i],
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = get_mid_block(
            mid_block_type,
            temb_channels=blocks_time_embed_dim,
            in_channels=block_out_channels[-1],
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            output_scale_factor=mid_block_scale_factor,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim[-1],
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
            resnet_skip_time_act=resnet_skip_time_act,
            cross_attention_norm=cross_attention_norm,
            attention_head_dim=attention_head_dim[-1],
            dropout=dropout,
            resnet_middle_expansion_type=mid_block_resnet_middle_expansion_type,
            resnet_middle_expansion=mid_block_resnet_middle_expansion,
            resnet_module=resnet_module,
            resnet_conv_module=resnet_conv_module,
            transformer2d_model_type=transformer2d_model_type,
            transformer_block_type=transformer_block_type,
            attn_module=attn_module,
            attn_processor_type=attn_processor_type,
            kv_heads=mid_block_kv_heads,
            qk_norm=mid_block_qk_norm,
            ff_mult=mid_block_ff_mult,
            use_self_attention=mid_block_use_self_attention,
            resnet_dw_bias=mid_block_resnet_dw_bias,
            resnet_pw_bias=mid_block_resnet_pw_bias,
            use_additional_resnet=mid_block_use_additional_resnet,
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        if isinstance(up_block_resnet_middle_expansion_type, str):
            up_block_resnet_middle_expansion_type = [up_block_resnet_middle_expansion_type] * len(up_block_types)
        assert len(up_block_resnet_middle_expansion_type) == len(up_block_types), \
            f"up_block_resnet_middle_expansion_type should be of length {len(up_block_types)}, but is {len(up_block_resnet_middle_expansion_type)}"

        if isinstance(up_block_resnet_middle_expansion, int) or up_block_resnet_middle_expansion is None:
            up_block_resnet_middle_expansion = [up_block_resnet_middle_expansion] * len(up_block_types)
        assert len(up_block_resnet_middle_expansion) == len(up_block_types), \
            f"up_block_resnet_middle_expansion should be of length {len(up_block_types)}, but is {len(up_block_resnet_middle_expansion)}"

        if isinstance(up_block_kv_heads, int) or up_block_kv_heads is None:
            up_block_kv_heads = [up_block_kv_heads] * len(up_block_types)
        assert len(up_block_kv_heads) == len(up_block_types), \
            f"up_block_kv_heads should be of length {len(up_block_types)}, but is {len(up_block_kv_heads)}"

        if isinstance(up_block_qk_norm, str) or up_block_qk_norm is None:
            up_block_qk_norm = [up_block_qk_norm] * len(up_block_types)
        assert len(up_block_qk_norm) == len(up_block_types), \
            f"up_block_qk_norm should be of length {len(up_block_types)}, but is {len(up_block_qk_norm)}"

        if isinstance(up_block_ff_mult, int) or up_block_ff_mult is None:
            up_block_ff_mult = [up_block_ff_mult] * len(up_block_types)
        assert len(up_block_ff_mult) == len(up_block_types), \
            f"up_block_ff_mult should be of length {len(up_block_types)}, but is {len(up_block_ff_mult)}"

        if isinstance(up_block_use_self_attention, bool):
            up_block_use_self_attention = [up_block_use_self_attention] * len(up_block_types)
        assert len(up_block_use_self_attention) == len(up_block_types), \
            f"up_block_use_self_attention should be of length {len(up_block_types)}, but is {len(up_block_use_self_attention)}"

        if isinstance(up_block_resnet_dw_bias, bool):
            up_block_resnet_dw_bias = [up_block_resnet_dw_bias] * len(up_block_types)
        assert len(up_block_resnet_dw_bias) == len(up_block_types), \
            f"up_block_resnet_dw_bias should be of length {len(up_block_types)}, but is {len(up_block_resnet_dw_bias)}"

        if isinstance(up_block_resnet_pw_bias, bool):
            up_block_resnet_pw_bias = [up_block_resnet_pw_bias] * len(up_block_types)
        assert len(up_block_resnet_pw_bias) == len(up_block_types), \
            f"up_block_resnet_pw_bias should be of length {len(up_block_types)}, but is {len(up_block_resnet_pw_bias)}"

        if isinstance(up_block_receive_additional_residuals, bool):
            up_block_receive_additional_residuals = [up_block_receive_additional_residuals] * len(up_block_types)
        assert len(up_block_receive_additional_residuals) == len(up_block_types), \
            f"up_block_receive_additional_residuals should be of length {len(up_block_types)}, but is {len(up_block_receive_additional_residuals)}"

        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block))
            if reverse_transformer_layers_per_block is None
            else reverse_transformer_layers_per_block
        )
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resolution_idx=i,
                resnet_groups=norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
                resnet_middle_expansion_type=up_block_resnet_middle_expansion_type[i],
                resnet_middle_expansion=up_block_resnet_middle_expansion[i],
                resnet_module=resnet_module,
                resnet_conv_module=resnet_conv_module,
                upsample_module=upsample_module,
                upsample_conv_module=upsample_conv_module,
                transformer2d_model_type=transformer2d_model_type,
                transformer_block_type=transformer_block_type,
                attn_module=attn_module,
                attn_processor_type=attn_processor_type,
                kv_heads=up_block_kv_heads[i],
                qk_norm=up_block_qk_norm[i],
                ff_mult=up_block_ff_mult[i],
                use_self_attention=up_block_use_self_attention[i],
                resnet_dw_bias=up_block_resnet_dw_bias[i],
                resnet_pw_bias=up_block_resnet_pw_bias[i],
                receive_additional_residuals=up_block_receive_additional_residuals[i],
            )
            self.up_blocks.append(up_block)

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )
            self.conv_act = get_activation(act_fn)
        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        conv_out_module = conv_out_module.lower()
        conv_out_kwargs = dict(
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            kernel_size=conv_out_kernel,
            stride=1,
            padding=conv_out_padding,
        )
        if conv_out_module == "conv2d":
            pass
        elif conv_out_module == "sepconv2d":
            conv_out_kwargs.update(dict(
                dw_groups=block_out_channels[0],
                dw_bias=conv_out_dw_bias,
                pw_bias=conv_out_pw_bias,
            ))
        else:
            raise NotImplementedError(f"`conv_out_module` {conv_out_module} not implemented. Please use `Conv2d` or `SepConv2d`.")
        self.conv_out = get_convolution_module(conv_out_module, **conv_out_kwargs)

        self._set_pos_net_if_use_gligen(attention_type=attention_type, cross_attention_dim=cross_attention_dim)

        self.use_rope = use_rope
        self.set_sample_size(sample_size)
        if height and width:
            self.set_sample_size_rsd(height // 8, width // 8)

    def _check_config(
        self,
        down_block_types: Tuple[str],
        up_block_types: Tuple[str],
        only_cross_attention: Union[bool, Tuple[bool]],
        block_out_channels: Tuple[int],
        layers_per_block: Union[int, Tuple[int]],
        cross_attention_dim: Union[int, Tuple[int]],
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple[int]]],
        reverse_transformer_layers_per_block: bool,
        attention_head_dim: int,
        num_attention_heads: Optional[Union[int, Tuple[int]]],
    ):
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )
        if isinstance(transformer_layers_per_block, list) and reverse_transformer_layers_per_block is None:
            for layer_number_per_block in transformer_layers_per_block:
                if isinstance(layer_number_per_block, list):
                    raise ValueError("Must provide 'reverse_transformer_layers_per_block` if using asymmetrical UNet.")

    def _set_time_proj(
        self,
        time_embedding_type: str,
        block_out_channels: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: int,
    ) -> Tuple[int, int]:
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        return time_embed_dim, timestep_input_dim

    def _set_encoder_hid_proj(
        self,
        encoder_hid_dim_type: Optional[str],
        cross_attention_dim: Union[int, Tuple[int]],
        encoder_hid_dim: Optional[int],
    ):
        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
            )

        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_image_proj":
            # image_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image_proj"` (Kandinsky 2.1)`
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2
            self.encoder_hid_proj = ImageProjection(
                image_embed_dim=encoder_hid_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim_type`: {encoder_hid_dim_type} must be None, 'text_proj', 'text_image_proj', or 'image_proj'."
            )
        else:
            self.encoder_hid_proj = None

    def _set_class_embedding(
        self,
        class_embed_type: Optional[str],
        act_fn: str,
        num_class_embeds: Optional[int],
        projection_class_embeddings_input_dim: Optional[int],
        time_embed_dim: int,
        timestep_input_dim: int,
    ):
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif class_embed_type == "simple_projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'simple_projection' requires `projection_class_embeddings_input_dim` be set"
                )
            self.class_embedding = nn.Linear(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

    def _set_add_embedding(
        self,
        addition_embed_type: str,
        addition_embed_type_num_heads: int,
        addition_time_embed_dim: Optional[int],
        flip_sin_to_cos: bool,
        freq_shift: float,
        cross_attention_dim: Optional[int],
        encoder_hid_dim: Optional[int],
        projection_class_embeddings_input_dim: Optional[int],
        time_embed_dim: int,
    ):
        if addition_embed_type == "text":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim

            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
        elif addition_embed_type == "text_image":
            # text_embed_dim and image_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image"` (Kandinsky 2.1)`
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif addition_embed_type == "image":
            # Kandinsky 2.2
            self.add_embedding = ImageTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        elif addition_embed_type == "image_hint":
            # Kandinsky 2.2 ControlNet
            self.add_embedding = ImageHintTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        elif addition_embed_type is not None:
            raise ValueError(
                f"`addition_embed_type`: {addition_embed_type} must be None, 'text', 'text_image', 'text_time', 'image', or 'image_hint'."
            )

    def _set_pos_net_if_use_gligen(self, attention_type: str, cross_attention_dim: int):
        if attention_type in ["gated", "gated-text-image"]:
            positive_len = 768
            if isinstance(cross_attention_dim, int):
                positive_len = cross_attention_dim
            elif isinstance(cross_attention_dim, (list, tuple)):
                positive_len = cross_attention_dim[0]

            feature_type = "text-only" if attention_type == "gated" else "text-image"
            self.position_net = GLIGENTextBoundingboxProjection(
                positive_len=positive_len, out_dim=cross_attention_dim, feature_type=feature_type
            )

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def set_attention_slice(self, slice_size: Union[str, int, List[int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module splits the input tensor in slices to compute attention in
        several steps. This is useful for saving some memory in exchange for a small decrease in speed.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism from https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stage blocks where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of values that
        are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        for i, upsample_block in enumerate(self.up_blocks):
            setattr(upsample_block, "s1", s1)
            setattr(upsample_block, "s2", s2)
            setattr(upsample_block, "b1", b1)
            setattr(upsample_block, "b2", b2)

    def disable_freeu(self):
        """Disables the FreeU mechanism."""
        freeu_keys = {"s1", "s2", "b1", "b2"}
        for i, upsample_block in enumerate(self.up_blocks):
            for k in freeu_keys:
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    setattr(upsample_block, k, None)

    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedAttnProcessor2_0())

    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def get_time_text_embed(
        self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], pooled_projections: Optional[torch.Tensor]
    ):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        emb = self.time_text_embed(timesteps, pooled_projections, weight_type=sample.dtype)

        return emb

    def get_time_embed(
        self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]
    ) -> Optional[torch.Tensor]:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)
        return t_emb

    def get_class_embed(self, sample: torch.Tensor, class_labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        class_emb = None
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
        return class_emb

    def get_aug_embed(
        self, emb: torch.Tensor, encoder_hidden_states: torch.Tensor, added_cond_kwargs: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        aug_emb = None
        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            # SDXL - style
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            # Kandinsky 2.2 ControlNet - style
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb = self.add_embedding(image_embs, hint)
        return aug_emb

    def process_encoder_hidden_states(
        self, encoder_hidden_states: torch.Tensor, added_cond_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            if hasattr(self, "text_encoder_hid_proj") and self.text_encoder_hid_proj is not None:
                encoder_hidden_states = self.text_encoder_hid_proj(encoder_hidden_states)

            image_embeds = added_cond_kwargs.get("image_embeds")
            image_embeds = self.encoder_hid_proj(image_embeds)
            encoder_hidden_states = (encoder_hidden_states, image_embeds)
        return encoder_hidden_states

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        pooled_projections: torch.FloatTensor = None,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        if self.use_pooled_projection:
            assert pooled_projections is not None, "pooled_projections must be provided when using set_use_pooled_projection=True."
        else:
            if pooled_projections is not None:
                logger.warning(
                    "pooled_projections is provided but set_use_pooled_projection=False. Ignoring pooled_projections."
                )
                pooled_projections = None
        if self.time_text_embedding_mode == "default":
            # * SDXL Style
            t_emb = self.get_time_embed(sample=sample, timestep=timestep)
            emb = self.time_embedding(t_emb, timestep_cond)

            class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
            if class_emb is not None:
                if self.config.class_embeddings_concat:
                    emb = torch.cat([emb, class_emb], dim=-1)
                else:
                    emb = emb + class_emb

            aug_emb = self.get_aug_embed(
                emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
            )
            if self.config.addition_embed_type == "image_hint":
                aug_emb, hint = aug_emb
                sample = torch.cat([sample, hint], dim=1)

            emb = emb + aug_emb if aug_emb is not None else emb

            if self.use_pooled_projection:
                pooled_projections_emb = self.text_projection(pooled_projections)
                emb = emb + pooled_projections_emb

            if self.time_embed_act is not None:
                emb = self.time_embed_act(emb)

            encoder_hidden_states = self.process_encoder_hidden_states(
                encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
            )
        elif self.time_text_embedding_mode == "combined":
            # * SD3.5 Large Style
            emb = self.get_time_text_embed(sample=sample, timestep=timestep, pooled_projections=pooled_projections)

            if self.time_embed_act is not None:
                emb = self.time_embed_act(emb)

            encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        else:
            raise ValueError(f"Unsupported `time_text_embedding_mode`: {self.time_text_embedding_mode}. Supported modes are: `default`, `combined`.")

        # 2. pre-process
        sample = self.conv_in(sample)
        garment_features=[]

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None
        # maintain backward compatibility for legacy usage, where
        #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
        #       but can only use one or the other
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                       and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                       for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for idx, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                if self.use_rope:
                    if idx == 0:
                        image_rotary_emb = (self.reso1_image_rotary_emb_cos, self.reso1_image_rotary_emb_sin)
                    elif idx == 1:
                        image_rotary_emb = (self.reso2_image_rotary_emb_cos, self.reso2_image_rotary_emb_sin)
                    elif idx == 2:
                        image_rotary_emb = (self.reso3_image_rotary_emb_cos, self.reso3_image_rotary_emb_sin)
                    else:
                        raise ValueError(
                            f"Invalid downsample block index {idx}. Expected 0, 1, or 2, but got {idx}."
                        )
                else:
                    image_rotary_emb = None

                if image_rotary_emb is not None:
                    sample, res_samples, out_garment_feat = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                        image_rotary_emb=image_rotary_emb,
                        **additional_residuals,
                    )
                    
                    garment_features += out_garment_feat
                else:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                        **additional_residuals,
                    )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                if self.use_rope:
                    image_rotary_emb = (self.reso3_image_rotary_emb_cos, self.reso3_image_rotary_emb_sin)
                else:
                    image_rotary_emb = None
                if image_rotary_emb is not None:
                    sample, out_garment_feat = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                        image_rotary_emb=image_rotary_emb,
                    )
                    garment_features += out_garment_feat
                else:
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                    )
            else:
                sample = self.mid_block(sample, emb)

            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                if self.use_rope:
                    if i == 0:
                        image_rotary_emb = (self.reso3_image_rotary_emb_cos, self.reso3_image_rotary_emb_sin)
                    elif i == 1:
                        image_rotary_emb = (self.reso2_image_rotary_emb_cos, self.reso2_image_rotary_emb_sin)
                    elif i == 2:
                        image_rotary_emb = (self.reso1_image_rotary_emb_cos, self.reso1_image_rotary_emb_sin)
                    else:
                        raise ValueError(
                            f"Invalid upsample block index {i}. Expected 0, 1, or 2, but got {i}."
                        )
                else:
                    image_rotary_emb = None
                if image_rotary_emb is not None:
                    sample, out_garment_feat = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        image_rotary_emb=image_rotary_emb,
                    )
                    garment_features += out_garment_feat
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                    )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,), garment_features

        return UNet2DConditionOutput(sample=sample), garment_features

    def calculate_2d_rope_emb(self, block_out_channel, num_attention_heads, image_height, image_width):
        image_rotary_emb = get_2d_rotary_pos_embed(
            block_out_channel // num_attention_heads,
            ((0, 0), (image_width, image_height)),
            (image_width, image_height),
            device=self.device,
            output_type="pt",
        )
        return image_rotary_emb

    def set_sample_size(self, sample_size):
        logger.info(f"Setting sample size to {sample_size}")
        self.sample_size = sample_size

        if self.use_rope:
            logger.info(f"Setting image rotary embeddings")

            num_attention_heads = self.config.num_attention_heads or self.config.attention_head_dim
            logger.info(f"num_attention_heads: {num_attention_heads}")

            assert len(self.config.block_out_channels) == 3, "Only 3 stages are supported"
            assert len(num_attention_heads) == 3, "Only 3 stages are supported"

            reso1_image_rotary_emb = self.calculate_2d_rope_emb(
                block_out_channel=self.config.block_out_channels[0],
                num_attention_heads=num_attention_heads[0],
                image_height=sample_size,
                image_width=sample_size,
            )
            assert isinstance(reso1_image_rotary_emb, tuple) and len(reso1_image_rotary_emb) == 2
            self.register_buffer("reso1_image_rotary_emb_cos", reso1_image_rotary_emb[0].to(self.device, dtype=self.dtype))
            self.register_buffer("reso1_image_rotary_emb_sin", reso1_image_rotary_emb[1].to(self.device, dtype=self.dtype))

            reso2_image_rotary_emb = self.calculate_2d_rope_emb(
                block_out_channel=self.config.block_out_channels[1],
                num_attention_heads=num_attention_heads[1],
                image_height=sample_size // 2,
                image_width=sample_size // 2,
            )
            assert isinstance(reso2_image_rotary_emb, tuple) and len(reso2_image_rotary_emb) == 2
            self.register_buffer("reso2_image_rotary_emb_cos", reso2_image_rotary_emb[0].to(self.device, dtype=self.dtype))
            self.register_buffer("reso2_image_rotary_emb_sin", reso2_image_rotary_emb[1].to(self.device, dtype=self.dtype))

            reso3_image_rotary_emb = self.calculate_2d_rope_emb(
                block_out_channel=self.config.block_out_channels[2],
                num_attention_heads=num_attention_heads[2],
                image_height=sample_size // 4,
                image_width=sample_size // 4,
            )
            assert isinstance(reso3_image_rotary_emb, tuple) and len(reso3_image_rotary_emb) == 2
            self.register_buffer("reso3_image_rotary_emb_cos", reso3_image_rotary_emb[0].to(self.device, dtype=self.dtype))
            self.register_buffer("reso3_image_rotary_emb_sin", reso3_image_rotary_emb[1].to(self.device, dtype=self.dtype))
    
    def calculate_2d_rope_emb_rsd(
        self, block_out_channel, num_attention_heads, image_height, image_width,
        prev_image_height, prev_image_width, align_corners=True
    ):
        device = self.device
        image_rotary_emb = get_2d_rotary_pos_embed(
            block_out_channel // num_attention_heads,
            ((0, 0), (prev_image_height, prev_image_width)),
            (prev_image_height, prev_image_width),
            device=device,
            output_type="pt",
        )
        cos, sin = image_rotary_emb
        D = (block_out_channel // num_attention_heads) // 2
        emb_h_0 = cos[:, :D]
        emb_w_0 = cos[:, D:]
        emb_h_1 = sin[:, :D]
        emb_w_1 = sin[:, D:]
        emb_h_0_2d = emb_h_0.view(prev_image_height, prev_image_width, D)
        emb_w_0_2d = emb_w_0.view(prev_image_height, prev_image_width, D)
        emb_h_1_2d = emb_h_1.view(prev_image_height, prev_image_width, D)
        emb_w_1_2d = emb_w_1.view(prev_image_height, prev_image_width, D)

        max_prev = max(prev_image_height, prev_image_width)
        max_tgt = max(image_height, image_width)
        
        def resize_and_center_crop(tensor, orig_h, orig_w, tgt_h, tgt_w, max_orig, max_tgt, align_corners):
            if orig_h != orig_w:
                pad_h = (max_orig - orig_h) // 2
                pad_w = (max_orig - orig_w) // 2
                tensor = F.pad(tensor, (0, 0, pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
            
            tensor = tensor.permute(2, 0, 1)  # (H, W, D) -> (D, H, W)
            tensor_rsd = resize_tensor(tensor, size=max_tgt, align_corners=align_corners)
            tensor_rsd = tensor_rsd.permute(1, 2, 0)  # (D, H, W) -> (H, W, D)
            
            if tgt_h != tgt_w:
                crop_start_h = (max_tgt - tgt_h) // 2
                crop_start_w = (max_tgt - tgt_w) // 2
                tensor_rsd = tensor_rsd[crop_start_h:crop_start_h+tgt_h, crop_start_w:crop_start_w+tgt_w, :]
            
            return tensor_rsd.contiguous()

        emb_h_0_2d_rsd = resize_and_center_crop(
            emb_h_0_2d, prev_image_height, prev_image_width, image_height, image_width, 
            max_prev, max_tgt, align_corners
        )
        emb_w_0_2d_rsd = resize_and_center_crop(
            emb_w_0_2d, prev_image_height, prev_image_width, image_height, image_width, 
            max_prev, max_tgt, align_corners
        )
        emb_h_1_2d_rsd = resize_and_center_crop(
            emb_h_1_2d, prev_image_height, prev_image_width, image_height, image_width, 
            max_prev, max_tgt, align_corners
        )
        emb_w_1_2d_rsd = resize_and_center_crop(
            emb_w_1_2d, prev_image_height, prev_image_width, image_height, image_width, 
            max_prev, max_tgt, align_corners
        )
    
        tgt_len = image_height * image_width
        emb_h_0_rsd = emb_h_0_2d_rsd.view(tgt_len, D)
        emb_w_0_rsd = emb_w_0_2d_rsd.view(tgt_len, D)
        emb_h_1_rsd = emb_h_1_2d_rsd.view(tgt_len, D)
        emb_w_1_rsd = emb_w_1_2d_rsd.view(tgt_len, D)

        cos_rsd = torch.cat([emb_h_0_rsd, emb_w_0_rsd], dim=1)
        sin_rsd = torch.cat([emb_h_1_rsd, emb_w_1_rsd], dim=1)
        image_rotary_emb_rsd = (cos_rsd, sin_rsd)

        return image_rotary_emb_rsd

    def set_sample_size_rsd(self, sample_size_height, sample_size_width=None):
        if sample_size_width is None:
            sample_size_width = sample_size_height
        
        logger.info(f"Setting sample size to {sample_size_height}x{sample_size_width}")
        
        prev_sample_size_height = self.sample_size
        prev_sample_size_width = self.sample_size
        
        if prev_sample_size_height == sample_size_height and prev_sample_size_width == sample_size_width:
            return
        
        self.sample_size_height = sample_size_height
        self.sample_size_width = sample_size_width

        if self.use_rope:
            logger.info(f"Setting image rotary embeddings")

            num_attention_heads = self.config.num_attention_heads or self.config.attention_head_dim
            logger.info(f"num_attention_heads: {num_attention_heads}")

            assert len(self.config.block_out_channels) == 3, "Only 3 stages are supported"
            assert len(num_attention_heads) == 3, "Only 3 stages are supported"

            reso1_image_rotary_emb = self.calculate_2d_rope_emb_rsd(
                block_out_channel=self.config.block_out_channels[0],
                num_attention_heads=num_attention_heads[0],
                image_height=sample_size_height,
                image_width=sample_size_width,
                prev_image_height=prev_sample_size_height,
                prev_image_width=prev_sample_size_width,
            )
            assert isinstance(reso1_image_rotary_emb, tuple) and len(reso1_image_rotary_emb) == 2
            self.register_buffer("reso1_image_rotary_emb_cos", reso1_image_rotary_emb[0].to(self.device, dtype=self.dtype))
            self.register_buffer("reso1_image_rotary_emb_sin", reso1_image_rotary_emb[1].to(self.device, dtype=self.dtype))

            reso2_image_rotary_emb = self.calculate_2d_rope_emb_rsd(
                block_out_channel=self.config.block_out_channels[1],
                num_attention_heads=num_attention_heads[1],
                image_height=sample_size_height // 2,
                image_width=sample_size_width // 2,
                prev_image_height=prev_sample_size_height // 2,
                prev_image_width=prev_sample_size_width // 2,
            )
            assert isinstance(reso2_image_rotary_emb, tuple) and len(reso2_image_rotary_emb) == 2
            self.register_buffer("reso2_image_rotary_emb_cos", reso2_image_rotary_emb[0].to(self.device, dtype=self.dtype))
            self.register_buffer("reso2_image_rotary_emb_sin", reso2_image_rotary_emb[1].to(self.device, dtype=self.dtype))

            reso3_image_rotary_emb = self.calculate_2d_rope_emb_rsd(
                block_out_channel=self.config.block_out_channels[2],
                num_attention_heads=num_attention_heads[2],
                image_height=sample_size_height // 4,
                image_width=sample_size_width // 4,
                prev_image_height=prev_sample_size_height // 4,
                prev_image_width=prev_sample_size_width // 4,
            )
            assert isinstance(reso3_image_rotary_emb, tuple) and len(reso3_image_rotary_emb) == 2
            self.register_buffer("reso3_image_rotary_emb_cos", reso3_image_rotary_emb[0].to(self.device, dtype=self.dtype))
            self.register_buffer("reso3_image_rotary_emb_sin", reso3_image_rotary_emb[1].to(self.device, dtype=self.dtype))