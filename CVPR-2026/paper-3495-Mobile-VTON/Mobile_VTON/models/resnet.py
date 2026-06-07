from functools import partial
from typing import Optional, Tuple, Union
import logging
import sys
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import deprecate
from diffusers.models.downsampling import (  # noqa
    downsample_2d,
)
from diffusers.models.upsampling import (  # noqa
    upsample_2d,
)

# Setup logger
logger = logging.getLogger(__name__)

# Setup working directory
WORK_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
logger.debug(f"Working directory: {WORK_DIR}")
if WORK_DIR not in sys.path:
    logger.warning(f"Working directory ({WORK_DIR}) is not in sys.path. Adding it.")
    sys.path.append(WORK_DIR)

from Mobile_VTON.models.convolutions import get_convolution_module
from Mobile_VTON.models.downsampling import get_downsampling_module
from Mobile_VTON.models.upsampling import get_upsampling_module
from Mobile_VTON.models.activations import get_activation


class ResnetBlock2D(nn.Module):
    r"""Modified from ResnetBlock2D in diffusers.models.resnet
            from diffusers.models.resnet import ResnetBlock2D
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift,
        kernel: Optional[torch.Tensor] = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        middle_channels: Optional[int] = None,
        conv1_module: str = "Conv2d",
        conv2_module: str = "Conv2d",
        downsample_module: str = "Downsample2D",
        downsample_conv_module: str = "Conv2d",
        upsample_module: str = "Upsample2D",
        upsample_conv_module: str = "Conv2d",
        dw_bias: bool = True,
        pw_bias: bool = False,
    ):
        super().__init__()
        if time_embedding_norm == "ada_group":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==ada_group`, please use `ResnetBlockCondNorm2D` instead",
            )
        if time_embedding_norm == "spatial":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==spatial`, please use `ResnetBlockCondNorm2D` instead",
            )

        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        middle_channels = out_channels if middle_channels is None else middle_channels
        self.middle_channels = middle_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        conv1_module = conv1_module.lower()
        conv1_kwargs = dict(
            in_channels=in_channels,
            out_channels=middle_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if conv1_module == "conv2d":
            pass
        elif conv1_module == "sepconv2d":
            conv1_kwargs.update(dict(
                dw_groups=in_channels,
                dw_bias=dw_bias,
                pw_bias=pw_bias,
            ))
        else:
            raise NotImplementedError(f"conv1_module {conv1_module} not supported. Please use 'Conv2d' or 'HSWSepConv2d'.")
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = get_convolution_module(conv1_module, **conv1_kwargs)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(temb_channels, middle_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(temb_channels, 2 * middle_channels)
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=middle_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        conv2_module = conv2_module.lower()
        conv2_kwargs = dict(
            in_channels=middle_channels,
            out_channels=conv_2d_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if conv2_module == "conv2d":
            pass
        elif conv2_module == "sepconv2d":
            conv2_kwargs.update(dict(
                dw_groups=middle_channels,
                dw_bias=dw_bias,
                pw_bias=pw_bias,
            ))
        else:
            raise ValueError(f"conv2_module {conv2_module} not supported. Please use 'Conv2d' or 'SepConv2d'.")
        # self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = get_convolution_module(conv2_module, **conv2_kwargs)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                upsample_module = upsample_module.lower()
                upsample_module_kwargs = dict(
                    channels=in_channels,
                    use_conv=False,
                )
                if upsample_module == "upsample2d":
                    upsample_conv_module = upsample_conv_module.lower()
                    upsample_module_kwargs.update(dict(
                        conv_module=upsample_conv_module
                    ))
                else:
                    raise ValueError(f"upsample_module {upsample_module} not supported. Please use 'HSWUpsample2D'.")
                # self.upsample = Upsample2D(in_channels, use_conv=False)
                self.upsample = get_upsampling_module(upsample_module, **upsample_module_kwargs)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                downsample_module = downsample_module.lower()
                downsample_module_kwargs = dict(
                    channels=in_channels,
                    use_conv=False,
                    padding=1,
                    name="op",
                )
                if downsample_module == "downsample2d":
                    downsample_conv_module = downsample_conv_module.lower()
                    downsample_module_kwargs.update(dict(
                        conv_module=downsample_conv_module
                    ))
                else:
                    raise ValueError(f"downsample_module {downsample_module} not supported. Please use 'Downsample2D'.")
                # self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")
                self.downsample = get_downsampling_module(downsample_module, **downsample_module_kwargs)

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class DecoderResnetBlock2D(nn.Module):
    r"""Modified from ResnetBlock2D in diffusers.models.resnet
            from diffusers.models.resnet import ResnetBlock2D
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift,
        kernel: Optional[torch.Tensor] = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        middle_channels: Optional[int] = None,
        conv1_module: str = "DecoderSepConv2d",
        conv2_module: str = "DecoderSepConv2d",
        dw_bias: bool = True,
        pw_bias: bool = False,
    ):
        super().__init__()
        if time_embedding_norm == "ada_group":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==ada_group`, please use `ResnetBlockCondNorm2D` instead",
            )
        if time_embedding_norm == "spatial":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==spatial`, please use `ResnetBlockCondNorm2D` instead",
            )

        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        middle_channels = out_channels if middle_channels is None else middle_channels
        self.middle_channels = middle_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        if groups_out is None:
            groups_out = groups

        # self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        #! SnapGen Paper '3.2. Tiny and Fast Decoder' Point 2 & 3
        conv1_module = conv1_module.lower()
        conv1_kwargs = dict(
            in_channels=in_channels,
            out_channels=middle_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if conv1_module == "decodersepconv2d":
            conv1_kwargs.update(dict(
                dw_groups=in_channels,
                dw_bias=dw_bias,
                pw_bias=pw_bias,
                usw_norm=False,
                groups=groups,
                eps=eps,
                use_dropout=False,
                dropout=dropout,
                use_non_linearity=False,
                non_linearity=non_linearity,
            ))
        else:
            raise ValueError(f"conv1_module {conv1_module} not supported. Please use 'DecoderSepConv2d'.")
        self.conv1 = get_convolution_module(conv1_module, **conv1_kwargs)

        if temb_channels is not None:
            raise ValueError(f"temb_channels should be None but got {temb_channels}.")
        else:
            self.time_emb_proj = None

        conv_2d_out_channels = conv_2d_out_channels or out_channels
        #! SnapGen Paper '3.2. Tiny and Fast Decoder' Point 2 & 3
        conv2_module = conv2_module.lower()
        conv2_kwargs = dict(
            in_channels=middle_channels,
            out_channels=conv_2d_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if conv2_module == "decodersepconv2d":
            conv2_kwargs.update(dict(
                dw_groups=middle_channels,
                dw_bias=dw_bias,
                pw_bias=pw_bias,
                usw_norm=True,
                groups=groups_out,
                eps=eps,
                use_dropout=True,
                dropout=dropout,
                use_non_linearity=True,
                non_linearity=non_linearity,
            ))
        else:
            raise ValueError(f"conv2_module {conv2_module} not supported. Please use 'DecoderSepConv2d'.")
        self.conv2 = get_convolution_module(conv2_module, **conv2_kwargs)

        # self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            raise ValueError("`up` is not supported in this class.")
        elif self.down:
            raise ValueError("`down` is not supported in this class.")

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        hidden_states = input_tensor

        if self.upsample is not None:
            raise ValueError("`upsample` is not supported in this class.")
        elif self.downsample is not None:
            raise ValueError("`downsample` is not supported in this class.")

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            raise ValueError(f"self.time_emb_proj should be None but got {self.time_emb_proj}.")

        if temb is not None:
            raise ValueError(f"temb should be None but got {temb}.")

        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


RESNET_MODULES = {
    "resnetblock2d": ResnetBlock2D,
    "decoderresnetblock2d": DecoderResnetBlock2D,
}


def get_resnet_module(resnet_module: str, *args, **kwargs) -> nn.Module:
    """
    Get the resnet module based on the resnet_module name.

    Args:
        resnet_module (str): The name of the resnet module.
        *args: Positional arguments to pass to the resnet module.
        **kwargs: Keyword arguments to pass to the resnet module.

    Returns:
        nn.Module: The resnet module.
    """
    resnet_module = resnet_module.lower()
    if resnet_module in RESNET_MODULES:
        return RESNET_MODULES[resnet_module](*args, **kwargs)
    else:
        raise ValueError(f"Unsupported resnet module: '{resnet_module}'. Supported modules are {list(RESNET_MODULES.keys())}.")
