from typing import Optional, Tuple
import logging
import sys
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import deprecate
from diffusers.models.normalization import RMSNorm
from diffusers.models.upsampling import upfirdn2d_native

# Setup logger
logger = logging.getLogger(__name__)

# Setup working directory
WORK_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
logger.debug(f"Working directory: {WORK_DIR}")
if WORK_DIR not in sys.path:
    logger.warning(f"Working directory ({WORK_DIR}) is not in sys.path. Adding it.")
    sys.path.append(WORK_DIR)

from Mobile_VTON.models.convolutions import get_convolution_module


class Downsample2D(nn.Module):
    r"""Modified from Downsample2D in diffusers.models.downsampling
            from diffusers.models.downsampling import Downsample2D
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        conv_module: str = "Conv2d",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv_module = conv_module.lower()
            conv_kwargs = dict(
                in_channels=self.channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            if conv_module == "conv2d":
                conv_kwargs.update(dict(
                    bias=bias
                ))
            elif conv_module == "sepconv2d":
                conv_kwargs.update(dict(
                    dw_groups=self.channels,
                    dw_bias=bias,
                    pw_bias=bias,
                ))
            else:
                raise NotImplementedError(f"conv_module {conv_module} not supported. Please use 'Conv2d' or 'SepConv2d'.")
            conv = get_convolution_module(conv_module, **conv_kwargs)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


DOWNSAMPLING_MODULES = {
    "downsample2d": Downsample2D,
}


def get_downsampling_module(downsampling_module: str, *args, **kwargs) -> nn.Module:
    """
    Get the downsampling module based on the downsampling_module name.

    Args:
        downsampling_module (str): The name of the downsampling module to get.
        *args: Positional arguments to pass to the downsampling module.
        **kwargs: Keyword arguments to pass to the downsampling module.

    Returns:
        nn.Module: The downsampling module.
    """
    downsampling_module = downsampling_module.lower()
    if downsampling_module in DOWNSAMPLING_MODULES:
        return DOWNSAMPLING_MODULES[downsampling_module](*args, **kwargs)
    else:
        raise ValueError(f"Unsupported downsampling module '{downsampling_module}'. Supported modules are {list(DOWNSAMPLING_MODULES.keys())}.")
