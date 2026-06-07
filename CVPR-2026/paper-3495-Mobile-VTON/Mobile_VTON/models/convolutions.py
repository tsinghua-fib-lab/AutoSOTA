import logging
import sys
import os.path as osp
import torch
import torch.nn as nn
from typing import Optional
from types import SimpleNamespace

# Setup logging
logger = logging.getLogger(__name__)

# Setup working directory
WORK_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
logger.debug(f"Working directory: {WORK_DIR}")
if WORK_DIR not in sys.path:
    logger.warning(f"Working directory ({WORK_DIR}) is not in sys.path. Adding it.")
    sys.path.append(WORK_DIR)

from Mobile_VTON.models.activations import get_activation


class SepConv2d(nn.Module):
    r'''
    Depthwise conv + Pointwise conv
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dw_groups: Optional[int] = None,
        dw_bias: bool = True,
        pw_bias: bool = True,
    ):
        super().__init__()
        # Config
        if dw_groups is None:
            # If dw_groups is not specified, set it to in_channels
            logger.warning(
                f"dw_groups is not specified. Setting it to in_channels ({in_channels})."
            )
            dw_groups = in_channels
        self.config = SimpleNamespace(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dw_groups=dw_groups,
            dw_bias=dw_bias,
            pw_bias=pw_bias,
        )
        logger.debug(f"SepConv2d::__init__ - Config: {self.config}")

        # Depthwise conv
        # Depthwise conv only changes the HxW dimensions of the input tensor
        # Depthwise conv does not change the number of channels
        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=dw_groups,
            bias=dw_bias,
        )
        logger.debug(f"SepConv2d::__init__ - Depthwise conv: {self.dw_conv}")

        # Pointwise conv
        # Pointwise conv changes the number of channels
        # Pointwise conv does not change the HxW dimensions of the input tensor
        self.pw_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=pw_bias,
        )
        logger.debug(f"SepConv2d::__init__ - Pointwise conv: {self.pw_conv}")

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        logger.debug(f"SepConv2d::forward - Input shape: {x.shape}")
        # Depthwise conv
        x = self.dw_conv(x)
        logger.debug(f"SepConv2d::forward - Depthwise conv output shape: {x.shape}")
        # Pointwise conv
        x = self.pw_conv(x)
        logger.debug(f"SepConv2d::forward - Pointwise conv output shape: {x.shape}")
        return x


class DecoderSepConv2d(nn.Module):
    r'''
    Depthwise conv + Optional(Norm+Nonlinearity+Dropout) + Pointwise conv
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dw_groups: Optional[int] = None,
        dw_bias: bool = True,
        pw_bias: bool = True,
        usw_norm: bool = False,
        groups: int = 32,
        eps: float = 1e-6,
        use_dropout: bool = False,
        dropout: float = 0.0,
        use_non_linearity: bool = False,
        non_linearity: str = "swish",
    ):
        super().__init__()
        # Config
        self.usw_norm = usw_norm
        self.use_dropout = use_dropout
        self.use_non_linearity = use_non_linearity
        if dw_groups is None:
            # If dw_groups is not specified, set it to in_channels
            dw_groups = in_channels
        self.config = SimpleNamespace(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dw_groups=dw_groups,
            dw_bias=dw_bias,
            pw_bias=pw_bias,
        )

        # Depthwise conv
        # Depthwise conv only changes the HxW dimensions of the input tensor
        # Depthwise conv does not change the number of channels
        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=dw_groups,
            bias=dw_bias,
        )

        # Norm+Nonlinearity+Dropout
        # Norm+Nonlinearity+Dropout is applied after depthwise conv
        # Norm+Nonlinearity+Dropout is applied before pointwise conv
        if self.usw_norm:
            self.norm = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        else:
            self.norm = None
        if self.use_dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        if self.use_non_linearity:
            self.nonlinearity = get_activation(non_linearity)
        else:
            self.nonlinearity = None

        # Pointwise conv
        # Pointwise conv changes the number of channels
        # Pointwise conv does not change the HxW dimensions of the input tensor
        self.pw_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=pw_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Depthwise conv
        x = self.dw_conv(x)
        # Norm+Nonlinearity+Dropout
        if self.usw_norm:
            x = self.norm(x)
        if self.use_non_linearity:
            x = self.nonlinearity(x)
        if self.use_dropout:
            x = self.dropout(x)
        # Pointwise conv
        x = self.pw_conv(x)
        return x


CONVOLUTION_MODULES = {
    "conv2d": nn.Conv2d,
    "sepconv2d": SepConv2d,
    "decodersepconv2d": DecoderSepConv2d,
}


def get_convolution_module(conv_module: str, *args, **kwargs) -> nn.Module:
    """
    Get the convolution module based on the conv_module name.

    Args:
        conv_module (str): The type of convolution to use. Options are 'conv2d' or 'sepconv2d'.
        *args: Positional arguments to pass to the convolution module.
        **kwargs: Keyword arguments to pass to the convolution module.

    Returns:
        nn.Module: The convolution module
    """
    conv_module = conv_module.lower()
    if conv_module in CONVOLUTION_MODULES:
        return CONVOLUTION_MODULES[conv_module](*args, **kwargs)
    else:
        raise ValueError(f"Unsupported convolution module '{conv_module}'. Supported types are {list(CONVOLUTION_MODULES.keys())}.")
