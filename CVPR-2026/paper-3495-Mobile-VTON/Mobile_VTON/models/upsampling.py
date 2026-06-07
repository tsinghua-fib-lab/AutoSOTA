from typing import Optional, Tuple
import logging
import sys
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import deprecate
from diffusers.utils.import_utils import is_torch_version
from diffusers.models.normalization import RMSNorm

# Setup logger
logger = logging.getLogger(__name__)

# Setup working directory
WORK_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
logger.debug(f"Working directory: {WORK_DIR}")
if WORK_DIR not in sys.path:
    logger.warning(f"Working directory ({WORK_DIR}) is not in sys.path. Adding it.")
    sys.path.append(WORK_DIR)

from Mobile_VTON.models.convolutions import get_convolution_module


class Upsample2D(nn.Module):
    r"""Modified from Upsample2D in diffusers.models.upsampling
            from diffusers.models.upsampling import Upsample2D
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
        conv_module: str = "Conv2d",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = nn.ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv_module = conv_module.lower()
            conv_kwargs = dict(
                in_channels=self.channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=1,
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

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16 until PyTorch 2.1
        # https://github.com/pytorch/pytorch/issues/86679#issuecomment-1783978767
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16 and is_torch_version("<", "2.1"):
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            # upsample_nearest_nhwc also fails when the number of output elements is large
            # https://github.com/pytorch/pytorch/issues/141831
            scale_factor = (
                2 if output_size is None else max([f / s for f, s in zip(output_size, hidden_states.shape[-2:])])
            )
            if hidden_states.numel() * scale_factor > pow(2, 31):
                hidden_states = hidden_states.contiguous()

            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # Cast back to original dtype
        if dtype == torch.bfloat16 and is_torch_version("<", "2.1"):
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states


UPSAMPLING_MODULES = {
    "upsample2d": Upsample2D,
}


def get_upsampling_module(upsampling_module: str, *args, **kwargs) -> nn.Module:
    """
    Get the upsampling module based on the upsampling_module name.

    Args:
        upsampling_module (`str`): The name of the upsampling module.
        *args: Positional arguments to pass to the upsampling module.
        **kwargs: Keyword arguments to pass to the upsampling module.

    Returns:
        `nn.Module`: The upsampling module.
    """
    upsampling_module = upsampling_module.lower()
    if upsampling_module in UPSAMPLING_MODULES:
        return UPSAMPLING_MODULES[upsampling_module](*args, **kwargs)
    else:
        raise ValueError(f"Unsupported upsampling module '{upsampling_module}'. Supported modules are {list(UPSAMPLING_MODULES.keys())}.")
