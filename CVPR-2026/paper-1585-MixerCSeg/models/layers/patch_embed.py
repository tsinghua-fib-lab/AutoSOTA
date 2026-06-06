'''
Author: Chenhongyi Yang
Reference: https://github.com/OliverRensu/Shunted-Transformer
'''
import torch
from mmcv.cnn.bricks.transformer import AdaptivePadding
from mmcv.cnn import (build_conv_layer, build_norm_layer)
from mmcv.runner.base_module import BaseModule
from mmcv.utils import to_2tuple
from torch import nn

class ConvPatchEmbed(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 num_convs=0,
                 conv_type='Conv2d',
                 patch_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super(ConvPatchEmbed, self).__init__(init_cfg=init_cfg)

        assert patch_size % 2 == 0

        self.embed_dims = embed_dims
        if stride is None:
            stride = patch_size // 2
        else:
            stride = stride // 2

        self.stem = torch.nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7,7), stride=(2,2), padding=3, bias=False),
            # nn.BatchNorm2d(64),
            nn.GroupNorm(num_channels=64, num_groups=4),
            nn.ReLU(True))

        if num_convs > 0:
            convs = []
            for _ in range(num_convs):
                convs.append(torch.nn.Conv2d(64, 64, (3,3), (1,1), padding=1, bias=False))
                convs.append(torch.nn.GroupNorm(num_channels=64, num_groups=4))
                convs.append(torch.nn.ReLU(True))
            self.convs = torch.nn.Sequential(*convs)
        else:
            self.convs = None

        kernel_size = to_2tuple(patch_size//2)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adaptive_padding = None
        padding = to_2tuple(padding)

        # self.projection = build_conv_layer(
        #     cfg=dict(
        #     type=conv_type,
        #     in_channels=64,
        #     out_channels=embed_dims,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding,
        #     dilation=dilation,
        #     bias=bias)

        self.projection = nn.Conv2d(
            in_channels=64,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # e.g. when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            _input_size = (input_size[0] // 2, input_size[1] // 2)
            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(_input_size)
                input_h, input_w = _input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                _input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (_input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (_input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

            - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (out_h, out_w).
        """
        x = self.stem(x)
        if self.convs is not None:
            x = self.convs(x)

        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size