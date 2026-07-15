# UNet
"""
https://pypi.org/project/unet/0.7.7/
PyTorch implementation of 2D and 3D U-Net (unet 0.7.7)
License: MIT License (MIT license)
Author: Fernando Perez-Garcia
Requires: Python >=3.6
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_classes: int = 2,
        dimensions: int = 3,
        num_encoding_blocks: int = 3,
        out_channels_first_layer: int = 8,
        normalization: Optional[str] = "batch",
        pooling_type: str = "max",
        upsampling_type: str = "linear",
        preactivation: bool = False,
        residual: bool = False,
        padding: int = 1,
        padding_mode: str = "zeros",
        activation: Optional[str] = "PReLU",
        initial_dilation: Optional[int] = None,
        dropout: float = 0,
        monte_carlo_dropout: float = 0,
    ):
        super().__init__()
        self.CHANNELS_DIMENSION = 1
        depth = num_encoding_blocks - 1

        # Force padding if residual blocks
        if residual:
            padding = 1

        # Encoder
        self.encoder = Encoder(
            in_channels,
            out_channels_first_layer,
            dimensions,
            pooling_type,
            depth,
            normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=initial_dilation,
            dropout=dropout,
        )

        # Bottom (last encoding block)
        in_channels = self.encoder.out_channels
        if dimensions == 2:
            out_channels_first = 2 * in_channels
        else:
            out_channels_first = in_channels

        self.bottom_block = EncodingBlock(
            in_channels,
            out_channels_first,
            dimensions,
            normalization,
            pooling_type=None,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=self.encoder.dilation,
            dropout=dropout,
        )

        # Decoder
        if dimensions == 2:
            power = depth - 1
        elif dimensions == 3:
            power = depth
        in_channels = self.bottom_block.out_channels
        in_channels_skip_connection = out_channels_first_layer * 2**power
        num_decoding_blocks = depth
        self.decoder = Decoder(
            in_channels_skip_connection,
            dimensions,
            upsampling_type,
            num_decoding_blocks,
            normalization=normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=self.encoder.dilation,
            dropout=dropout,
        )

        # Monte Carlo dropout
        self.monte_carlo_layer = None
        if monte_carlo_dropout:
            dropout_class = getattr(nn, f"Dropout{dimensions}d")
            self.monte_carlo_layer = dropout_class(p=monte_carlo_dropout)

        # Classifier
        if dimensions == 2:
            in_channels = out_channels_first_layer
        elif dimensions == 3:
            in_channels = 2 * out_channels_first_layer
        self.classifier = ConvolutionalBlock(
            dimensions, in_channels, out_classes, kernel_size=1, activation=None
        )

    def forward(self, x):
        skip_connections, encoding = self.encoder(x)
        encoding = self.bottom_block(encoding)
        x = self.decoder(skip_connections, encoding)
        if self.monte_carlo_layer is not None:
            x = self.monte_carlo_layer(x)
        x = self.classifier(x)
        x = F.softmax(x, dim=self.CHANNELS_DIMENSION)
        return x


# Conv


class ConvolutionalBlock(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        normalization: Optional[str] = None,
        kernel_size: int = 3,
        activation: Optional[str] = "ReLU",
        preactivation: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        dilation: Optional[int] = None,
        dropout: float = 0,
    ):
        super().__init__()

        block = nn.ModuleList()

        dilation = 1 if dilation is None else dilation
        if padding:
            total_padding = kernel_size + 2 * (dilation - 1) - 1
            padding = total_padding // 2

        class_name = f"Conv{dimensions}d"
        conv_class = getattr(nn, class_name)
        no_bias = not preactivation and (normalization is not None)
        conv_layer = conv_class(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            bias=not no_bias,
        )

        norm_layer = None
        if normalization is not None:
            class_name = f"{normalization.capitalize()}Norm{dimensions}d"
            norm_class = getattr(nn, class_name)
            num_features = in_channels if preactivation else out_channels
            norm_layer = norm_class(num_features)

        activation_layer = None
        if activation is not None:
            activation_layer = getattr(nn, activation)()

        if preactivation:
            self.add_if_not_none(block, norm_layer)
            self.add_if_not_none(block, activation_layer)
            self.add_if_not_none(block, conv_layer)
        else:
            self.add_if_not_none(block, conv_layer)
            self.add_if_not_none(block, norm_layer)
            self.add_if_not_none(block, activation_layer)

        dropout_layer = None
        if dropout:
            class_name = f"Dropout{dimensions}d"
            dropout_class = getattr(nn, class_name)
            dropout_layer = dropout_class(p=dropout)
            self.add_if_not_none(block, dropout_layer)

        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.dropout_layer = dropout_layer

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

    @staticmethod
    def add_if_not_none(module_list, module):
        if module is not None:
            module_list.append(module)


# Decoding

CHANNELS_DIMENSION = 1
UPSAMPLING_MODES = ("nearest", "linear", "bilinear", "bicubic", "trilinear")


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels_skip_connection: int,
        dimensions: int,
        upsampling_type: str,
        num_decoding_blocks: int,
        normalization: Optional[str],
        preactivation: bool = False,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        initial_dilation: Optional[int] = None,
        dropout: float = 0,
    ):
        super().__init__()
        upsampling_type = fix_upsampling_type(upsampling_type, dimensions)
        self.decoding_blocks = nn.ModuleList()
        self.dilation = initial_dilation
        for _ in range(num_decoding_blocks):
            decoding_block = DecodingBlock(
                in_channels_skip_connection,
                dimensions,
                upsampling_type,
                normalization=normalization,
                preactivation=preactivation,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.dilation,
                dropout=dropout,
            )
            self.decoding_blocks.append(decoding_block)
            in_channels_skip_connection //= 2
            if self.dilation is not None:
                self.dilation //= 2

    def forward(self, skip_connections, x):
        zipped = zip(reversed(skip_connections), self.decoding_blocks)
        for skip_connection, decoding_block in zipped:
            x = decoding_block(skip_connection, x)
        return x


class DecodingBlock(nn.Module):
    def __init__(
        self,
        in_channels_skip_connection: int,
        dimensions: int,
        upsampling_type: str,
        normalization: Optional[str],
        preactivation: bool = True,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        dilation: Optional[int] = None,
        dropout: float = 0,
    ):
        super().__init__()

        self.residual = residual

        if upsampling_type == "conv":
            in_channels = out_channels = 2 * in_channels_skip_connection
            self.upsample = get_conv_transpose_layer(
                dimensions, in_channels, out_channels
            )
        else:
            self.upsample = get_upsampling_layer(upsampling_type)
        in_channels_first = in_channels_skip_connection * (1 + 2)
        out_channels = in_channels_skip_connection
        self.conv1 = ConvolutionalBlock(
            dimensions,
            in_channels_first,
            out_channels,
            normalization=normalization,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )
        in_channels_second = out_channels
        self.conv2 = ConvolutionalBlock(
            dimensions,
            in_channels_second,
            out_channels,
            normalization=normalization,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )

        if residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions,
                in_channels_first,
                out_channels,
                kernel_size=1,
                normalization=None,
                activation=None,
            )

    def forward(self, skip_connection, x):
        x = self.upsample(x)
        skip_connection = self.center_crop(skip_connection, x)
        x = torch.cat((skip_connection, x), dim=CHANNELS_DIMENSION)
        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x += connection
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        return x

    def center_crop(self, skip_connection, x):
        skip_shape = torch.tensor(skip_connection.shape)
        x_shape = torch.tensor(x.shape)
        crop = skip_shape[2:] - x_shape[2:]
        half_crop = crop // 2
        # If skip_connection is 10, 20, 30 and x is (6, 14, 12)
        # Then pad will be (-2, -2, -3, -3, -9, -9)
        pad = -torch.stack((half_crop, half_crop)).t().flatten()
        skip_connection = F.pad(skip_connection, pad.tolist())
        return skip_connection


def get_upsampling_layer(upsampling_type: str) -> nn.Upsample:
    if upsampling_type not in UPSAMPLING_MODES:
        message = 'Upsampling type is "{}" but should be one of the following: {}'
        message = message.format(upsampling_type, UPSAMPLING_MODES)
        raise ValueError(message)
    upsample = nn.Upsample(scale_factor=2, mode=upsampling_type, align_corners=False)
    return upsample


def get_conv_transpose_layer(dimensions, in_channels, out_channels):
    class_name = f"ConvTranspose{dimensions}d"
    conv_class = getattr(nn, class_name)
    conv_layer = conv_class(in_channels, out_channels, kernel_size=2, stride=2)
    return conv_layer


def fix_upsampling_type(upsampling_type: str, dimensions: int):
    if upsampling_type == "linear":
        if dimensions == 2:
            upsampling_type = "bilinear"
        elif dimensions == 3:
            upsampling_type = "trilinear"
    return upsampling_type


# Encoding


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_first: int,
        dimensions: int,
        pooling_type: str,
        num_encoding_blocks: int,
        normalization: Optional[str],
        preactivation: bool = False,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        initial_dilation: Optional[int] = None,
        dropout: float = 0,
    ):
        super().__init__()

        self.encoding_blocks = nn.ModuleList()
        self.dilation = initial_dilation
        is_first_block = True
        for _ in range(num_encoding_blocks):
            encoding_block = EncodingBlock(
                in_channels,
                out_channels_first,
                dimensions,
                normalization,
                pooling_type,
                preactivation,
                is_first_block=is_first_block,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.dilation,
                dropout=dropout,
            )
            is_first_block = False
            self.encoding_blocks.append(encoding_block)
            if dimensions == 2:
                in_channels = out_channels_first
                out_channels_first = in_channels * 2
            elif dimensions == 3:
                in_channels = 2 * out_channels_first
                out_channels_first = in_channels
            if self.dilation is not None:
                self.dilation *= 2

    def forward(self, x):
        skip_connections = []
        for encoding_block in self.encoding_blocks:
            x, skip_connnection = encoding_block(x)
            skip_connections.append(skip_connnection)
        return skip_connections, x

    @property
    def out_channels(self):
        return self.encoding_blocks[-1].out_channels


class EncodingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_first: int,
        dimensions: int,
        normalization: Optional[str],
        pooling_type: Optional[str],
        preactivation: bool = False,
        is_first_block: bool = False,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        dilation: Optional[int] = None,
        dropout: float = 0,
    ):
        super().__init__()

        self.preactivation = preactivation
        self.normalization = normalization

        self.residual = residual

        if is_first_block:
            normalization = None
            preactivation = None
        else:
            normalization = self.normalization
            preactivation = self.preactivation

        self.conv1 = ConvolutionalBlock(
            dimensions,
            in_channels,
            out_channels_first,
            normalization=normalization,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )

        if dimensions == 2:
            out_channels_second = out_channels_first
        elif dimensions == 3:
            out_channels_second = 2 * out_channels_first
        self.conv2 = ConvolutionalBlock(
            dimensions,
            out_channels_first,
            out_channels_second,
            normalization=self.normalization,
            preactivation=self.preactivation,
            padding=padding,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )

        if residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions,
                in_channels,
                out_channels_second,
                kernel_size=1,
                normalization=None,
                activation=None,
            )

        self.downsample = None
        if pooling_type is not None:
            self.downsample = get_downsampling_layer(dimensions, pooling_type)

    def forward(self, x):
        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x += connection
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        if self.downsample is None:
            return x
        else:
            skip_connection = x
            x = self.downsample(x)
            return x, skip_connection

    @property
    def out_channels(self):
        return self.conv2.conv_layer.out_channels


def get_downsampling_layer(
    dimensions: int, pooling_type: str, kernel_size: int = 2
) -> nn.Module:
    class_name = f"{pooling_type.capitalize()}Pool{dimensions}d"
    class_ = getattr(nn, class_name)
    return class_(kernel_size)
