from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from glasflow.nflows.nn.nets.resnet import ResidualBlock
from zuko.nn import MLP
from torch import Tensor

from utils import torchutils


# ---------- Default MLP(s) ----------
class DefaultMLP(nn.Module):
    """Default MLP: two hidden layers of size 10*dim each, ReLU, output 2 logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_mult: int = 8,
        emb_net: str = "none",
        theta_dim: Optional[int] = None,
        x_dim: Optional[torch.Size] = None,
        out_dim: Optional[int] = None,
        image_size: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__()
        self.embedding_net = get_embedding_network(
            emb_net, output_dim=out_dim, image_size=image_size
        )
        self.theta_dim = theta_dim if theta_dim is not None else None
        self.x_dim = x_dim if x_dim is not None else None
        if out_dim is not None and theta_dim is not None:
            h = hidden_mult * (out_dim + theta_dim)
            in_dim = out_dim + theta_dim
        else:
            h = hidden_mult * input_dim
            in_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, 2),
        )

    def forward(self, inputs):
        if isinstance(self.embedding_net, nn.Identity):
            return self.net(inputs)
        else:
            theta, x = inputs[:, : self.theta_dim], inputs[:, self.theta_dim :]
            x = x.view(x.shape[0], *self.x_dim)
            xemb = self.embedding_net(x)
            inputs = torch.cat((theta, xemb), dim=1)
            return self.net(inputs)


class SmallMLP(nn.Module):
    """Smaller MLP for quick tests: two hidden layers of size 4*dim."""

    def __init__(self, input_dim: int, hidden_mult: int = 4):
        super().__init__()
        h = hidden_mult * input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 2),
        )

    def forward(self, x):
        return self.net(x)


class LinearClassifier(nn.Module):
    """Simple linear classifier (no hidden layer)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc(x)


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as commonly used in transformer architectures.

    Positional encoding introduces a way to inject information about the order of
    the input data (e.g., sequence positions) into a neural network that otherwise
    lacks a sense of position due to its permutation-invariant nature. This class
    computes sinusoidal encodings based on the position of each element in the input
    and concatenates them with the original input features.

    Attributes
    ----------
    frequencies : torch.Tensor
        A tensor containing the frequencies used to calculate the sinusoidal components.
        The frequencies are powers of 2, scaled by the base frequency.
    encode_all : bool
        Determines whether the positional encoding is applied to all features of the input
        or only the first feature (e.g., time component).
    base_freq : float
        The base frequency used to scale the sinusoidal components, defaulting to `2 * pi`.

    Parameters
    ----------
    nr_frequencies : int
        The number of sinusoidal frequencies to compute. This determines the dimensionality
        of the positional encoding for each input feature.
    encode_all : bool, optional (default=True)
        If True, the positional encoding is computed for all features in the input.
        Otherwise, it is computed only for the first feature (e.g., the time dimension).
    base_freq : float, optional (default=2 * np.pi)
        The base frequency used for sinusoidal encoding.

    Methods
    -------
    forward(t_theta)
        Computes the positional encoding for the input tensor `t_theta` and concatenates
        it with the original input features.
        - If `encode_all` is True, the positional encoding is computed for all features.
        - If `encode_all` is False, the positional encoding is applied only to the first
          feature, such as time, while other features remain unchanged.
    """

    def __init__(self, nr_frequencies, base_freq=2 * np.pi):
        super(PositionalEncoding, self).__init__()
        frequencies = base_freq * torch.pow(
            2 * torch.ones(nr_frequencies), torch.arange(0, nr_frequencies)
        ).view(1, 1, nr_frequencies)
        self.register_buffer("frequencies", frequencies)

    def forward(self, t_theta):
        """
        Computes and concatenates positional encodings with the input tensor.

        Parameters
        ----------
        t_theta : torch.Tensor
            Input tensor of shape (batch_size, input_dim), where `input_dim` is the
            dimensionality of the input features.

        Returns
        -------
        torch.Tensor
            A tensor containing the input features concatenated with the positional
            encodings. The output shape will be:
            - (batch_size, input_dim + 2 * nr_frequencies) if `encode_all` is False,
              but positional encodings are computed only for the first input feature.
        """
        batch_size = t_theta.size(0)
        x = t_theta[:, 0:1].view(batch_size, 1, 1) * self.frequencies
        cos_enc, sin_enc = torch.cos(x).view(batch_size, -1), torch.sin(x).view(batch_size, -1)
        return torch.cat((t_theta, cos_enc, sin_enc), dim=1)


class ConvNN1DLight_v2(nn.Module):
    def __init__(self, output_dim: int = 10, input_len: int = 50):
        super(ConvNN1DLight_v2, self).__init__()
        # Fewer conv layers: just 3 blocks
        self.conv1 = nn.Conv1d(1, 16, 3, 1, dilation=2, padding=1)
        self.conv2 = nn.Conv1d(16, 64, 3, 2, dilation=2, padding=1)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, dilation=2, padding=1)
        self.conv4 = nn.Conv1d(128, 128, 3, 2, dilation=2, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=3, stride=1)

        # Compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            out = self._forward_features(dummy)
            flat_dim = out.view(1, -1).shape[1]

        # Fully connected layers
        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))  # (N, 32, L)
        x = self.relu(self.conv2(x))  # (N, 64, L/2)
        x = self.pool(x)  # (N, 64, L/4)
        x = self.relu(self.conv3(x))  # (N, 128, L/8)
        x = self.relu(self.conv4(x))  # (N, 128, L/16)
        x = self.pool(x)  # (N, 128, L/16)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (N, L)
        x = x.unsqueeze(1)  # (N, 1, L)
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvNN1DLight(nn.Module):
    def __init__(self, output_dim: int = 10, input_len: int = 50):
        super(ConvNN1DLight, self).__init__()
        # Fewer conv layers: just 3 blocks
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

        # Compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            out = self._forward_features(dummy)
            flat_dim = out.view(1, -1).shape[1]

        # Fully connected layers
        self.fc1 = nn.Linear(flat_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))  # (N, 32, L)
        x = self.relu(self.conv2(x))  # (N, 64, L/2)
        x = self.pool(x)  # (N, 64, L/4)
        x = self.relu(self.conv3(x))  # (N, 128, L/8)
        x = self.pool(x)  # (N, 128, L/16)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (N, L)
        x = x.unsqueeze(1)  # (N, 1, L)
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvNN1D(nn.Module):
    def __init__(self, output_dim: int = 10):
        super(ConvNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 3, 1, dilation=2, padding=1)
        self.conv2 = nn.Conv1d(16, 64, 3, 2, dilation=2, padding=1)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, dilation=2, padding=1)
        self.conv4 = nn.Conv1d(128, 128, 3, 2, dilation=2, padding=1)
        self.conv5 = nn.Conv1d(128, 128, 3, 1, dilation=2, padding=1)
        self.conv6 = nn.Conv1d(128, 128, 3, 2, dilation=2, padding=1)
        self.conv7 = nn.Conv1d(128, 128, 3, 1, dilation=2, padding=1)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, x.shape[1])
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.pool(x)
        x = self.relu(self.conv7(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ImageSummaryStats(nn.Module):
    """Non-learnable summary statistics for image data.

    Reduces [batch, C, H, W] to a compact feature vector by combining:
    - Spatially pooled features via AdaptiveAvgPool2d -> C * pool_size^2
    - Per-channel mean and std -> 2 * C

    For [3, 64, 64] with pool_size=4: 48 + 6 = 54 features.
    """

    def __init__(self, pool_size: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.pool_size = pool_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, C, H, W] -> [batch, C * pool_size^2 + 2 * C]"""
        pooled = self.pool(x).flatten(1)  # [batch, C * pool_size^2]
        mean = x.mean(dim=(2, 3))  # [batch, C]
        std = x.std(dim=(2, 3))  # [batch, C]
        return torch.cat([pooled, mean, std], dim=1)


class ConvNN2DLT(nn.Module):
    def __init__(self, output_dim: int = 20, image_size: Tuple[int, int, int] = (3, 64, 64)):
        super(ConvNN2DLT, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 2, dilation=1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, dilation=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 2, dilation=1)
        self.conv4 = nn.Conv2d(128, 64, 1, 1, dilation=1)
        self.conv5 = nn.Conv2d(64, 3, 1, 1, dilation=1)
        if image_size[1] == 64:
            in_size = 12
        elif image_size[1] == 100:
            in_size = 27
        else:
            raise ValueError("Unsupported image size. Supported sizes are 64 and 100.")
        self.fc1 = nn.Linear(in_size, 100)
        self.fc2 = nn.Linear(100, output_dim)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input has shape (batch_size, C, H, W)"""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Conv2DTimeembedding(nn.Module):
    def __init__(
        self,
        time_embedding: bool,
        output_dim: int = 20,
        n_freqs: int = 5,
        image_size: Tuple[int, int, int] = (3, 64, 64),
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.n_freqs = n_freqs
        self.image_size = image_size
        self.conv = ConvNN2DLT(output_dim=output_dim, image_size=image_size)
        if time_embedding:
            self.positional_encoding = PositionalEncoding(nr_frequencies=n_freqs)
        else:
            self.positional_encoding = torch.nn.Identity()

    def forward(self, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Conv2DTimeembedding network.
        Args:
            t (torch.Tensor): Time tensor of shape (batch_size, 1).
            theta (torch.Tensor): Theta tensor of shape (batch_size, input_dim).
        Returns:
        torch.Tensor: Output tensor of shape (batch_size, output_dim + 2*n_freqs) or
        (batch_size, output_dim + 1) if no positional encoding.
        """
        embed = self.conv(theta)
        # Concatenate time and theta, and apply positional encoding
        t_theta = torch.cat((t, embed), dim=1)
        out = self.positional_encoding(t_theta)
        return out


class Conv1DTimeembedding(nn.Module):
    def __init__(
        self,
        time_embedding: bool,
        output_dim: int = 20,
        n_freqs: int = 5,
        light=False,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.n_freqs = n_freqs
        if light:
            self.conv = ConvNN1DLight(output_dim=output_dim)
        else:
            self.conv = ConvNN1D(output_dim=output_dim)
        if time_embedding:
            self.positional_encoding = PositionalEncoding(nr_frequencies=n_freqs)
        else:
            self.positional_encoding = torch.nn.Identity()

    def forward(self, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Conv1DTimeembedding network.
        Args:
            t (torch.Tensor): Time tensor of shape (batch_size, 1).
            theta (torch.Tensor): Theta tensor of shape (batch_size, input_dim).
        Returns:
        torch.Tensor: Output tensor of shape (batch_size, output_dim + 2*n_freqs) or
        (batch_size, output_dim + 1) if no positional encoding.
        """
        embed = self.conv(theta)
        # Concatenate time and theta, and apply positional encoding
        t_theta = torch.cat((t, embed), dim=1)
        out = self.positional_encoding(t_theta)
        return out


class IdentityEmbedding(nn.Module):
    def __init__(self):
        super(IdentityEmbedding, self).__init__()

    def forward(self, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the IdentityEmbedding network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return torch.cat((t, theta), dim=1)


class Residual(nn.Module):
    r"""Creates a residual block from a non-linear function :math:`f`.

    .. math:: y = x + f(x)

    Arguments:
        f: A function :math:`f`.
    """

    def __init__(self, f: nn.Module):
        super().__init__()

        self.f = f

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.f})"

    def forward(self, x: Tensor) -> Tensor:
        return x + self.f(x)


class ResMLP(nn.Module):
    r"""Creates a residual multi-layer perceptron (ResMLP).

    A ResMLP is a series of residual blocks where each block is a (shallow) MLP. Using
    residual blocks instead of regular non-linear functions prevents the gradients from
    vanishing, which allows for deeper networks.

    Arguments:
        in_features: The number of input features.
        out_features: The number of output features.
        hidden_features: The numbers of hidden features.
        kwargs: Keyword arguments passed to :class:`MLP`.

    Example:
        >>> net = ResMLP(64, 1, [32, 16], activation=nn.ELU)
        >>> net
        ResMLP(
          (0): Linear(in_features=64, out_features=32, bias=True)
          (1): Residual(MLP(
            (0): Linear(in_features=32, out_features=32, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=32, out_features=32, bias=True)
          ))
          (2): Linear(in_features=32, out_features=16, bias=True)
          (3): Residual(MLP(
            (0): Linear(in_features=16, out_features=16, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=16, out_features=16, bias=True)
          ))
          (4): Linear(in_features=16, out_features=1, bias=True)
        )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int] = (64, 64),
        **kwargs,
    ):
        super(ResMLP, self).__init__()
        blocks = []

        for before, after in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            if after != before:
                blocks.append(nn.Linear(before, after))

            blocks.append(Residual(MLP(after, after, [after], **kwargs)))

        blocks = blocks[:-1]
        self.blocks = nn.ModuleList(blocks)

        # Zero-init the final output layer so the initial velocity field is zero,
        # making the ODE an identity map at initialization.
        for block in reversed(self.blocks):
            if isinstance(block, nn.Linear):
                nn.init.zeros_(block.weight)
                nn.init.zeros_(block.bias)
                break

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, inputs):
        """
        Forward pass through the ResMLP.
        Parameters
        ----------
        xt : torch.Tensor
            Input tensor of shape (batch_size, in_features).
        t : torch.Tensor
            Time tensor (not used in this implementation).
        cond : torch.Tensor
            Conditional tensor (not used in this implementation).
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_features).
        """
        xt, t, cond = inputs
        x = torch.cat((xt, t, cond), dim=1)
        for block in self.blocks:
            x = block(x)
        return x


class TimeEmbeddedResMLP(nn.Module):
    r"""Creates a residual MLP with sinusoidal time embeddings.

    This variant of ResMLP applies positional (sinusoidal) encoding to the time
    input before concatenating with other features. This provides better time
    conditioning, especially near t=0 and t=1 boundaries where standard linear
    time encoding may have difficulty.

    The time embedding uses sinusoidal functions at multiple frequencies,
    similar to transformer positional encodings, which helps the network
    learn smooth interpolations across the full time range.

    Arguments:
        in_features: The number of input features (xt + cond dimensions).
        out_features: The number of output features.
        hidden_features: The numbers of hidden features.
        n_freqs: Number of frequencies for sinusoidal time embedding (default: 8).
        kwargs: Keyword arguments passed to the underlying MLP blocks.

    Example:
        >>> net = TimeEmbeddedResMLP(64, 5, [128, 128], n_freqs=8)
        >>> xt = torch.randn(32, 5)
        >>> t = torch.rand(32, 1)
        >>> cond = torch.randn(32, 59)
        >>> out = net((xt, t, cond))  # shape: (32, 5)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int] = (64, 64),
        n_freqs: int = 8,
        **kwargs,
    ):
        super(TimeEmbeddedResMLP, self).__init__()

        # Time embedding: t (1 dim) -> 1 + 2*n_freqs dims via sinusoidal encoding
        self.time_embed = PositionalEncoding(nr_frequencies=n_freqs)
        time_embed_dim = 1 + 2 * n_freqs  # Original t + sin + cos terms

        # Total input dimension: xt + time_embedding + cond
        # Original in_features = xt_dim + 1 (time) + cond_dim
        # New input = xt_dim + time_embed_dim + cond_dim
        # So we add (time_embed_dim - 1) extra dimensions
        adjusted_in_features = in_features + (time_embed_dim - 1)

        blocks = []
        for before, after in zip(
            (adjusted_in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            if after != before:
                blocks.append(nn.Linear(before, after))
            blocks.append(Residual(MLP(after, after, [after], **kwargs)))

        blocks = blocks[:-1]
        self.blocks = nn.ModuleList(blocks)

        # Zero-init the final output layer so the initial velocity field is zero,
        # making the ODE an identity map at initialization.
        for block in reversed(self.blocks):
            if isinstance(block, nn.Linear):
                nn.init.zeros_(block.weight)
                nn.init.zeros_(block.bias)
                break

        self.in_features = in_features
        self.out_features = out_features
        self.n_freqs = n_freqs

    def forward(self, inputs):
        """
        Forward pass through the TimeEmbeddedResMLP.

        Parameters
        ----------
        inputs : tuple of (xt, t, cond)
            xt : torch.Tensor of shape (batch_size, xt_dim)
            t : torch.Tensor of shape (batch_size, 1)
            cond : torch.Tensor of shape (batch_size, cond_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_features).
        """
        xt, t, cond = inputs

        # Apply sinusoidal embedding to time
        # time_embed expects shape (batch, 1) and returns (batch, 1 + 2*n_freqs)
        t_embedded = self.time_embed(t)

        # Concatenate: xt + embedded_time + cond
        x = torch.cat((xt, t_embedded, cond), dim=1)

        for block in self.blocks:
            x = block(x)
        return x


class DenseResidualNet(nn.Module):
    """
    A neural network module consisting of a sequence of dense residual blocks to embed
    high-dimensional input into a compressed output. Linear resizing layers adjust the
    input and output to match the first and last hidden dimensions, respectively.

    When the output dimension is multi-dimensional (e.g., for images), the output is
    passed through a tanh activation and rescaled to [0, 1]. For single-dimensional outputs,
    no activation or rescaling is applied.

    Module specs
    --------
        input dimension:    (batch_size, input_dim)
        output dimension:   (batch_size, output_dim) or (batch_size, *output_dim) if multi-dimensional
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | torch.Size,
        hidden_dims: Tuple,
        activation: Callable = F.elu,
        dropout: float = 0.0,
        batch_norm: bool = True,
        context_features: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of the input to this module.
        output_dim : int or torch.Size
            Output dimension of this module. If int, output is (batch_size, output_dim).
            If torch.Size with len > 1, output is (batch_size, *output_dim) with tanh activation
            and values rescaled to [0, 1].
        hidden_dims : tuple
            Tuple with dimensions of hidden layers.
        activation : callable
            Activation function used in residual blocks.
        dropout : float
            Dropout probability for residual blocks used for regularization.
        batch_norm : bool
            Flag to specify whether to use batch normalization.
        context_features : int, optional
            Number of additional context features for gated linear units. If None, no context is expected.
        """
        super(DenseResidualNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = torch.Size((output_dim,)) if isinstance(output_dim, int) else output_dim
        self.hidden_dims = hidden_dims
        self.num_res_blocks = len(self.hidden_dims)

        self.initial_layer = nn.Linear(self.input_dim, hidden_dims[0])
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=self.hidden_dims[n],
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout,
                    use_batch_norm=batch_norm,
                )
                for n in range(self.num_res_blocks)
            ]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_dims[n - 1], self.hidden_dims[n])
                if self.hidden_dims[n - 1] != self.hidden_dims[n]
                else nn.Identity()
                for n in range(1, self.num_res_blocks)
            ]
            + [nn.Linear(self.hidden_dims[-1], self.output_dim.numel())]
        )
        # Zero-init the final output layer so the initial velocity field is zero,
        # making the ODE an identity map at initialization.
        nn.init.zeros_(self.resize_layers[-1].weight)
        nn.init.zeros_(self.resize_layers[-1].bias)

    def forward(self, x, context=None):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            context: Optional context tensor for gated linear units.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, *output_dim).
                          If output_dim is multi-dimensional (len > 1), applies tanh activation
                          and rescales values to [0, 1]. Otherwise, returns reshaped output without modification.
        """
        x = self.initial_layer(x)
        for block, resize_layer in zip(self.blocks, self.resize_layers):
            x = block(x, context=context)
            x = resize_layer(x)

        x = x.view(-1, *self.output_dim)

        return x


class ContinuousFlow(nn.Module):
    """
    A continuous normalizing flow network. It defines a time-dependent vector field on
    the parameter space (score or flow), which optionally depends on additional context
    information.

    v = v(f(t, theta), g(context))

    This class combines the network v for the continuous flow itself, as well as embedding
    networks f, g, for the context and parameters, respectively.

    The parameters and context can optionally be provided as gated linear unit (GLU)
    context to the main network, rather than as the main input to the network. For a
    DenseResidualNet, this context is input repeatedly via GLUs, for each residual block.
    """

    def __init__(
        self,
        continuous_flow_net: nn.Module,
        context_embedding_net: nn.Module = torch.nn.Identity(),
        theta_embedding_net: nn.Module = torch.nn.Identity(),
        context_with_glu: bool = False,
        theta_with_glu: bool = False,
    ):
        """
        Parameters
        ----------
        continuous_flow_net: nn.Module
            Main network for the continuous flow.
        context_embedding_net: nn.Module = torch.nn.Identity()
            Embedding network for the context information (e.g., observed data).
        theta_embedding_net: nn.Module = torch.nn.Identity()
            Embedding network for the parameters.
        context_with_glu: bool = False
            Whether to provide context as GLU or main input to the continuous_flow_net.
        theta_with_glu: bool = False
            Whether to provide theta (and t) as GLU or main input to the
            continuous_flow_net.
        """
        super(ContinuousFlow, self).__init__()
        self.continuous_flow_net = continuous_flow_net
        self.context_embedding_net = context_embedding_net
        self.theta_embedding_net = theta_embedding_net
        self.theta_with_glu = theta_with_glu
        self.context_with_glu = context_with_glu

        self._use_cache = None
        self._cached_context = None
        self._cached_context_embedding = None

    @property
    def use_cache(self):
        # unless set explicitly, use_cache is True in eval mode and False in train mode
        if self._use_cache is not None:
            return self._use_cache
        else:
            return not self.training

    @use_cache.setter
    def use_cache(self, value):
        self._use_cache = value

    def _update_cached_context(self, *context: torch.Tensor):
        """
        Update the cache for *context. This sets new values for self._cached_context and
        self._cached_context_embedding if self._cached_context != context.
        """
        try:
            # This may fail when batch size of context and _cached_context is different
            # (but both > 1).
            if (
                self._cached_context is not None
                and len(self._cached_context) == len(context)
                and all([(x == y).all() for x, y in zip(self._cached_context, context)])
            ):
                return
        except RuntimeError:
            pass
        # if all tensors in batch are the same: do forward pass with batch_size 1
        if all([(x == x[:1]).all() for x in context]):
            self._cached_context = tuple(x[:1] for x in context)
            self._cached_context_embedding = self.context_embedding_net(
                *self._cached_context
            ).detach()

        else:
            self._cached_context = context
            self._cached_context_embedding = self.context_embedding_net(
                *self._cached_context
            ).detach()

    def _get_cached_context_embedding(self, batch_size):
        if self._cached_context_embedding.size(0) == 1:
            return self._cached_context_embedding.repeat(
                batch_size,
                *[1 for _ in range(len(self._cached_context_embedding.shape) - 1)],
            )
        return self._cached_context_embedding

    def forward(self, inputs):
        theta, t, context = inputs
        # print(f"shape of theta: {theta.shape}")
        # print(f"shape of t: {t.shape}")
        # print(f"shape of context: {context.shape}")
        # embed theta (self.embedding_net_theta might just be identity)
        t_and_theta_embedding = self.theta_embedding_net(t, theta)
        # for unconditional forward pass
        if len(context) == 0:
            assert not self.theta_with_glu
            return self.continuous_flow_net(t_and_theta_embedding)

        # embed context (self.context_embedding_net might just be identity)
        context_embedding = self.context_embedding_net(context)

        if len(t_and_theta_embedding.shape) != 2 or len(context_embedding.shape) != 2:
            raise NotImplementedError()

        # a = context_embedding and b = t_and_theta_embedding now need to be provided
        # to the continuous flow network, which predicts a vector field as a function
        # of a and b. The flow network has two entry points: the normal input to the
        # feedforward network (first argument in forward pass) and via a glu between
        # the residual blocks (second argument in forward pass, optional). The flags
        # self.theta_with_glu and self.context_with_glu specify whether we use the
        # first entrypoint (= False) or the second (= True).
        if self.context_with_glu and self.theta_with_glu:
            main_input = torch.Tensor([])
            glu_context = torch.cat((context_embedding, t_and_theta_embedding), dim=1)
        elif not self.context_with_glu and not self.theta_with_glu:
            # print(f"shape of context_embedding: {context_embedding.shape}")
            # print(f"shape of t_and_theta_embedding: {t_and_theta_embedding.shape}")
            main_input = torch.cat((context_embedding, t_and_theta_embedding), dim=1)
            glu_context = None
        elif self.context_with_glu:
            main_input = t_and_theta_embedding
            glu_context = context_embedding
        else:  # if self.theta_with_glu:
            main_input = context_embedding
            glu_context = t_and_theta_embedding

        if glu_context is None:
            return self.continuous_flow_net(main_input)
        else:
            return self.continuous_flow_net(main_input, glu_context)


def create_cf(
    posterior_kwargs: dict, embedding_kwargs: dict = None, theta_embedding_kwargs: dict = None
) -> nn.Module:
    """
    Build a continuous flow based on settings dictionaries.

    Parameters
    ----------
    posterior_kwargs: dict
        Settings for the flow. This includes the settings for the parameter embedding.
    embedding_kwargs: dict
        Settings for the context embedding network.
    initial_weights: dict
        Initial weights for the embedding network (of SVD projection type).

    Returns
    -------
    nn.Module
        Neural network for the continuous flow.
    """
    theta_dim = posterior_kwargs["input_dim"]
    theta_dim = torch.Size((theta_dim,)) if isinstance(theta_dim, int) else torch.Size(theta_dim)
    context_dim = posterior_kwargs["context_dim"]

    # get embeddings modules for context
    if embedding_kwargs:
        context_embedding = get_embedding_network(**embedding_kwargs)
    else:
        context_embedding = torch.nn.Identity()

    # get embeddings modules for theta (which is actually cat(t, theta))
    if theta_embedding_kwargs:
        print("Using theta embedding network.")
        theta_embedding = get_theta_embedding_net(**theta_embedding_kwargs)
    else:
        theta_embedding = IdentityEmbedding()

    # get output dimensions of embedded context and theta
    theta_with_glu = posterior_kwargs.get("theta_with_glu", False)
    context_with_glu = posterior_kwargs.get("context_with_glu", False)
    embedded_theta_dim = theta_embedding(torch.zeros(10, 1), torch.zeros(10, *theta_dim))
    assert len(embedded_theta_dim.shape) == 2, (
        f"Theta embedding should return a single dimension but got {embedded_theta_dim.shape} dimensions."
    )
    embedded_theta_dim = embedded_theta_dim.shape[1]

    glu_dim = theta_with_glu * embedded_theta_dim + context_with_glu * context_dim
    input_dim = embedded_theta_dim + context_dim - glu_dim
    if glu_dim == 0:
        glu_dim = None

    activation_fn = torchutils.get_activation_function_from_string(posterior_kwargs["activation"])
    continuous_flow_net = DenseResidualNet(
        input_dim=input_dim,
        output_dim=theta_dim,
        hidden_dims=posterior_kwargs["hidden_dims"],
        activation=activation_fn,
        dropout=posterior_kwargs["dropout"],
        batch_norm=posterior_kwargs["batch_norm"],
        context_features=glu_dim,
    )

    model = ContinuousFlow(
        continuous_flow_net,
        context_embedding,
        theta_embedding,
        theta_with_glu=posterior_kwargs.get("theta_with_glu", False),
        context_with_glu=posterior_kwargs.get("context_with_glu", False),
    )
    return model


def get_theta_embedding_net(name: str, **embedding_kwargs):
    if name == "conv2d":
        embedding_net = Conv2DTimeembedding(
            time_embedding=embedding_kwargs.get("time_embedding", True),
            output_dim=embedding_kwargs["output_dim"],
            n_freqs=embedding_kwargs.get("n_freqs", 5),
            image_size=embedding_kwargs.get("image_size", (3, 100, 100)),
        )
    if name == "conv1d":
        embedding_net = Conv1DTimeembedding(
            time_embedding=embedding_kwargs.get("time_embedding", True),
            output_dim=embedding_kwargs["output_dim"],
            n_freqs=embedding_kwargs.get("n_freqs", 5),
        )
    if name == "conv1d_light":
        embedding_net = Conv1DTimeembedding(
            time_embedding=embedding_kwargs.get("time_embedding", True),
            output_dim=embedding_kwargs["output_dim"],
            n_freqs=embedding_kwargs.get("n_freqs", 5),
            light=True,
        )
    return embedding_net


# def get_dim_positional_embedding(encoding: dict, input_dim: int):
#     if encoding.get("encode_all"):
#         return (1 + 2 * encoding["frequencies"]) * input_dim
#     return 2 * encoding["frequencies"] + input_dim


def get_embedding_network(name: str, **kwargs) -> nn.Module:
    if name in ["pendulum", "no_misspec_pendulum", "conv1d"]:
        return ConvNN1D(output_dim=kwargs["output_dim"])
    if name in ["conv1d_v2"]:
        return ConvNN1DLight_v2(
            output_dim=kwargs["output_dim"], input_len=kwargs.get("input_len", 50)
        )
    elif name in ["wind_tunnel", "no_misspec_wind_tunnel", "conv1d_light"]:
        return ConvNN1DLight(output_dim=kwargs["output_dim"])
    elif name in ["light_tunnel", "conv2d"]:
        return ConvNN2DLT(output_dim=kwargs["output_dim"], image_size=kwargs["image_size"])
    elif name == "summary_stats":
        return ImageSummaryStats(pool_size=kwargs.get("pool_size", 4))
    else:
        return nn.Identity()
