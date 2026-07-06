from typing import Protocol
from torch import Tensor, nn


class MLP(Protocol):
    @property
    def shape(self) -> tuple[int, int]:
        pass


class ShortAddBlock(nn.Module, MLP):
    """Basic residual block with LayerNorm before activation."""

    def __init__(self, input_size: int, hidden_size: int, activation: nn.Module = nn.ReLU()):
        if input_size != hidden_size:
            raise ValueError("input_size must equal hidden_size for ShortAddBlock")
        super().__init__()

        self.linear = nn.Linear(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.activation = activation

    @property
    def shape(self) -> tuple[int, int]:
        return (self.linear.in_features, self.linear.out_features)

    def forward(self, x: Tensor) -> Tensor:
        h = self.linear(x)
        h = self.ln(h)
        h = self.activation(h)
        o = h + x
        return o


class FFNBase(nn.Module):
    """Base feed-forward network for agent identity, temperature, and agent amount prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_blocks: int,
        activation: str = "ReLU",
    ):
        super().__init__()

        activation_fn = get_activation_function(activation)

        self.input_block = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            activation_fn,
        )

        self.blocks = nn.ModuleList(
            [ShortAddBlock(hidden_size, hidden_size, activation_fn) for _ in range(n_blocks - 1)]
        )

        self.ffn_readout = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_block(x)
        for block in self.blocks:
            x = block(x)
        return self.ffn_readout(x)

    def get_intermediate(self, x: Tensor, layer_idx: int) -> Tensor:
        """Get output after specified layer"""
        x = self.input_block(x)
        for i, block in enumerate(self.blocks):
            if i == layer_idx:
                return x
            x = block(x)
        return x


def get_activation_function(activation: str) -> nn.Module:
    """Gets an activation function module given the name of the activation."""
    if activation == "ReLU":
        return nn.ReLU()
    elif activation == "LeakyReLU":
        return nn.LeakyReLU(0.1)
    elif activation == "PReLU":
        return nn.PReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "SELU":
        return nn.SELU()
    elif activation == "ELU":
        return nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')
