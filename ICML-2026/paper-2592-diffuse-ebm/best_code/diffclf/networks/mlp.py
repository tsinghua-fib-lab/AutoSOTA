# Time-conditioned MLP and Fourier-feature MLP architectures

# Libraries
import torch
from .utils import PositionalEmbedding


class TimeNet(torch.nn.Module):
    """Build a time dependent neural network"""

    def __init__(self, dim_out, activation=torch.nn.GELU, num_layers=4,
                 channels=64, last_bias_init=None, last_weight_init=None):
        """Constructor fot TimeNet

        Args:
            * dim_out (int): Output dimension
            * activation (constructor): Activation function (default is torch.nn.GELU)
            * num_layers (int): Number of MLP layers (default is 4)
            * channels (int): Width of MLP layers (default is 64)
            * last_bias_init (callable): Initialization function for last biais (default is None)
            * last_weight_init (callable): Initialization function for last weights (default is None)
        """

        super().__init__()
        self.activation = activation()
        self.register_buffer(
            "timestep_coeff",
            torch.linspace(start=0.1, end=100, steps=channels).unsqueeze(0),
            persistent=False,
        )
        self.timestep_phase = torch.nn.Parameter(torch.randn(1, channels))
        self.hidden_layer = torch.nn.ModuleList([torch.nn.Linear(2 * channels, channels)])
        self.hidden_layer += [
            torch.nn.Linear(channels, channels) for _ in range(num_layers - 2)
        ]
        self.out_layer = torch.nn.Linear(channels, dim_out)
        if last_bias_init:
            if ((hasattr(last_bias_init, '__code__') and 'weight' in last_bias_init.__code__.co_varnames)) \
                    or ((hasattr(last_bias_init, 'func') and 'weight' in last_bias_init.func.__code__.co_varnames)):
                last_bias_init(self.out_layer.bias, weight=self.out_layer.weight)
            else:
                last_bias_init(self.out_layer.bias)
        if last_weight_init is not None:
            last_weight_init(self.out_layer.weight)

    def forward(self, t):
        """Compute the output

        Args:
            * t (torch.Tensor of shape (batch_size, 1)): Time

        Returns:
            * ret (torch.Tensor of shape (batch_size, dim_out)): Output
        """
        t = t.view(-1, 1).float()
        sin_embed_t = torch.sin((self.timestep_coeff * t) + self.timestep_phase)
        cos_embed_t = torch.cos((self.timestep_coeff * t) + self.timestep_phase)
        embed_t = torch.cat([sin_embed_t, cos_embed_t], dim=1)
        for layer in self.hidden_layer:
            embed_t = self.activation(layer(embed_t))
        return self.out_layer(embed_t)


class FourierNet(torch.nn.Module):
    """Build a FourierNet"""

    def __init__(self, dim, dim_out, activation=torch.nn.GELU, num_layers=4, channels=64,
                 use_pos_embedding=False, last_bias_init=None, last_weight_init=None):
        """Constructor fot FourierNet

        Args:
            * dim (int): Input dimension
            * dim_out (int): Output dimension
            * activation (constructor): Activation function (default is torch.nn.GELU)
            * num_layers (int): Number of MLP layers (default is 4)
            * channels (int): Width of MLP layers (default is 64)
            * use_pos_embedding (bool): Whether to use PositionalEmbedding (default is False)
            * last_bias_init (callable): Initialization function for last biais (default is None)
            * last_weight_init (callable): Initialization function for last weights (default is None)
        """

        super().__init__()
        self.activation = activation()
        self.input_embed = torch.nn.Linear(dim, channels)
        if use_pos_embedding:
            self.timestep_embed = PositionalEmbedding(num_channels=channels)
        else:
            self.timestep_embed = TimeNet(
                dim_out=channels,
                activation=activation,
                num_layers=2,
                channels=channels,
            )
        self.hidden_layer = torch.nn.ModuleList(
            [torch.nn.Linear(channels, channels) for _ in range(num_layers - 2)]
        )
        self.out_layer = torch.nn.Linear(channels, dim_out)
        if last_bias_init:
            if ((hasattr(last_bias_init, '__code__') and 'weight' in last_bias_init.__code__.co_varnames)) \
                    or ((hasattr(last_bias_init, 'func') and 'weight' in last_bias_init.func.__code__.co_varnames)):
                last_bias_init(self.out_layer.bias, weight=self.out_layer.weight)
            else:
                last_bias_init(self.out_layer.bias)
        if last_weight_init is not None:
            last_weight_init(self.out_layer.weight)

    def forward(self, t, x):
        """Compute the output

        Args:
            * t (torch.Tensor of shape (batch_size, 1)): Time
            * x (torch.Tensor of shape (batch_size, dim)): State

        Returns:
            * ret (torch.Tensor of shape (batch_size, dim_out)): Output
        """
        assert t.shape == (x.shape[0], 1)
        t = t.view(-1, 1)
        embed_t = self.timestep_embed(t)
        embed_x = self.input_embed(x)
        embed = embed_x + embed_t
        for layer in self.hidden_layer:
            embed = layer(self.activation(embed))
        return self.out_layer(self.activation(embed))


class ImprovedFourierNet(torch.nn.Module):
    """Build an improved version of FourierNet"""

    def __init__(self, dim, dim_out, activation=torch.nn.GELU, num_layers=5, channels=64,
                 use_pos_embedding=False, last_bias_init=None, last_weight_init=None):
        """Constructor fot ImprovedFourierNet

        Args:
            * dim (int): Input dimension
            * dim_out (int): Output dimension
            * activation (constructor): Activation function (default is torch.nn.GELU)
            * num_layers (int): Number of MLP layers (default is 4)
            * channels (int): Width of MLP layers (default is 64)
            * use_pos_embedding (bool): Whether to use PositionalEmbedding (default is False)
            * last_bias_init (callable): Initialization function for last biais (default is None)
            * last_weight_init (callable): Initialization function for last weights (default is None)
        """

        super().__init__()

        self.activation = activation()
        self.input_embed = torch.nn.Linear(dim, channels)
        if use_pos_embedding:
            self.timestep_embed = PositionalEmbedding(num_channels=channels)
        else:
            self.timestep_embed = TimeNet(
                dim_out=channels,
                activation=activation,
                num_layers=1,
                channels=channels,
            )
        self.layers_post_time_skip = torch.nn.ModuleList([
            torch.nn.Linear(channels, channels) for _ in range(num_layers - 2)
        ])
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(channels, channels) for _ in range(num_layers - 2)
        ])
        self.out_layer = torch.nn.Linear(channels, dim_out)

        if last_bias_init is not None:
            if ((hasattr(last_bias_init, '__code__') and 'weight' in last_bias_init.__code__.co_varnames)) \
                    or ((hasattr(last_bias_init, 'func') and 'weight' in last_bias_init.func.__code__.co_varnames)):
                last_bias_init(self.out_layer.bias, weight=self.out_layer.weight)
            else:
                last_bias_init(self.out_layer.bias)
        if last_weight_init is not None:
            last_weight_init(self.out_layer.weight)

    def forward(self, t, x):
        """Compute the output

        Args:
            * t (torch.Tensor of shape (batch_size, 1)): Time
            * x (torch.Tensor of shape (batch_size, dim)): State

        Returns:
            * ret (torch.Tensor of shape (batch_size, dim_out)): Output
        """
        embed_t = self.timestep_embed(t)
        embed_x = self.input_embed(x)
        z = embed_x + embed_t
        for layer, layer_post_time_skip in zip(self.layers, self.layers_post_time_skip):
            z_ = embed_t + layer(self.activation(z))
            z_ = layer_post_time_skip(self.activation(z_))
            z = z_ + z
        return self.out_layer(self.activation(z))
