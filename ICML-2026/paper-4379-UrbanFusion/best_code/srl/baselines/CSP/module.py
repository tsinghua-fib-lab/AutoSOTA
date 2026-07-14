"""
Original code from: https://github.com/gengchenmai/csp

Original source:
Mai, Gengchen; Lao, Ni; He, Yutong; Song, Jiaming; Ermon, Stefano.
"CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations."
Proceedings of the 40th International Conference on Machine Learning (ICML), 2023.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.nn import init


class LayerNorm(nn.Module):
    """
    layer normalization
    Simple layer norm object optionally used with the convolutional encoder.
    """

    def __init__(self, feature_dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones((feature_dim,)))
        self.register_parameter("gamma", self.gamma)
        self.beta = nn.Parameter(torch.zeros((feature_dim,)))
        self.register_parameter("beta", self.beta)
        self.eps = eps

    def forward(self, x):
        # x: [batch_size, embed_dim]
        # normalize for each embedding
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # output shape is the same as x
        # Type not match for self.gamma and self.beta??????????????????????
        # output: [batch_size, embed_dim]
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def get_activation_function(activation, context_str):
    if activation == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise Exception(f"{context_str} activation not recognized.")


class SingleFeedForwardNN(nn.Module):
    """
    Creates a single layer fully connected feed forward neural network.
    this will use non-linearity, layer normalization, dropout
    this is for the hidden layer, not the last layer of the feed forard NN
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        dropout_rate=None,
        activation="sigmoid",
        use_layernormalize=False,
        skip_connection=False,
        context_str="",
    ):
        """

        Args:
            input_dim (int32): the input embedding dim
            output_dim (int32): dimension of the output of the network.
            dropout_rate (scalar tensor or float): Dropout keep prob.
            activation (string): tanh or relu or leakyrelu or sigmoid
            use_layernormalize (bool): do layer normalization or not
            skip_connection (bool): do skip connection or not
            context_str (string): indicate which spatial relation encoder is using the current FFN

        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        self.act = get_activation_function(activation, context_str)

        if use_layernormalize:
            # the layer normalization is only used in the hidden layer, not the last layer
            self.layernorm = nn.LayerNorm(self.output_dim)
        else:
            self.layernorm = None

        # the skip connection is only possible, if the input and out dimention is the same
        if self.input_dim == self.output_dim:
            self.skip_connection = skip_connection
        else:
            self.skip_connection = False

        self.linear = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform(self.linear.weight)

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size,..., output_dim]
            note there is no non-linearity applied to the output.

        Raises:
            Exception: If given activation or normalizer not supported.
        """
        assert input_tensor.size()[-1] == self.input_dim
        # Linear layer
        output = self.linear(input_tensor)
        # non-linearity
        output = self.act(output)
        # dropout
        if self.dropout is not None:
            output = self.dropout(output)

        # skip connection
        if self.skip_connection:
            output = output + input_tensor

        # layer normalization
        if self.layernorm is not None:
            output = self.layernorm(output)

        return output


class MultiLayerFeedForwardNN(nn.Module):
    """
    Creates a fully connected feed forward neural network.
    N fully connected feed forward NN, each hidden layer will use non-linearity, layer normalization, dropout
    The last layer do not have any of these
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_hidden_layers=0,
        dropout_rate=None,
        hidden_dim=-1,
        activation="sigmoid",
        use_layernormalize=False,
        skip_connection=False,
        context_str=None,
    ):
        """

        Args:
            input_dim (int32): the input embedding dim
            num_hidden_layers (int32): number of hidden layers in the network, set to 0 for a linear network.
            output_dim (int32): dimension of the output of the network.
            dropout (scalar tensor or float): Dropout keep prob.
            hidden_dim (int32): size of the hidden layers
            activation (string): tanh or relu
            use_layernormalize (bool): do layer normalization or not
            context_str (string): indicate which spatial relation encoder is using the current FFN

        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_layernormalize = use_layernormalize
        self.skip_connection = skip_connection
        self.context_str = context_str

        self.layers = nn.ModuleList()
        if self.num_hidden_layers <= 0:
            self.layers.append(
                SingleFeedForwardNN(
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    use_layernormalize=False,
                    skip_connection=False,
                    context_str=self.context_str,
                )
            )
        else:
            self.layers.append(
                SingleFeedForwardNN(
                    input_dim=self.input_dim,
                    output_dim=self.hidden_dim,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    use_layernormalize=self.use_layernormalize,
                    skip_connection=self.skip_connection,
                    context_str=self.context_str,
                )
            )

            for i in range(self.num_hidden_layers - 1):
                self.layers.append(
                    SingleFeedForwardNN(
                        input_dim=self.hidden_dim,
                        output_dim=self.hidden_dim,
                        dropout_rate=self.dropout_rate,
                        activation=self.activation,
                        use_layernormalize=self.use_layernormalize,
                        skip_connection=self.skip_connection,
                        context_str=self.context_str,
                    )
                )

            self.layers.append(
                SingleFeedForwardNN(
                    input_dim=self.hidden_dim,
                    output_dim=self.output_dim,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    use_layernormalize=False,
                    skip_connection=False,
                    context_str=self.context_str,
                )
            )

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size, ..., output_dim]
            note there is no non-linearity applied to the output.

        Raises:
            Exception: If given activation or normalizer not supported.
        """
        assert input_tensor.size()[-1] == self.input_dim
        output = input_tensor
        for i in range(len(self.layers)):
            output = self.layers[i](output)

        return output
