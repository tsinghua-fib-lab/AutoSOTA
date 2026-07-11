#
# Software Name : learning-parities-with-product-networks
# SPDX-FileCopyrightText: Copyright (c) 2026 Orange S.A.
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License .,
# see the "LICENSE.md" file for more details or https://opensource.org/licenses/MIT
#
# Author: Guillaume Larue, guillaume.larue@orange.com
# Software description: Source code of the paper "Learning High-Dimensional Parity Functions with Product Networks"
#

import torch
import torch.nn as nn

class ParallelMLP(nn.Module):
    """
    Parallel Multi-Layer Perceptron with independent networks computed efficiently.
    
    Instead of running n_outputs separate MLPs sequentially, this implementation
    processes all outputs in parallel using vectorized operations.
    
    Args:
        n_inputs: Number of input features
        n_outputs: Number of parallel independent networks
        hidden_sizes: Tuple of hidden layer sizes (e.g., (512, 512, 64))
        activation: Activation function to use (default: nn.ReLU())
        output_activation: Optional activation for output layer (default: None)
        use_bias: Whether to use bias terms (default: True)
    
    Example:
        >>> model = ParallelMLP(n_inputs=10, n_outputs=4, hidden_sizes=(64, 32))
        >>> inputs = torch.randn(5, 10)  # batch_size=5
        >>> outputs = model(inputs)  # shape: [5, 4, 1]
    """
    def __init__(
        self,
        n_inputs,
        n_outputs,
        hidden_sizes=(512, 512, 64, 1),
        activation=None,
        output_activation=None,
        use_bias=True,
    ):
        super(ParallelMLP, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_sizes = hidden_sizes
        self.use_bias = use_bias
        
        # Default activation
        self.activation = activation if activation is not None else nn.ReLU()
        self.output_activation = output_activation
        
        # Build parallel layers
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList() if use_bias else None
        
        # Input to first hidden layer
        layer_sizes = [n_inputs] + list(hidden_sizes)# + [1]
        
        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]
            
            # Weight shape: [n_outputs, out_features, in_features]
            weight = nn.Parameter(torch.randn(n_outputs, out_features, in_features))
            self.weights.append(weight)
            
            if use_bias:
                # Bias shape: [n_outputs, out_features]
                bias = nn.Parameter(torch.zeros(n_outputs, out_features))
                self.biases.append(bias)
            
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for weight in self.weights:
            # Each parallel network gets independent initialization
            for i in range(self.n_outputs):
                nn.init.kaiming_uniform_(weight[i], a=0, mode='fan_in', nonlinearity='relu')
        
        if self.use_bias:
            for bias in self.biases:
                nn.init.zeros_(bias)
    
    def forward(self, inputs):
        """
        Forward pass through parallel networks.
        
        Args:
            inputs: Tensor of shape [batch_size, n_inputs]
        
        Returns:
            Tensor of shape [batch_size, n_outputs, 1] (or [batch_size, n_outputs] if squeezed)
        """
        batch_size = inputs.size(0)
        
        # Expand inputs for parallel processing
        # [batch_size, n_inputs] -> [batch_size, n_outputs, n_inputs]
        x = inputs.unsqueeze(1).expand(batch_size, self.n_outputs, -1)
        
        # Process through layers
        num_layers = len(self.weights)
        for layer_idx, (w, b) in enumerate(zip(
            self.weights, 
            self.biases if self.use_bias else [None] * num_layers
        )):
            # x: [batch_size, n_outputs, in_features]
            # w: [n_outputs, out_features, in_features]
            # Result: [batch_size, n_outputs, out_features]
            x = torch.einsum('bpi,poi->bpo', x, w)
            
            if self.use_bias:
                x = x + b.unsqueeze(0)
            
            # Apply activation (except possibly on last layer)
            if layer_idx < num_layers - 1:
                # Hidden layer activation
                x = self.activation(x)
            else:
                # Output layer activation (if specified)
                if self.output_activation is not None:
                    x = self.output_activation(x)
        
        return x
    
    def set_weights(self, layer_idx, weights, biases=None):
        """
        Set weights for a specific layer.
        
        Args:
            layer_idx: Index of the layer (0-indexed)
            weights: Tensor of shape [n_outputs, out_features, in_features]
            biases: Optional tensor of shape [n_outputs, out_features]
        """
        self.weights[layer_idx].data = weights.clone()
        if biases is not None and self.use_bias:
            self.biases[layer_idx].data = biases.clone()
