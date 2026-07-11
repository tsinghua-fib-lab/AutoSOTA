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

class MultiBinaryProductLayer(nn.Module):
    """
    Multi-output binary product layer for XOR computation.
    
    Implements the product node: y = prod_i (w_i * z_i + 1) where z_i = x_i - 1
    This provides a continuous extension of XOR operation.
    
    Args:
        n_outputs: Number of parallel XOR units
        weight_constraint: Optional constraint function for weights
        hard_step: If True, use hard threshold at 0.5 (non-differentiable)
        use_gaussian_init: If True, use Gaussian initialization instead of Uniform[0,1]
        gaussian_mean: Mean for Gaussian initialization (default: 0.5)
        gaussian_std: Standard deviation for Gaussian initialization (default: 0.25)
    """
    def __init__(
        self, 
        n_outputs, 
        weight_constraint=None, 
        hard_step=False,
        use_gaussian_init=True,
        gaussian_mean=0.5,
        gaussian_std=0.25,
    ):
        super(MultiBinaryProductLayer, self).__init__()
        self.n_outputs = n_outputs
        self.weight_constraint = weight_constraint
        self.hard_step = hard_step
        self.use_gaussian_init = use_gaussian_init
        self.gaussian_mean = gaussian_mean
        self.gaussian_std = gaussian_std
        
        # Weights will be initialized in forward pass or can be pre-initialized
        self.product_weights = None

    def _initialize_weights(self, n_inputs, device=None):
        """Initialize weights with Gaussian or Uniform distribution."""
        if self.use_gaussian_init:
            # Gaussian initialization
            self.product_weights = nn.Parameter(
                torch.normal(
                    mean=self.gaussian_mean, 
                    std=self.gaussian_std, 
                    size=(n_inputs, self.n_outputs),
                    device=device
                )
            )
        else:
            # Uniform [0, 1] initialization (legacy)
            self.product_weights = nn.Parameter(
                torch.rand(n_inputs, self.n_outputs, device=device)
            )
        
        # Apply constraint if provided (e.g., clamp to [0,1])
        if self.weight_constraint is not None:
            with torch.no_grad():
                self.product_weights.data = self.weight_constraint(self.product_weights.data)

    def set_xor_parameters(self, xor_weights):
        """
        Set weights to specific values.
        
        Args:
            xor_weights: Tensor of shape [n_inputs, n_outputs] with XOR weights
        """
        # Simply clone the weights - PyTorch handles device placement automatically
        # If the layer already has weights, they will be replaced with new ones on the same device
        self.product_weights = nn.Parameter(xor_weights.clone())

    def forward(self, inputs):
        """
        Forward pass of the product layer.
        
        Args:
            inputs: Tensor of shape [batch_size, n_inputs] with binary values {0, 1} # THATS THE THEORY BUT NOTHING ENFORCES IT IN THE MODEL
        
        Returns:
            Tensor of shape [batch_size, n_outputs] with values in [0, 1] # THATS THE THEORY BUT NOTHING ENFORCES IT IN THE MODEL
        """
        # Initialize weights on first forward pass
        if self.product_weights is None:
            n_inputs = inputs.shape[-1]
            self._initialize_weights(n_inputs, device=inputs.device)
        
        # Expand inputs to [batch_size, n_inputs, n_outputs]
        x = inputs.unsqueeze(-1)  # [batch_size, n_inputs, 1]
        x = x.expand(-1, -1, self.n_outputs)  # [batch_size, n_inputs, n_outputs]
        
        # Convert to bipolar form: {0, 1} -> {1, -1}
        x = 1 - 2 * x
        
        # Apply weights
        if self.hard_step:
            # Non-differentiable mode: hard threshold at 0.5
            w = (torch.sign(self.product_weights - 0.5) + 1) / 2
            x = x * w + (1 - w)
        else:
            # Model mode: use continuous weights
            w = self.product_weights
            
            # Apply constraint if provided
            if self.weight_constraint is not None:
                w = self.weight_constraint(w)
            
            x = x * w + (1 - w)
        
        # Product reduction over inputs
        x = torch.prod(x, dim=-2)  # [batch_size, n_outputs]
        
        # Convert back to binary form: {-1, 1} -> {0, 1}
        x = (1 - x) / 2
        
        return x


class BinaryProductLayer(nn.Module):
    """
    Single-output binary product layer (DEPRECATED - use MultiBinaryProductLayer with n_outputs=1).
    
    Args:
        n_inputs: Number of inputs
        weight_constraint: Optional constraint function for weights
        use_gaussian_init: If True, use Gaussian initialization instead of Uniform[0,1]
        gaussian_mean: Mean for Gaussian initialization (default: 0.5)
        gaussian_std: Standard deviation for Gaussian initialization (default: 0.25)
    """
    def __init__(
        self, 
        n_inputs, 
        weight_constraint=None,
        use_gaussian_init=True,
        gaussian_mean=0.5,
        gaussian_std=0.25,
    ):
        super(BinaryProductLayer, self).__init__()
        self.n_inputs = n_inputs
        self.weight_constraint = weight_constraint
        self.use_gaussian_init = use_gaussian_init
        
        # Initialize weights
        if use_gaussian_init:
            self.product_weights = nn.Parameter(
                torch.normal(mean=gaussian_mean, std=gaussian_std, size=(n_inputs,))
            )
        else:
            self.product_weights = nn.Parameter(torch.rand(n_inputs))
        
        # Apply constraint if provided (e.g., clamp to [0,1])
        if self.weight_constraint is not None:
            with torch.no_grad():
                self.product_weights.data = self.weight_constraint(self.product_weights.data)

    def set_xor_parameters(self, xor_weights):
        """
        Set weights directly.
        
        Args:
            xor_weights: Tensor of shape [n_inputs] with XOR weights
        """
        # Simply clone the weights - PyTorch handles device placement automatically
        self.product_weights = nn.Parameter(xor_weights.clone())

    def forward(self, inputs):
        """
        Forward pass.
        
        Args:
            inputs: Tensor of shape [batch_size, n_inputs] with binary values {0, 1}  # THATS THE THEORY BUT NOTHING ENFORCES IT IN THE MODEL 

        Returns:
            Tensor of shape [batch_size] with values in [0, 1] # THATS THE THEORY BUT NOTHING ENFORCES IT IN THE MODEL
        """
        x = inputs.reshape(-1, self.n_inputs)
        
        # Convert to bipolar form
        x = 1 - 2 * x
        
        # Apply weights with constraint
        w = self.product_weights
        if self.weight_constraint is not None:
            w = self.weight_constraint(w)
        
        x = x * w + (1 - w)
        
        # Product reduction
        x = torch.prod(x, dim=-1)
        
        # Convert back to binary form
        x = (1 - x) / 2
        
        return x
