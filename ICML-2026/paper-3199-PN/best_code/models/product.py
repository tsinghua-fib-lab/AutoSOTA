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

from .layers.product import MultiBinaryProductLayer, BinaryProductLayer
from .layers.channels import BinarySymmetricChannelLayer

class MultiBinaryProductModel(nn.Module):
    """
    Multi-output binary product model for parallel XOR computation.
    
    Args:
        n_outputs: Number of parallel XOR units
        weight_constraint: Optional constraint function for weights
        use_gaussian_init: If True, use Gaussian initialization instead of Uniform[0,1]
        gaussian_mean: Mean for Gaussian initialization (default: 0.5)
        gaussian_std: Standard deviation for Gaussian initialization (default: 0.25)
    """
    def __init__(
        self,
        n_outputs,
        weight_constraint=None,
        use_gaussian_init=True,
        gaussian_mean=0.5,
        gaussian_std=0.25,
    ):
        super(MultiBinaryProductModel, self).__init__()
        self.n_outputs = n_outputs
        self.weight_constraint = weight_constraint
        
        self.product = MultiBinaryProductLayer(
            n_outputs=self.n_outputs,
            weight_constraint=self.weight_constraint,
            use_gaussian_init=use_gaussian_init,
            gaussian_mean=gaussian_mean,
            gaussian_std=gaussian_std,
        )

    def forward(self, inputs):
        """
        Forward pass.
        
        Args:
            inputs: Tensor of shape [batch_size, n_inputs] with binary values # THATS THE THEORY BUT NOTHING ENFORCES IT IN THE MODEL
        
        Returns:
            XOR outputs of shape [batch_size, n_outputs]
        """
        xor = self.product(inputs)
        return xor


class MultiBinaryProductModelWithOracle(nn.Module):
    """
    Multi-output binary product model with oracle for supervised XOR learning.
    
    This model includes:
    - A BSC channel to add noise to inputs
    - An oracle (non-trainable) that computes true XOR outputs
    - A trainable model that learns to replicate the oracle
    
    Args:
        n_outputs: Number of parallel XOR units
        p_e: Error probability for the BSC channel (default: 0.0)
        weight_constraint: Optional constraint function for weights
        use_gaussian_init: If True, use Gaussian initialization instead of Uniform[0,1]
        gaussian_mean: Mean for Gaussian initialization (default: 0.5)
        gaussian_std: Standard deviation for Gaussian initialization (default: 0.25)
    """
    def __init__(
        self,
        n_outputs,
        p_e=0.0,
        weight_constraint=None,
        use_gaussian_init=True,
        gaussian_mean=0.5,
        gaussian_std=0.25,
    ):
        super(MultiBinaryProductModelWithOracle, self).__init__()
        self.n_outputs = n_outputs
        self.weight_constraint = weight_constraint
        
        # BSC channel
        self.bsc_layer = BinarySymmetricChannelLayer(p_e=p_e)
        
        # Oracle (non-trainable, hard step)
        self.product_oracle = MultiBinaryProductLayer(
            n_outputs=self.n_outputs,
            weight_constraint=None,
            hard_step=True,
            use_gaussian_init=use_gaussian_init,
            gaussian_mean=gaussian_mean,
            gaussian_std=gaussian_std,
        )
        # Freeze oracle parameters
        for param in self.product_oracle.parameters():
            param.requires_grad = False
        
        # Trainable model
        self.product_model = MultiBinaryProductLayer(
            n_outputs=self.n_outputs,
            weight_constraint=self.weight_constraint,
            hard_step=False,
            use_gaussian_init=use_gaussian_init,
            gaussian_mean=gaussian_mean,
            gaussian_std=gaussian_std,
        )

    def set_bsc_error_probability(self, p_e):
        """Set the BSC error probability."""
        self.bsc_layer.set_error_probability(p_e)

    def set_oracle_parameters(self, oracle_weights):
        """Set the oracle weights (ground truth)."""
        self.product_oracle.set_xor_parameters(oracle_weights)

    def set_model_parameters(self, model_weights):
        """Set the model weights (trainable)."""
        self.product_model.set_xor_parameters(model_weights)


    def forward(self, inputs):
        """
        Forward pass.
        
        Args:
            inputs: Tensor of shape [batch_size, n_inputs] with binary values
        
        Returns:
            Tuple of (xor_oracle, xor_model, p_epsilon)
            - xor_oracle: Oracle outputs [batch_size, n_outputs]
            - xor_model: Model outputs [batch_size, n_outputs]
            - p_epsilon: Parameter error rate per XOR unit [n_outputs]
        """
        # Apply BSC noise
        bsc = self.bsc_layer(inputs)
        
        # Compute oracle and model outputs
        xor_oracle = self.product_oracle(bsc)
        xor_model = self.product_model(bsc)
        
        # Compute parameter binary errors
        # Convert continuous weights to binary by thresholding at 0.5
        with torch.no_grad():
            model_params = (torch.sign(self.product_model.product_weights - 0.5) + 1) / 2
            oracle_params = (torch.sign(self.product_oracle.product_weights - 0.5) + 1) / 2
            # Average binary error rate across inputs for each output unit
            p_epsilon = torch.mean(torch.abs(model_params - oracle_params), dim=0)

        # Compute abs parameter errors
        # Convert continuous weights to binary by thresholding at 0.5
        with torch.no_grad():
            model_params = self.product_model.product_weights
            oracle_params = self.product_oracle.product_weights
            # Average abs error rate across inputs for each output unit
            p_diff = torch.mean(torch.abs(model_params - oracle_params), dim=0)
            

        return xor_oracle, xor_model, p_epsilon, p_diff


class BinaryProductModel(nn.Module):
    """
    Single-output binary product model (DEPRECATED - use MultiBinaryProductModel with n_outputs=1).
    
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
        super(BinaryProductModel, self).__init__()
        self.n_inputs = n_inputs
        self.weight_constraint = weight_constraint
        
        self.product = BinaryProductLayer(
            n_inputs=self.n_inputs,
            weight_constraint=self.weight_constraint,
            use_gaussian_init=use_gaussian_init,
            gaussian_mean=gaussian_mean,
            gaussian_std=gaussian_std,
        )

    def forward(self, inputs):
        """Forward pass."""
        xor = self.product(inputs)
        return xor
