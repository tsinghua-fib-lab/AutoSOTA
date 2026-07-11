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

class BinarySymmetricChannelLayer(nn.Module):
    """
    Binary Symmetric Channel layer that flips bits with probability p_e.
    
    This simulates a BSC channel where each bit is flipped independently
    with probability p_e.
    
    Args:
        p_e: Error probability (probability of bit flip)
    """
    def __init__(self, p_e):
        super(BinarySymmetricChannelLayer, self).__init__()
        # Store as a buffer (not a parameter, so it won't be trained)
        self.register_buffer('error_probability', torch.tensor([p_e], dtype=torch.float32))

    def set_error_probability(self, p_e):
        """Update the error probability."""
        self.error_probability.data = torch.tensor([p_e], dtype=torch.float32, device=self.error_probability.device)

    def forward(self, inputs):
        """
        Apply BSC noise to inputs.
        
        Args:
            inputs: Tensor of binary values {0, 1} # AGAIN NOTHING ENFORCE IT IN THE MODEL 
        
        Returns:
            Noisy tensor with bits flipped according to probability p_e # AGAIN NOTHING ENFORCE IT IN THE MODEL
        """
        # Convert to bipolar form: {0, 1} -> {1, -1}
        x = 1 - 2 * inputs
        
        # Generate random flip mask
        # If uniform < p_e, flip the bit (multiply by -1)
        # Otherwise keep it (multiply by 1)
        random_vals = torch.rand_like(inputs)
        flipped_bits = torch.where(
            random_vals < self.error_probability,
            torch.tensor(-1.0, device=inputs.device),
            torch.tensor(1.0, device=inputs.device)
        )
        
        # Apply flips
        x = x * flipped_bits
        
        # Convert back to binary form: {-1, 1} -> {0, 1}
        x = (1 - x) / 2
        
        return x
