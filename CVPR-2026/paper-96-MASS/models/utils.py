"""Model construction helpers.

These small lookup functions map config strings to convolution blocks and
normalization layers used by the Iris encoder and decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_block(name):
    """
    Get block implementation by name.
    
    Args:
        name: Block name
        
    Returns:
        Block class
    """
    from .components.conv_blocks import (
        SingleConv, 
        BasicBlock, 
        Bottleneck
    )
    
    block_map = {
        'SingleConv': SingleConv,
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck,
    }
    return block_map[name]

def get_norm(name):
    """
    Get normalization layer by name.
    
    Args:
        name: Normalization name ('bn', 'in', 'ln')
        
    Returns:
        Normalization class
    """
    from .components.attention_blocks import LayerNorm
    
    norm_map = {
        'bn': nn.BatchNorm3d,
        'in': nn.InstanceNorm3d,
        'ln': LayerNorm
    }
    return norm_map[name]

def get_act(name):
    """
    Get activation function by name.
    
    Args:
        name: Activation name ('relu', 'gelu', 'swish')
        
    Returns:
        Activation class
    """
    act_map = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'swish': nn.SiLU
    }
    return act_map[name]
