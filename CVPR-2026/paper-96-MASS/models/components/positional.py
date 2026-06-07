"""3D positional encoding layers.

These utilities provide fixed sinusoidal and learnable positional encodings for
volumetric attention blocks.
"""

import torch
import torch.nn as nn
import numpy as np
import math

__all__ = [
    'get_emb',
    'positional_encoding_3d',
    'positional_encoding_permute_3d',
    'LearnablePositionalEncoding3D'
]

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


def positional_encoding_3d(batch_size, x, y, z, channels):
    """
    Compute positional encoding for a 5D tensor.
    
    :param channels: The last dimension of the tensor you want to apply pos emb to.
    :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
    """
    
    channels = int(np.ceil(channels / 6) * 2)
    if channels % 2:
        channels += 1

    inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
    inv_freq = inv_freq

    pos_x = torch.arange(x, dtype=inv_freq.dtype)
    pos_y = torch.arange(y, dtype=inv_freq.dtype)
    pos_z = torch.arange(z, dtype=inv_freq.dtype)
    
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    sin_inp_z = torch.einsum("i,j->ij", pos_z, inv_freq)
    
    emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
    emb_y = get_emb(sin_inp_y).unsqueeze(1)
    emb_z = get_emb(sin_inp_z)
    
    emb = torch.zeros((x, y, z, channels * 3))
    emb[:, :, :, :channels] = emb_x
    emb[:, :, :, channels:2*channels] = emb_y
    emb[:, :, :, 2*channels:] = emb_z

    return emb[None, :, :, :, :channels].repeat(batch_size, 1, 1, 1, 1).float()


def positional_encoding_permute_3d(batch_size, channels, x, y, z):
    """
    Compute positional encoding for a 5D tensor with permuted dimensions.
    Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
    
    :param tensor: A 5d tensor of size (batch_size, ch, x, y, z)
    :param channels: The number of channels in the input tensor
    :return: Positional Encoding Matrix of size (batch_size, ch, x, y, z)
    """
    enc = positional_encoding_3d(batch_size, x, y, z, channels)
    return enc.permute(0, 4, 1, 2, 3)


class LearnablePositionalEncoding3D(nn.Module):
    def __init__(self, channels, max_x, max_y, max_z):
        super(LearnablePositionalEncoding3D, self).__init__()
        
        self.channels = channels
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z
        
        self.embedding = nn.Parameter(torch.randn(1, channels, max_x, max_y, max_z))
    
    def forward(self, x):
        # x shape: (batch_size, x, y, z, channels)
        B, C, D, H, W = x.shape
        
        # Slice the embedding tensor to match the input dimensions
        pos_embed = self.embedding[:, :, :D, :H, :W]
        
        # Expand to match batch size
        pos_embed = pos_embed.expand(B, -1, -1, -1, -1)
        
        return x + pos_embed
