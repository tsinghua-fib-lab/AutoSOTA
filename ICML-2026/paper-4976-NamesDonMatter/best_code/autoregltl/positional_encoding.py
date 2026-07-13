import numpy as np
import torch

def get_angles(position, i, d_embedding):
    """
    Args:
        position: int, position
        i: int, dimension
        d_embedding: int, embedding dimension
    """
    angel_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_embedding))
    return position * angel_rates


def positional_encoding(position, d_embedding, device=None, dtype=torch.float32):
    """
    Returns a sinusoidal positional encoding
    Args:
        position: int, position
        d_embedding: int, embedding dimension
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_embedding)[np.newaxis, :], d_embedding)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, device=device, dtype=dtype)