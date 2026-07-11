import math

import torch

from einops import rearrange


def gaussian_orthogonal_random_matrix(nrows, ncols, scaling=0, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    nblocks = int(math.ceil(nrows / ncols))
    # TD [2021-10-28]: Sometimes QR fails on CUDA
    unstructured_blocks = torch.randn((nblocks, ncols, ncols), device='cpu')
    q, r = torch.linalg.qr(unstructured_blocks)
    # To make sure Q is uniform from the Haar distribution https://arxiv.org/pdf/math-ph/0609050.pdf
    q *= rearrange(torch.diagonal(r, dim1=-2, dim2=-1).sign(), 'b c -> b 1 c')
    q = q.to(**factory_kwargs)
    # TD [2021-10-28] Idk why the transpose is necessary. I suspect it isn't.
    # https://github.com/google-research/google-research/blob/ea313c6e96acce6c863de41615c6cf4079b8ca94/performer/fast_attention/jax/fast_attention.py#L362
    q = rearrange(q, 'b c c1 -> b c1 c')
    g_ortho = rearrange(q, 'b c1 c -> (b c1) c')[:nrows]

    if scaling == 0:
        multiplier = torch.randn((nrows, ncols), **factory_kwargs).norm(dim=1)
        return rearrange(multiplier, 'r -> r 1') * g_ortho
    elif scaling == 1:
        return math.sqrt(ncols) * g_ortho
    else:
        raise ValueError(f'Invalid scaling {scaling}')

