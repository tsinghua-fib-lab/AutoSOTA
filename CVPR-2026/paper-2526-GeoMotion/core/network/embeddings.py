# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Union
import torch
import torch.nn as nn

def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: Union[int, Tuple[int, int]]
) -> torch.Tensor:
    """
    This function initializes a grid and generates a 2D positional embedding using sine and cosine functions.
    It is a wrapper of get_2d_sincos_pos_embed_from_grid.
    Args:
    - embed_dim: The embedding dimension.
    - grid_size: The grid size.
    Returns:
    - pos_embed: The generated 2D positional embedding.
    """
    if isinstance(grid_size, tuple):
        grid_size_h, grid_size_w = grid_size
    else:
        grid_size_h = grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float)
    grid_w = torch.arange(grid_size_w, dtype=torch.float)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(0, 3, 1, 2)

def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: torch.Tensor
) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - grid: The grid to generate the embedding from.

    Returns:
    - emb: The generated 2D positional embedding.
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=2)  # (H*W, D)
    return emb

def get_2d_sincos_pos_embed_from_seq(
    embed_dim: int, traj: torch.Tensor
) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - grid: The grid to generate the embedding from.

    Returns:
    - emb: The generated 2D positional embedding. [B, in_dim, N, L]
    """
    assert embed_dim % 2 == 0
    L = traj.shape[-1]
    pe_all = []
    for l in range(L):
        pe = get_2d_embedding(traj[..., l], embed_dim, True)
        pe_all.append(pe)
    pe_all = torch.stack(pe_all, dim=0)  # Shape: [L, 1, N, D]
    return pe_all.permute(1, 3, 2, 0)

def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: torch.Tensor
) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    omega = omega.to(pos.device)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb[None].float()

def get_2d_embedding(xy: torch.Tensor, C: int, cat_coords: bool = True) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from given coordinates using sine and cosine functions.

    Args:
    - xy: The coordinates to generate the embedding from.
    - C: The size of the embedding.
    - cat_coords: A flag to indicate whether to concatenate the original coordinates to the embedding.

    Returns:
    - pe: The generated 2D positional embedding.
    """
    B, N, D = xy.shape
    assert D == 2

    x = xy[:, :, 0:1]
    y = xy[:, :, 1:2]
    div_term = (
        torch.arange(0, C, 2, device=xy.device, dtype=torch.float32) * (1000.0 / C)
    ).reshape(1, 1, int(C / 2))

    pe_x = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)

    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)

    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)

    pe = torch.cat([pe_x, pe_y], dim=2)  # (B, N, C*3)
    if cat_coords:
        pe = torch.cat([xy, pe], dim=2)  # (B, N, C*3+3)
    return pe

class FreqEmbedder(nn.Module):
    """FreqEmbedder module. Embed inputs into higher dimensions.
    For example, x = sin(2**N * x) or sin(N * x) for N in range(0, 10)
    ref: https://github.com/ventusff/neurecon/blob/main/models/base.py
    """

    def __init__(
        self,
        input_dim,
        n_freqs,
        log_sampling=True,
        include_input=True,
        periodic_fns=(torch.sin, torch.cos),
        *args,
        **kwargs
    ):
        """
        Args:
            input_dim: dimension of input to be embedded. For example, xyz is dim=3
            n_freqs: number of frequency bands. If 0, will not encode the inputs.
            log_sampling: if True, use log factor sin(2**N * x). Else use scale factor sin(N * x).
                      By default is True
            include_input: if True, raw input is included in the embedding. Appear at beginning. By default is True
            periodic_fns: a list of periodic functions used to embed input. By default is (sin, cos)

        Returns:
            Embedded inputs with shape:
                (inputs_dim * len(periodic_fns) * N_freq + include_input * inputs_dim)
            For example, inputs_dim = 3, using (sin, cos) encoding, N_freq = 10, include_input, will results at
                3 * 2 * 10 + 3 = 63 output shape.
        """
        super(FreqEmbedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        # get output dim
        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim
        self.out_dim += self.input_dim * n_freqs * len(self.periodic_fns)

        if n_freqs == 0 and include_input:  # inputs only
            self.freq_bands = []
        else:
            if log_sampling:
                self.freq_bands = 2.**torch.linspace(0., n_freqs - 1, n_freqs)
            else:
                self.freq_bands = torch.linspace(2.**0., 2.**(n_freqs - 1), n_freqs)

    def get_output_dim(self):
        """Get output dim"""
        return self.out_dim

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: tensor of shape [B, input_dim]

        Returns:
            embed_x: tensor of shape [B, out_dim]
        """
        assert (x.shape[-1] == self.input_dim), 'Input shape should be (B, {})'.format(self.input_dim)

        embed_x = []
        if self.include_input:
            embed_x.append(x)

        for freq in self.freq_bands:
            for fn in self.periodic_fns:
                embed_x.append(fn(x * freq))

        if len(embed_x) > 1:
            embed_x = torch.cat(embed_x, dim=-1)
        else:
            embed_x = embed_x[0]

        return embed_x