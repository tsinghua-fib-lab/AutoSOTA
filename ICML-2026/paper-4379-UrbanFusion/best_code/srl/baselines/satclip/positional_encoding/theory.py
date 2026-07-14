"""
Original code from: https://github.com/microsoft/satclip

Original source:
Klemmer, Konstantin; Rolf, Esther; Robinson, Caleb; Mackey, Lester; Rußwurm, Marc.
"SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery."
arXiv preprint published November 28, 2023. Later published as conference paper at AAAI 2025.
"""

import math

import numpy as np
import torch
from torch import nn

from .common import _cal_freq_list

"""
Theory based location encoder
"""


class Theory(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """

    def __init__(
        self,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=1000,
        freq_init="geometric",
    ):
        """
        Args:
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super().__init__()
        self.frequency_num = frequency_num
        self.coord_dim = coord_dim
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.freq_init = freq_init

        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        # there unit vectors which is 120 degree apart from each other
        self.unit_vec1 = np.asarray([1.0, 0.0])  # 0
        self.unit_vec2 = np.asarray(
            [-1.0 / 2.0, math.sqrt(3) / 2.0]
        )  # 120 degree
        self.unit_vec3 = np.asarray(
            [-1.0 / 2.0, -math.sqrt(3) / 2.0]
        )  # 240 degree

        self.embedding_dim = self.cal_embedding_dim()

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(
            self.freq_init,
            self.frequency_num,
            self.max_radius,
            self.min_radius,
        )

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 6)
        self.freq_mat = np.repeat(freq_mat, 6, axis=1)

    def cal_embedding_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(2 * 3 * self.frequency_num)

    def forward(self, coords):
        device = coords.device
        dtype = coords.dtype
        N = coords.size(0)

        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords.cpu())
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # compute the dot product between [deltaX, deltaY] and each unit_vec
        # (batch_size, num_context_pt, 1)
        angle_mat1 = np.expand_dims(
            np.matmul(coords_mat, self.unit_vec1), axis=-1
        )
        # (batch_size, num_context_pt, 1)
        angle_mat2 = np.expand_dims(
            np.matmul(coords_mat, self.unit_vec2), axis=-1
        )
        # (batch_size, num_context_pt, 1)
        angle_mat3 = np.expand_dims(
            np.matmul(coords_mat, self.unit_vec3), axis=-1
        )

        # (batch_size, num_context_pt, 6)
        angle_mat = np.concatenate(
            [
                angle_mat1,
                angle_mat1,
                angle_mat2,
                angle_mat2,
                angle_mat3,
                angle_mat3,
            ],
            axis=-1,
        )
        # (batch_size, num_context_pt, 1, 6)
        angle_mat = np.expand_dims(angle_mat, axis=-2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = np.repeat(angle_mat, self.frequency_num, axis=-2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = angle_mat * self.freq_mat
        # (batch_size, num_context_pt, frequency_num*6)
        spr_embeds = np.reshape(angle_mat, (batch_size, num_context_pt, -1))

        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, frequency_num*6=input_embed_dim)
        spr_embeds[:, :, 0::2] = np.sin(spr_embeds[:, :, 0::2])  # dim 2i
        spr_embeds[:, :, 1::2] = np.cos(spr_embeds[:, :, 1::2])  # dim 2i+1

        return torch.from_numpy(spr_embeds.reshape(N, -1)).to(dtype).to(device)
