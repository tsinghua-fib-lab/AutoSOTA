#!/usr/bin/env python3
"""
Description: Implementation of the GridAndSphere class, which encodes the
spatial relation between two points in a 2D space using a combination ofgrid
and spherical harmonics encoding. Implementation by the paper "GEOGRAPHIC
LOCATION ENCODING WITH SPHERICAL SHARMONICS AND SINUSOIDAL REPRESENTATION
NETWORKS"
"""

import math

import numpy as np
import torch
from torch import nn

from .common import _cal_freq_list


class GridAndSphere(nn.Module):
    def __init__(
        self,
        coord_dim: int = 2,
        frequency_num: int = 16,
        max_radius: float = 0.01,
        min_radius: float = 0.00001,
        freq_init: str = "geometric",
        name: str = "grid",
    ) -> None:
        """
        Given a list of (deltaX,deltaY), encode them using the position
        encoding function. The encoding function is a combination of grid
        encoding and spherical harmonics encoding.

        Parameters:
        ----------
        coord_dim : int
            The dimension of the space (2D, 3D, etc.).
        frequency_num : int
            The number of different sinusoidal waves with different
            frequencies/wavelengths.
        max_radius : float
            The largest context radius this model can handle.
        min_radius : float
            The smallest context radius this model can handle.
        freq_init : str
            The frequency initialization method.
        name : str
            The name of the model.
        """

        super().__init__()

        # change name attribute to emulate the subclass
        if name == "grid":
            GridAndSphere.__qualname__ = "Grid"
            GridAndSphere.__name__ = "Grid"
        elif name == "spherec":
            GridAndSphere.__qualname__ = "SphereC"
            GridAndSphere.__name__ = "SphereC"
        elif name == "spherecplus":
            GridAndSphere.__qualname__ = "SphereCPlus"
            GridAndSphere.__name__ = "SphereCPlus"
        elif name == "spherem":
            GridAndSphere.__qualname__ = "SphereM"
            GridAndSphere.__name__ = "SphereM"
        elif name == "spheremplus":
            GridAndSphere.__qualname__ = "SphereMPlus"
            GridAndSphere.__name__ = "SphereMPlus"

        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()
        self.name = name
        self.embedding_dim = self.cal_embedding_dim()

    def cal_elementwise_angle(
        self,
        coord: float,
        cur_freq: int,
    ) -> float:
        """
        Calculate the angle for the given frequency and coordinate.

        Parameters:
        ----------
        coord : float
            The deltaX or deltaY.
        cur_freq : int
            The frequency.

        Returns:
        -------
        float
            The angle.
        """
        return coord / (
            np.power(
                self.max_radius, cur_freq * 1.0 / (self.frequency_num - 1)
            )
        )

    def cal_coord_embed(self, coords_tuple: tuple) -> list:
        """
        Calculate the coordinate embedding for the given coordinates.

        Parameters:
        ----------
        coords_tuple : tuple
            The deltaX and deltaY.

        Returns:
        -------
        list
            The coordinate embedding.
        """
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(
                    math.sin(self.cal_elementwise_angle(coord, cur_freq))
                )
                embed.append(
                    math.cos(self.cal_elementwise_angle(coord, cur_freq))
                )
        # embed: shape (input_embed_dim)
        return embed

    def cal_embedding_dim(self) -> int:
        """
        Calculate the dimension of the encoded spatial relation embedding.

        Returns:
        -------
        int
            The dimension of the encoded spatial relation embedding.
        """
        # compute the dimention of the encoded spatial relation embedding

        if self.name == "grid":
            return int(4 * self.frequency_num)
        elif self.name == "spherec":
            return int(6 * self.frequency_num)  # xyz instead of lon lat
        elif self.name == "spherecplus":
            return int(12 * self.frequency_num)
        elif self.name == "spherem":
            return int(10 * self.frequency_num)
        elif self.name == "spheremplus":
            return int(16 * self.frequency_num)  # FIX

    def cal_freq_list(self) -> None:
        """
        Calculate the frequency list.
        """
        self.freq_list = _cal_freq_list(
            self.freq_init,
            self.frequency_num,
            self.max_radius,
            self.min_radius,
        )

    def cal_freq_mat(self) -> None:
        """
        Calculate the frequency matrix.
        """
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 2)
        self.freq_mat = np.repeat(freq_mat, 2, axis=1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GridAndSphere model.

        Parameters:
        ----------
        coords : torch.Tensor
            The coordinates.

        Returns:
        -------
        torch.Tensor
            The encoded spatial relation embedding.
        """
        device = coords.device
        dtype = coords.dtype
        N = coords.size(0)

        # add 1 context point dimension (unused here)
        coords = coords[:, None, :]

        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords.cpu())
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis=4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        coords_mat = np.repeat(coords_mat, 2, axis=4)
        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        spr_embeds = coords_mat * self.freq_mat

        if self.name == "grid":
            # eq 3 in https://arxiv.org/pdf/2201.10489.pdf
            # code from https://github.com/gengchenmai/space2vec/blob/a29793336e6a1ebdb497289c286a0b4d5a83079f/spacegraph/spacegraph_codebase/SpatialRelationEncoder.py#L135

            spr_embeds[:, :, :, :, 0::2] = np.sin(
                spr_embeds[:, :, :, :, 0::2]
            )  # dim 2i
            spr_embeds[:, :, :, :, 1::2] = np.cos(
                spr_embeds[:, :, :, :, 1::2]
            )  # dim 2i+1

        elif self.name == "spherec":
            # eq 4 in https://arxiv.org/pdf/2201.10489.pdf
            # lambda: longitude, theta=latitude

            # sin_lon, sin_lat = np.sin(
            # spr_embeds[:, 0, :, :, 0]
            # ).transpose(1, 0, 2)
            # cos_lon, cos_lat = np.cos(
            # spr_embeds[:, 0, :, :, 1]
            # ).transpose(1, 0, 2)

            # eq 4
            # sin theta, cos_theta * cos_lambda, cos_theta * sin_lambda
            # sin lat, cos_lat cos_lon, cos_lat sin_lon
            # spr_embeds = np.stack([
            # sin_lat, cos_lat*cos_lon, cos_lat*sin_lon
            # ], axis=-1)

            spr_embeds = spr_embeds  # * math.pi / 180

            # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
            lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
            lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)

            # make sinuniod function
            # lon_sin, lon_cos: shape (
            # batch_size, num_context_pt, 1, frequency_num, 1
            # )
            lon_sin = np.sin(lon)
            lon_cos = np.cos(lon)

            # lat_sin, lat_cos: shape (
            # batch_size, num_context_pt, 1, frequency_num, 1
            # )
            lat_sin = np.sin(lat)
            lat_cos = np.cos(lat)

            # spr_embeds_: shape (
            # batch_size, num_context_pt, 1, frequency_num, 3
            # )
            spr_embeds_ = np.concatenate(
                [lat_sin, lat_cos * lon_cos, lat_cos * lon_sin], axis=-1
            )

            # (batch_size, num_context_pt, frequency_num*3)
            spr_embeds = np.reshape(
                spr_embeds_, (batch_size, num_context_pt, -1)
            )
        elif self.name == "spherecplus":
            # eq 10 in https://arxiv.org/pdf/2201.10489.pdf
            # (basically grid + spherec)
            spr_embeds = spr_embeds  # * math.pi / 180

            # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
            lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
            lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)

            # make sinuniod function
            # lon_sin, lon_cos: shape (
            # batch_size, num_context_pt, 1, frequency_num, 1
            # )
            lon_sin = np.sin(lon)
            lon_cos = np.cos(lon)

            # lat_sin, lat_cos: shape (
            # batch_size, num_context_pt, 1, frequency_num, 1
            # )
            lat_sin = np.sin(lat)
            lat_cos = np.cos(lat)

            # spr_embeds_: shape (
            # batch_size, num_context_pt, 1, frequency_num, 6
            # )
            spr_embeds_ = np.concatenate(
                [
                    lat_sin,
                    lat_cos,
                    lon_sin,
                    lon_cos,
                    lat_cos * lon_cos,
                    lat_cos * lon_sin,
                ],
                axis=-1,
            )

            # (batch_size, num_context_pt, 2*frequency_num*6)
            spr_embeds = np.reshape(
                spr_embeds_, (batch_size, num_context_pt, -1)
            )

        elif self.name == "spherem":
            """code from https://github.com/gengchenmai/sphere2vec/blob/8e923bbceab6065cbb4f26398122a5a6f08e0135/main/SpatialRelationEncoder.py#L1753"""

            # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
            lon_single = np.expand_dims(coords_mat[:, :, 0, :, :], axis=2)
            lat_single = np.expand_dims(coords_mat[:, :, 1, :, :], axis=2)

            # make sinuniod function
            # lon_sin, lon_cos: shape (
            # batch_size, num_context_pt, 1, frequency_num, 1
            # )
            lon_single_sin = np.sin(lon_single)
            lon_single_cos = np.cos(lon_single)

            # lat_sin, lat_cos: shape (
            # batch_size, num_context_pt, 1, frequency_num, 1
            # )
            lat_single_sin = np.sin(lat_single)
            lat_single_cos = np.cos(lat_single)

            # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
            lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
            lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)

            # make sinuniod function
            # lon_sin, lon_cos: shape (
            # batch_size, num_context_pt, 1, frequency_num, 1
            # )
            lon_sin = np.sin(lon)
            lon_cos = np.cos(lon)

            # lat_sin, lat_cos: shape (
            # batch_size, num_context_pt, 1, frequency_num, 1
            # )
            lat_sin = np.sin(lat)
            lat_cos = np.cos(lat)

            # spr_embeds_: shape (
            # batch_size, num_context_pt, 1, frequency_num, 3
            # )
            spr_embeds = np.concatenate(
                [
                    lat_sin,
                    lat_cos * lon_single_cos,
                    lat_single_cos * lon_cos,
                    lat_cos * lon_single_sin,
                    lat_single_cos * lon_sin,
                ],
                axis=-1,
            )

        elif self.name == "spheremplus":

            # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
            lon_single = np.expand_dims(coords_mat[:, :, 0, :, :], axis=2)
            lat_single = np.expand_dims(coords_mat[:, :, 1, :, :], axis=2)

            # make sinuniod function
            # lon_sin, lon_cos: shape (
            # batch_size, num_context_pt, 1, frequency_num, 1
            # )
            lon_single_sin = np.sin(lon_single)
            lon_single_cos = np.cos(lon_single)

            # lat_sin, lat_cos: shape (
            # batch_size, num_context_pt, 1, frequency_num, 1
            # )
            lat_single_sin = np.sin(lat_single)
            lat_single_cos = np.cos(lat_single)

            # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
            lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
            lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)

            # make sinuniod function
            # lon_sin, lon_cos: shape (
            # batch_size, num_context_pt, 1, frequency_num, 1
            # )
            lon_sin = np.sin(lon)
            lon_cos = np.cos(lon)

            # lat_sin, lat_cos: shape (
            # batch_size, num_context_pt, 1, frequency_num, 1
            # )
            lat_sin = np.sin(lat)
            lat_cos = np.cos(lat)

            # spr_embeds_: shape (
            # batch_size, num_context_pt, 1, frequency_num, 3
            # )
            spr_embeds = np.concatenate(
                [
                    lat_sin,
                    lat_cos,
                    lon_sin,
                    lon_cos,
                    lat_cos * lon_single_cos,
                    lat_single_cos * lon_cos,
                    lat_cos * lon_single_sin,
                    lat_single_cos * lon_sin,
                ],
                axis=-1,
            )

        return torch.from_numpy(spr_embeds.reshape(N, -1)).to(dtype).to(device)
