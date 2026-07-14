#!/usr/bin/env python3
"""
Description: Implementation of a coordinate encoder that applies a positional
encoding to coordinates and passes the result through a neural network.
Implementation by the paper "GEOGRAPHIC LOCATION ENCODING WITH SPHERICAL
HARMONICS AND SINUSOIDAL REPRESENTATION NETWORKS"
"""

import torch
from torch import nn

import srl.encoders.coordinate_encoder.nn as NN
import srl.encoders.coordinate_encoder.pe as PE


class CoordinateEncoder(nn.Module):
    def __init__(
        self,
        positional_encoding_name: str = "sphericalharmonics",
        neural_network_name: str = "siren",
        embed_dim: int = 256,
        seq_len: int = 1,
        dim_hidden: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
        legendre_polys: int = 10,
        harmonics_calculation: str = "analytic",
        min_radius: float = None,
        max_radius: float = None,
        frequency_num: int = None,
        return_encoding: bool = False,
        precomputed_features: bool = False,
        cartesian_3d_branch: bool = False,
        cartesian_3d_input_dropout: float = 1.0,
        synthetic: bool = False,
    ) -> None:
        """
        Encoder that applies a positional encoding to coordinates and passes
        the result through a neural network. By default, the encoder uses a
        spherical harmonics for encoding the coordinates, followed by a SIREN
        network that weights and recombines the basis functions. Also other
        positional encodings and neural networks can be used.

        Parameters
        ----------
        positional_encoding_name : str
            Name of the positional encoding to use. By default
            "sphericalharmonics".
        neural_network_name : str
            Name of the neural network to use. By default "siren".
        embed_dim : int, default=256
            Embedding dimension for neural network output, used as input for
            the multi-modal network.
        seq_len : int, default=1
            Length of the output token sequence.
        dim_hidden : int, default=256
            Hidden layer size for neural networks. Only used if
            neural_network_name is "siren" or "fcnet".
        num_layers : int, default=3
            Number of layers for applicable networks. Only used if
            neural_network_name is "siren" or "fcnet".
        dropout : float, default=0.0
            Dropout probability for networks supporting dropout.
        legendre_polys : int, default=10
            Number of Legendre polynomials for spherical harmonics.
        harmonics_calculation : str, default='analytic'
            Calculation method for spherical harmonics ('analytic' or
            'discretized').
        min_radius : float, optional
            Minimum radius for encodings requiring radial bounds. Only used if
            positional_encoding_name is "theory" or "wrap".
        max_radius : float, optional
            Maximum radius for encodings requiring radial bounds. Only used if
            positional_encoding_name is "theory" or "wrap".
        frequency_num : int, optional
            Number of frequency bands for encodings. Only used if
            positional_encoding_name is "theory" or "wrap".
        return_positional_encoding : bool, default=True
            If True, return the positional encoding along with the output.
        precomputed_features : bool, default=False
            If True, the input is assumed to be precomputed features
            (e.g., spherical harmonics) instead of raw coordinates.
            Precomputed features are useful for speeding up the training
            process, as they can be computed once and reused for multiple
            training runs. If False, the input is assumed to be raw
            coordinates, and the positional encoding will be computed
            on-the-fly.
        cartesian_3d_branch : bool, default=False
            If True, the encoder will also include a branch for Cartesian 3D
            coordinates. This is useful for models that require both spherical
            and Cartesian representations of the input coordinates.
        cartesian_3d_input_dropout : float, default=1.0
            Dropout probability for the Cartesian 3D input branch. This is
            applied to the Cartesian coordinates before passing them through
            the Cartesian 3D encoder. A value of 1.0 means no dropout is
            applied, while a value of 0.0 means the input is dropped out
            entirely. This parameter is only used if `cartesian_3d_branch` is
            set to True.
        synthetic : bool, default=False
            If True, the encoder will use synthetic data for training, used
            for ablation studies for analyzing partial information
            decomposition.
        """
        super().__init__()

        # Store hyperparameters
        self.positional_encoding_name = positional_encoding_name
        self.neural_network_name = neural_network_name
        self.embed_dim = embed_dim
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.legendre_polys = legendre_polys
        self.harmonics_calculation = harmonics_calculation
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.frequency_num = frequency_num
        self.seq_len = seq_len
        self.return_encoding = return_encoding
        self.precomputed_features = precomputed_features
        self.cartesian_3d_branch = cartesian_3d_branch
        self.cartesian_3d_input_dropout = cartesian_3d_input_dropout
        self.synthetic = synthetic

        # Initialize modules
        self.positional_encoder = self._build_positional_encoder().double()
        self.neural_network = self._build_neural_network().double()
        if self.cartesian_3d_branch:
            self.cartesian_3d_encoder = (
                self._build_cartesian_3d_branch().double()
            )

    def _build_positional_encoder(self) -> nn.Module:
        """
        Build the positional encoding module based on the specified name.

        Returns
        -------
        nn.Module
            Positional encoding module.
        """
        name = self.positional_encoding_name
        if name == "direct":
            return PE.Direct()
        if name == "direct_no_rad":
            return PE.DirectNoRad()
        if name == "cartesian3d":
            return PE.Cartesian3D()
        if name == "sphericalharmonics":
            if self.harmonics_calculation == "discretized":
                return PE.DiscretizedSphericalHarmonics(
                    legendre_polys=self.legendre_polys
                )
            return PE.SphericalHarmonics(
                legendre_polys=self.legendre_polys,
                harmonics_calculation=self.harmonics_calculation,
            )
        if name == "theory":
            return PE.Theory(
                min_radius=self.min_radius,
                max_radius=self.max_radius,
                frequency_num=self.frequency_num,
            )
        if name == "wrap":
            return PE.Wrap()
        if name in {
            "grid",
            "spherec",
            "spherecplus",
            "spherem",
            "spheremplus",
        }:
            return PE.GridAndSphere(
                min_radius=self.min_radius,
                max_radius=self.max_radius,
                frequency_num=self.frequency_num,
                name=name,
            )
        raise ValueError(f"Unknown positional encoding: {name}")

    def _build_neural_network(self) -> nn.Module:
        """
        Build the neural network module based on the specified name.
        Positional encoding serves as input to the neural network.
        Neural network can be a linear layer, SIREN, or FCNet.

        Returns
        -------
        nn.Module
            Neural network module.
        """
        name = self.neural_network_name
        input_dim = self.positional_encoder.embedding_dim
        if name == "linear":
            return nn.Linear(input_dim, self.embed_dim * self.seq_len)
        if name == "siren":
            return NN.SirenNet(
                dim_in=input_dim,
                dim_hidden=self.dim_hidden,
                num_layers=self.num_layers,
                dim_out=self.embed_dim * self.seq_len,
                dropout=self.dropout,
            )
        if name == "fcnet":
            return NN.FCNet(
                num_inputs=input_dim,
                num_classes=self.embed_dim * self.seq_len,
                dim_hidden=self.dim_hidden,
            )
        if name == "rff":
            return PE.RFF(
                output_dim=self.embed_dim * self.seq_len,
                synthetic=self.synthetic,
            )
        if name == "rff_siren":
            return PE.RFF(
                output_dim=self.embed_dim * self.seq_len,
                siren=True,
                num_layers_siren=self.num_layers,
            )

    def _build_cartesian_3d_branch(self) -> nn.Module:
        """
        Experimental feature for improving modelling of high-frequency
        variations for spherical harmonics-based location encoding.
        """
        layers = []
        layers.append(PE.Cartesian3D())
        layers.append(nn.Linear(3, self.dim_hidden))
        layers.append(nn.GELU())
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.dim_hidden, self.dim_hidden))
            layers.append(nn.GELU())
        layers.append(
            nn.Linear(self.dim_hidden, self.embed_dim * self.seq_len)
        )
        self.coord_dropout = nn.Dropout2d(p=self.cartesian_3d_input_dropout)
        return nn.Sequential(*layers)

    def get_harmoinics(self, lonlats: torch.Tensor) -> torch.Tensor:
        """
        Process the input coordinates through the positional encoder.
        This method is used to obtain the harmonics of the input.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The processed tensor.
        """
        x = self.positional_encoder(lonlats)
        return x

    def forward_features(self, lonlats: torch.Tensor) -> torch.Tensor:
        """
        Compute encoded outputs from longitude/latitude inputs,
        before neural network projection.

        Parameters
        ----------
        lonlats : torch.Tensor
            Tensor of shape (batch_size, seq_len, 2) containing coordinate
            inputs.

        Returns
        -------
        torch.Tensor
            Output of the neural network applied to positional encodings.
        """
        return self.positional_encoder(lonlats.double())

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        coord_order: str = "latlon",
    ) -> torch.Tensor:
        """
        Compute encoded outputs from longitude/latitude inputs.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch_size, 2) containing coordinate
            inputs. Optionally, it can be a precomputed feature tensor of shape
            (batch_size, n_features).
        return_features : bool
            If True, return the coordinates along with the output.
            Default is False.

        Returns
        -------
        torch.Tensor
            Output of the neural network applied to positional encodings.
        """
        if self.precomputed_features is False:
            coords = x.detach()
            if (
                self.neural_network_name == "rff"
                or self.neural_network_name == "rff_siren"
            ):
                output = self.neural_network(x[:, -2:].double())
            if coord_order == "latlon":
                x = x[:, [1, 0]]  # convert to [lon, lat]
            encoding = self.positional_encoder(x.double())
            if self.cartesian_3d_branch:
                cartesian_input = x.double()
                cartesian_features = self.cartesian_3d_encoder(cartesian_input)
                cartesian_features = cartesian_features.unsqueeze(1)
                cartesian_features = self.coord_dropout(cartesian_features)
                cartesian_features = cartesian_features.squeeze(1)
            if (
                self.neural_network_name != "rff"
                and self.neural_network_name != "rff_siren"
            ):
                output = self.neural_network(encoding)
        else:
            coords = x[:, -2:].detach()
            encoding = x[:, :-2].double()
            if self.cartesian_3d_branch:
                cartesian_input = x[:, -2:].double()
                cartesian_features = self.cartesian_3d_encoder(cartesian_input)
                cartesian_features = cartesian_features.unsqueeze(1)
                cartesian_features = self.coord_dropout(cartesian_features)
                cartesian_features = cartesian_features.squeeze(1)
            if (
                self.neural_network_name == "rff"
                or self.neural_network_name == "rff_siren"
            ):
                output = self.neural_network(x[:, -2:].double())
            else:
                output = self.neural_network(encoding)
        if self.cartesian_3d_branch:
            output = output + cartesian_features
        output = output.float()

        # Reshape to (batch_size, seq_len, embed_dim)
        output = output.view(x.shape[0], self.seq_len, -1)

        if self.return_encoding:
            if return_features:
                return output, encoding.detach(), coords
            else:
                return output, encoding.detach()
        if return_features:
            return output, coords
        else:
            return output
