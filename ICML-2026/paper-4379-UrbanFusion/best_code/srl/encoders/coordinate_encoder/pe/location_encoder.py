#!/usr/bin/env python3
"""
Description: Implementation of a coordinate encoder using random fourier
features and MLP for location encoding.

Based on code from: https://github.com/VicenteVivan/geo-clip
"""
import torch
import torch.nn as nn

from srl.encoders.coordinate_encoder.nn.siren import SirenNet

from .rff import GaussianEncoding

# Constants
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336


def equal_earth_projection(L: torch.Tensor) -> torch.Tensor:
    """
    Equal Earth projection for latitude and longitude.

    Parameters
    ----------
    L : torch.Tensor
        A tensor of shape (N, 2) containing latitude and longitude in degrees.

    Returns
    -------
    torch.Tensor
        A tensor of shape (N, 2) containing the projected coordinates.
    """
    latitude = L[:, 0]
    longitude = L[:, 1]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    denominator = 3 * (
        9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1
    )
    x = (
        2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)
    ) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    return (torch.stack((x, y), dim=1) * SF) / 180


class LocationEncoderCapsule(nn.Module):
    def __init__(
        self,
        sigma: float,
        output_dim: int,
        siren: bool = False,
        num_layers_siren: int = 2,
        synthetic: bool = False,
    ) -> None:
        """
        Location encoder using random fourier features and MLP.

        Parameters
        ----------
        sigma : float
            The standard deviation for the Gaussian encoding.
        output_dim : int
            The output dimension of the encoder.
        siren : bool, optional
            Whether to use SIREN for the MLP head (default is False).
        num_layers_siren : int, optional
            The number of layers in the SIREN (default is 2).
        synthetic : bool, optional
            Whether to use synthetic data (default is False).

        """
        super().__init__()
        rff_encoding = GaussianEncoding(
            sigma=sigma, input_size=2, encoded_size=256
        )
        self.km = sigma
        if siren:
            self.capsule = rff_encoding
            self.head = SirenNet(
                dim_in=512,
                dim_hidden=1024,
                dim_out=output_dim,
                num_layers=num_layers_siren,
                dropout_rate=0.0,
            )
        elif synthetic:
            self.capsule = nn.Sequential(
                rff_encoding,
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            self.head = nn.Sequential(nn.Linear(128, output_dim))
        else:
            self.capsule = nn.Sequential(
                rff_encoding,
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
            )
            self.head = nn.Sequential(nn.Linear(1024, output_dim))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x


class RFF(nn.Module):
    def __init__(
        self,
        output_dim: int,
        sigma: list = [2**0, 2**4, 2**8],
        siren: bool = False,
        num_layers_siren: int = 2,
        from_pretrained: bool = False,
        synthetic: bool = False,
    ) -> None:
        """
        Location encoder using hierarchical random Fourier features and
        MLPs.

        Parameters
        ----------
        output_dim : int
            The output dimension of the encoder.
        sigma : list, optional
            The list of sigma values for the Gaussian encoding (default is
            [2**0, 2**4, 2**8]).
        siren : bool, optional
            Whether to use SIREN for the MLP head (default is False).
        num_layers_siren : int, optional
            The number of layers in the SIREN (default is 2).
        from_pretrained : bool, optional
            Whether to load weights from a pretrained model (default is False).
        synthetic : bool, optional
            Whether to use synthetic data (default is False).
        """

        super().__init__()
        self.sigma = sigma

        self.n = len(self.sigma)
        self.output_dim = output_dim

        for i, s in enumerate(self.sigma):
            self.add_module(
                "LocEnc" + str(i),
                LocationEncoderCapsule(
                    sigma=s,
                    output_dim=output_dim,
                    siren=siren,
                    num_layers_siren=num_layers_siren,
                    synthetic=synthetic,
                ),
            )

    def forward(self, location: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the location encoder.

        Parameters
        ----------
        location : torch.Tensor
            The input location tensor (latitude, longitude).

        Returns
        -------
        torch.Tensor
            The output feature tensor.
        """
        location = equal_earth_projection(location)
        location_features = torch.zeros(location.shape[0], self.output_dim).to(
            location.device
        )

        for i in range(self.n):
            location_features += self._modules["LocEnc" + str(i)](location)

        return location_features
