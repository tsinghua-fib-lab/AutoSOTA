"""
Original code from: https://github.com/VicenteVivan/geo-clip

Original source:
Vivanco, Vicente; Nayak, Gaurav Kumar; Shah, Mubarak.
"GeoCLIP: CLIP-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization."
NeurIPS 2023. arXiv preprint published September 27, 2023.
"""

import torch
import torch.nn as nn

from .misc import file_dir
from .rff import GaussianEncoding

# Constants
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336


def equal_earth_projection(L):
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
    def __init__(self, sigma, synthetic_experiment=False):
        super().__init__()
        rff_encoding = GaussianEncoding(
            sigma=sigma, input_size=2, encoded_size=256
        )
        self.km = sigma
        if synthetic_experiment:
            self.capsule = nn.Sequential(
                rff_encoding,
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            self.head = nn.Sequential(nn.Linear(128, 3))
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
            self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x


class LocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8], synthetic_experiment=False):
        super().__init__()
        self.sigma = sigma
        self.synthetic_experiment = synthetic_experiment

        self.n = len(self.sigma)

        for i, s in enumerate(self.sigma):
            self.add_module(
                "LocEnc" + str(i),
                LocationEncoderCapsule(
                    sigma=s, synthetic_experiment=self.synthetic_experiment
                ),
            )

    def forward(self, location):
        location = equal_earth_projection(location)
        if self.synthetic_experiment:
            location_features = torch.zeros(location.shape[0], 3).to(
                location.device
            )
        else:
            location_features = torch.zeros(location.shape[0], 512).to(
                location.device
            )

        for i in range(self.n):
            location_features += self._modules["LocEnc" + str(i)](location)

        return location_features
