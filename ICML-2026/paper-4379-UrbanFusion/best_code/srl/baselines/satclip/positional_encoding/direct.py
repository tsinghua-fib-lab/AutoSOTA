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
Direct encoding
"""


class Direct(nn.Module):
    def __init__(self):
        super().__init__()

        # adding this class variable is important to determine
        # the dimension of the follow-up neural network
        self.embedding_dim = 2

    def forward(self, coords):
        # place lon lat coordinates in a -pi, pi range
        coords = torch.deg2rad(coords) - torch.pi
        return coords
