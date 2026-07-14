"""
Original code from: https://github.com/VicenteVivan/geo-clip

Original source:
Vivanco, Vicente; Nayak, Gaurav Kumar; Shah, Mubarak.
"GeoCLIP: CLIP-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization."
NeurIPS 2023. arXiv preprint published September 27, 2023.
"""

import os

import numpy as np
import pandas as pd
import torch

file_dir = os.path.dirname(os.path.realpath(__file__))


def load_gps_data(csv_file):
    data = pd.read_csv(csv_file)
    lat_lon = data[["LAT", "LON"]]
    gps_tensor = torch.tensor(lat_lon.values, dtype=torch.float32)
    return gps_tensor
