import pickle
from typing import Tuple

import numpy as np
import torch
from sbi.utils import BoxUniform
from torch.distributions import TransformedDistribution

from simulator.base import Simulator
from utils.transform import TriangularTransform


class SIR(Simulator):
    def __init__(self, results_path: str):
        super().__init__(obs_dim=365, theta_dim=2, name="sir")
        self.results_path = results_path
        self.training_size = None
        self.calibration_size = None
        base_dist = BoxUniform(
            low=torch.tensor([0.0, 0.0]),
            high=torch.tensor([1.0, 1.0]),
        )
        self.prior_dist = TransformedDistribution(
            base_dist,
            TriangularTransform(),
        )
        self.like_dist = None
        self.post_dist = None
        self.denoise_dist = None

    def generate_dataset(self, num_simulations: int, num_cal_max: int) -> Tuple:
        self.training_size = num_simulations
        self.calibration_size = num_cal_max
        data = pickle.load(open(self.results_path, "rb"))["data"]
        theta = data["theta"]
        x = data["x_raw"]
        y = data["y_raw"]
        theta = torch.from_numpy(np.array(theta)).to(torch.float32)
        x = torch.from_numpy(np.array(x)).to(torch.float32)
        y = torch.from_numpy(np.array(y)).to(torch.float32)
        theta_mean = theta.mean(dim=0)
        theta_std = theta.std(dim=0)
        x_mean = x.mean(dim=0)
        x_std = x.std(dim=0)
        y_mean = y.mean(dim=0)
        y_std = y.std(dim=0)
        scales = {
            "theta_mean": theta_mean,
            "theta_std": theta_std,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
        }
        return (
            theta[:num_simulations],
            x[:num_simulations],
            y[:num_simulations],
            theta[num_simulations:num_cal_max],
            x[num_simulations:num_cal_max],
            y[num_simulations:num_cal_max],
            scales,
        )

    def generate_test_dataset(self, num_test: int) -> Tuple:
        if self.training_size is None:
            raise ValueError("Training size not set. Please call generate_dataset first.")
        if self.calibration_size is None:
            raise ValueError("Calibration size not set. Please call generate_dataset first.")
        data = pickle.load(open(self.results_path, "rb"))["data"]
        theta = data["theta"]
        x = data["x_raw"]
        y = data["y_raw"]
        theta = torch.from_numpy(np.array(theta)).to(torch.float32)
        x = torch.from_numpy(np.array(x)).to(torch.float32)
        y = torch.from_numpy(np.array(y)).to(torch.float32)
        theta_mean = theta.mean(dim=0)
        theta_std = theta.std(dim=0)
        x_mean = x.mean(dim=0)
        x_std = x.std(dim=0)
        y_mean = y.mean(dim=0)
        y_std = y.std(dim=0)
        scales = {
            "theta_mean": theta_mean,
            "theta_std": theta_std,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
        }
        idx = self.training_size + self.calibration_size
        return (
            theta[idx : idx + num_test],
            x[idx : idx + num_test],
            y[idx : idx + num_test],
            scales,
        )

    def get_simulator(self, misspecified: bool):
        raise NotImplementedError("Simulator function not implemented for SIR.")

    def sample_reference_posterior(
        self,
        num_samples: int,
        observations: torch.Tensor,
        misspecified: bool,
    ) -> torch.Tensor:
        raise NotImplementedError("Reference posterior sampling not implemented for SIR.")

    def get_noisy_process(self, theta: torch.Tensor) -> torch.Tensor:
        # This function is not used in the SIR simulator
        raise NotImplementedError("Noisy process not implemented for SIR.")
