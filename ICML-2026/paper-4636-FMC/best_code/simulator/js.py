from pathlib import Path
import pickle
from typing import Any, Dict, Optional, Tuple
from typing_extensions import override
from causalchamber.datasets import ImageExperiment
import pandas as pd
import math
from simulator.base import Simulator
import numpy as np
import torch
from torch import Tensor
from pyro import distributions as dist

from utils.transform import LogitBoxTransform, PowerSpecDens


class JS(Simulator):
    def __init__(
        self,
        theta_dim: int = 3,
        obs_dim: int = 33,
        data_dir: str = "/home/pruhlman/data/ropefm",  # Base data directory
        noise: float = 0.1,
        fs: int = 1000,
        nbins: int = 32,
    ):
        theta_dim = int(theta_dim)
        obs_dim = int(obs_dim)
        super().__init__(theta_dim=theta_dim, obs_dim=obs_dim, name="js")
        self.callable_simulator = False
        self.callable_dgp = False
        self.supported_generation = ["independent"]

        # Construct data path from data_dir and task name
        self.data_path = Path(data_dir) / "js"

        self.prior_params = {
            "low": torch.tensor([10.0, 50.0, 100]),
            "high": torch.tensor([250.0, 500.0, 5000.0]),
        }

        self.prior_dist = dist.Independent(
            dist.Uniform(self.prior_params["low"], self.prior_params["high"]), 1
        )
        self.prior_dist.set_default_validate_args(False)

        # Tranform for theta
        self.transform = LogitBoxTransform(a=self.prior_params["low"], b=self.prior_params["high"])
        self.obs_transform = PowerSpecDens(fs=fs, nbins=nbins, logscale=True, dtype=torch.float32)

        self.noise = noise

    def get_simulator(self, misspecified: bool = False, **kwargs) -> Any:
        if misspecified:
            raise NotImplementedError("No simulator available for JS.")
        else:
            raise NotImplementedError("No DGP available for JS.")

    def get_noisy_process(self, x: Tensor) -> Tensor:
        return x + self.noise * torch.randn_like(x)

    def simulations_from_files(self, n: int) -> Tuple[Tensor, Tensor]:
        with open(self.data_path / "train.pkl", "rb") as f:
            data = pickle.load(f)
        thetas = data["theta"].float()
        xs = data["x"]
        s = self.obs_transform(xs)[1]
        assert s.shape[1] == self.obs_dim
        assert n <= thetas.shape[0]
        return thetas[:n, :-1], s[:n]

    def get_total_observations(self) -> int:
        """Get total number of observations available in calibration file."""
        with open(self.data_path / "cal.pkl", "rb") as f:
            data = pickle.load(f)
        return data["theta"].shape[0]

    def obs_from_files(
        self, n: int, mode: str = "training", indices: Optional[Tuple[int, ...]] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Load observations from calibration pickle file.

        Args:
            n: Number of samples to load
            mode: "training" or "testing" (deprecated, use indices instead)
            indices: Explicit indices to load. If provided, mode is ignored.

        Returns:
            Tuple of (theta, x, y) tensors
        """
        with open(self.data_path / "cal.pkl", "rb") as f:
            data = pickle.load(f)
        thetas = data["theta"].float()
        xs = data["x"]

        total_available = thetas.shape[0]
        if n > total_available:
            raise ValueError(f"Requested {n} samples but only {total_available} available.")

        if indices is not None:
            if len(indices) != n:
                raise ValueError(f"indices length ({len(indices)}) must match n ({n})")
            selected_indices = list(indices)
        else:
            # Fallback: sequential indices (not recommended - use DataSplitter)
            import warnings
            warnings.warn(
                "obs_from_files called without explicit indices. "
                "Use DataSplitter for reproducible train/test splits.",
                DeprecationWarning,
            )
            selected_indices = list(range(n))

        # Select data at indices
        thetas = thetas[selected_indices]
        xs = xs[selected_indices]

        sx = self.obs_transform(xs)[1]
        ys = self.get_noisy_process(xs)
        sy = self.obs_transform(ys)[1]

        return thetas[:, :-1], sx, sy
