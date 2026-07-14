import os
import torch
import numpy as np
from torch.utils.data import Dataset, random_split
from power_spherical import HypersphericalUniform
from scipy.stats import qmc  # (kept for parity; not used here)

def _renorm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def uniform_noise(ssp_dim: int) -> torch.Tensor:
    """Unit-norm noise on S^{d-1} via HypersphericalUniform."""
    return HypersphericalUniform(dim=ssp_dim).rsample((1,)).squeeze(0)

def gaussian_noise(ssp_dim: int) -> torch.Tensor:
    """Gaussian → unit-norm."""
    z = torch.randn(ssp_dim, dtype=torch.float32)
    return _renorm(z)

class SSPDataset(Dataset):
    """
    Pairs a noise start **z0** with a target SSP **z1** loaded from ``data_dir``.

    **z1** always comes from disk (same encoding as when the dataset was built).

    **z0** is always a fresh noise draw from ``noise_type`` only:
    ``uniform_hypersphere`` (unit sphere) or ``gaussian`` (Gaussian then normalized).
    It is never blended with **z1** here. ``signal_strength`` is kept for API
    compatibility with trainers/eval callers but is not used in ``__getitem__``;
    evaluation can form a separate initial state for the flow from (z0, z1) if needed.
    """
    def __init__(
        self,
        data_dir: str,
        ssp_dim: int,
        target_type: str = "coordinate",
        noise_type: str = "uniform_hypersphere",  # {"uniform_hypersphere","gaussian"}
        signal_strength: float = 0.0,             # unused in __getitem__; see class docstring
        mode: str = "train",
        device: str = "cpu",                      # kept for API parity; tensors on CPU
    ):
        self.data_dir        = data_dir
        self.ssp_dim         = ssp_dim
        self.target_type     = target_type
        self.noise_type      = noise_type
        self.signal_strength = float(signal_strength)
        self.mode            = mode
        self.device          = device

        if self.mode not in ("train", "test"):
            raise ValueError(f"Unknown mode: {self.mode}")

        self.data_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".npy")])
        if not self.data_files:
            raise FileNotFoundError(f"No .npy files found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int):
        # Load target SSP and renormalize (safety)
        target_path = os.path.join(self.data_dir, self.data_files[idx])
        z1 = torch.tensor(np.load(target_path), dtype=torch.float32)
        z1 = _renorm(z1)

        if self.noise_type == "uniform_hypersphere":
            z0 = uniform_noise(self.ssp_dim)
        elif self.noise_type == "gaussian":
            z0 = gaussian_noise(self.ssp_dim)
        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")

        return z0, z1

    def split_dataset(self, val_split: float = 0.1):
        """Random split into train/val subsets."""
        n_val = int(len(self) * val_split)
        n_tr  = len(self) - n_val
        return random_split(self, [n_tr, n_val])
