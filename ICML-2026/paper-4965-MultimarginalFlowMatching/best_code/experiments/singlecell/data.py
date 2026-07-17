"""
Single-cell dataset loading and preprocessing for Embryoid Body (EB) data.

Author(s): Raghav Kansal
"""

import logging
import urllib.request
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)

DATA_URL = "https://github.com/KrishnaswamyLab/TrajectoryNet/raw/master/data/eb_velocity_v5.npz"


def download_eb_data(data_dir: Path = Path("data")) -> Path:
    """Download eb_velocity_v5.npz if not present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    filepath = data_dir / "eb_velocity_v5.npz"

    if not filepath.exists():
        logger.info(f"Downloading EB data from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, filepath)
        logger.info(f"Data downloaded to {filepath}")

    return filepath


def load_eb_data(
    data_dir: Path = Path("data"),
    pca_dim: int = 100,
    normalize: bool = True,
    ot_coupling: bool = False,
    ot_method: str = "emd",
    holdout_times: list[int] | None = None,
) -> dict:
    """
    Load and preprocess EB data.

    Args:
        data_dir: Directory containing/to download data
        pca_dim: Number of PCA dimensions to use
        normalize: Whether to normalize the data
        ot_coupling: Whether to compute OT couplings
        ot_method: OT method ('emd' or 'sinkhorn')
        holdout_times: Times to hold out for evaluation

    Returns:
        Dictionary with pcs, labels, marginals, scaler, train_times, etc.
    """
    # Download data if needed
    data_path = download_eb_data(data_dir)

    # Load data
    data = np.load(data_path, allow_pickle=True)
    pcs = data["pcs"][:, :pca_dim].astype(np.float32)
    labels = data["sample_labels"].astype(np.int64)

    # Normalize
    scaler = None
    if normalize:
        scaler = StandardScaler()
        pcs = scaler.fit_transform(pcs).astype(np.float32)

    # Create marginals dictionary
    unique_times = sorted(set(labels))
    marginals = {}
    for t in unique_times:
        mask = labels == t
        marginals[t] = torch.tensor(pcs[mask], dtype=torch.float32)

    # Determine train times
    holdout_times = holdout_times if holdout_times is not None else [1, 3]
    train_times = [t for t in unique_times if t not in holdout_times]

    # Compute OT alignments if requested
    ot_alignments = None
    if ot_coupling:
        ot_alignments = compute_ot_alignments(pcs, labels, train_times, method=ot_method)

    return {
        "pcs": pcs,
        "labels": labels,
        "marginals": marginals,
        "scaler": scaler,
        "train_times": train_times,
        "holdout_times": holdout_times,
        "ot_alignments": ot_alignments,
    }


def compute_ot_alignments(
    pcs: np.ndarray,
    labels: np.ndarray,
    train_times: list[int],
    method: str = "emd",
    reg: float = 0.01,
) -> dict:
    """
    Compute OT alignments between consecutive train time points.

    Args:
        pcs: PCA coordinates (n_cells, dim)
        labels: Time labels (n_cells,)
        train_times: List of training times
        method: 'emd' or 'sinkhorn'
        reg: Regularization for sinkhorn

    Returns:
        Dictionary mapping (t_src, t_tgt) -> mapping array
    """
    try:
        import ot
    except ImportError:
        logger.warning("POT not installed, skipping OT alignment computation")
        return None

    alignments = {}

    for i in range(len(train_times) - 1):
        t_src, t_tgt = train_times[i], train_times[i + 1]
        logger.info(f"Computing OT alignment from t={t_src} to t={t_tgt}...")

        source = pcs[labels == t_src]
        target = pcs[labels == t_tgt]

        # Cost matrix
        cost = ot.dist(source, target, metric="sqeuclidean")

        # Uniform marginals
        a = np.ones(len(source)) / len(source)
        b = np.ones(len(target)) / len(target)

        # Compute transport plan
        if method == "emd":
            plan = ot.emd(a, b, cost)
        else:
            plan = ot.sinkhorn(a, b, cost, reg=reg)

        # Get deterministic mapping
        mapping = np.argmax(plan, axis=1)
        alignments[(t_src, t_tgt)] = mapping

    return alignments


class EBMultiMarginalDataset(Dataset):
    """
    PyTorch Dataset for EB multi-marginal training.

    Each sample contains one cell from each training time point,
    either with independent or OT-coupled pairing.
    """

    def __init__(
        self,
        pcs: np.ndarray,
        labels: np.ndarray,
        holdout_times: list[int] | None = None,
        ot_alignments: dict | None = None,
    ):
        self.times = sorted(set(labels))
        self.holdout_times = holdout_times or []
        self.train_times = [t for t in self.times if t not in self.holdout_times]
        self.ot_alignments = ot_alignments
        self.use_ot = ot_alignments is not None

        # Group data by time
        self.data_by_time = {}
        for t in self.train_times:
            mask = labels == t
            self.data_by_time[t] = torch.tensor(pcs[mask], dtype=torch.float32)

        # Build OT chains if using OT coupling
        if self.use_ot:
            self._build_ot_chains()
            self.dataset_size = len(self.data_by_time[self.train_times[0]])
        else:
            self.dataset_size = min(len(d) for d in self.data_by_time.values())
            self.indices = {t: torch.randperm(len(self.data_by_time[t])) for t in self.train_times}

    def _build_ot_chains(self):
        """Build OT coupling chains."""
        n_source = len(self.data_by_time[self.train_times[0]])
        n_times = len(self.train_times)
        self.ot_chains = np.zeros((n_source, n_times), dtype=np.int64)
        self.ot_chains[:, 0] = np.arange(n_source)

        normalized_alignments = {(int(k[0]), int(k[1])): v for k, v in self.ot_alignments.items()}

        for t_idx in range(n_times - 1):
            t_src, t_tgt = self.train_times[t_idx], self.train_times[t_idx + 1]
            mapping = normalized_alignments[(t_src, t_tgt)]
            for src_idx in range(n_source):
                prev_idx = self.ot_chains[src_idx, t_idx]
                self.ot_chains[src_idx, t_idx + 1] = mapping[prev_idx]

        self.chain_perm = np.random.permutation(n_source)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        samples = []
        if self.use_ot:
            chain_idx = self.chain_perm[idx % len(self.chain_perm)]
            for t_idx, t in enumerate(self.train_times):
                actual_idx = self.ot_chains[chain_idx, t_idx]
                samples.append(self.data_by_time[t][actual_idx])
        else:
            for t in self.train_times:
                actual_idx = self.indices[t][idx % len(self.data_by_time[t])]
                samples.append(self.data_by_time[t][actual_idx])
        return samples

    def reshuffle(self):
        """Reshuffle for new epoch."""
        if self.use_ot:
            self.chain_perm = np.random.permutation(len(self.chain_perm))
        else:
            self.indices = {t: torch.randperm(len(self.data_by_time[t])) for t in self.train_times}

    def get_ot_aligned_samples(self, n_samples: int = 5) -> torch.Tensor:
        """Get OT-aligned samples across all training times."""
        if not self.use_ot:
            raise ValueError("Dataset was created without OT alignments")

        n_times = len(self.train_times)
        n_samples = min(n_samples, len(self.ot_chains))
        dim = self.data_by_time[self.train_times[0]].shape[-1]

        samples = torch.zeros(n_samples, n_times, dim)
        for i in range(n_samples):
            for t_idx, t in enumerate(self.train_times):
                cell_idx = self.ot_chains[i, t_idx]
                samples[i, t_idx] = self.data_by_time[t][cell_idx]

        return samples


def create_eb_dataloaders(
    pcs: np.ndarray,
    labels: np.ndarray,
    holdout_times: list[int] | None = None,
    batch_size: int = 256,
    val_split: float = 0.2,
    ot_alignments: dict | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train/val DataLoaders for EB dataset.

    Args:
        pcs: PCA coordinates (n_cells, dim)
        labels: Time labels (n_cells,)
        holdout_times: Times to exclude from training
        batch_size: Batch size
        val_split: Validation split fraction
        ot_alignments: Optional OT alignments dict

    Returns:
        Tuple of (train_loader, val_loader)
    """
    dataset = EBMultiMarginalDataset(
        pcs, labels, holdout_times=holdout_times, ot_alignments=ot_alignments
    )

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader
