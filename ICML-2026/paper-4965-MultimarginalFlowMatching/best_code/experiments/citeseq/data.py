"""
CITE dataset loading and preprocessing for NeurIPS 2022 multimodal single-cell data.

The CITE dataset comes from the NeurIPS 2022 Multimodal Single-cell Integration
challenge (Burkhardt et al., 2022). It contains single-cell measurements from CD34+
hematopoietic stem and progenitor cells (HSPCs).

PCA is pre-computed (50 PCs). The CSV file ``cite_pca50.csv`` contains the "samples"
column (timepoint index 0–3 mapping to days 2, 3, 4, 7) and columns x1–x50.

Author(s): Raghav Kansal
"""

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)

CITE_CSV_FILENAME = "cite_pca50.csv"
CITE_CSV_URL = "https://raw.githubusercontent.com/DongyiWang-66/VGFM/main/data/cite_pca50.csv"

# Cite-seq has 4 timepoints: days 2, 3, 4, 7
CITE_DAYS = [2, 3, 4, 7]
CITE_DAY_TO_IDX = {day: idx for idx, day in enumerate(CITE_DAYS)}
CITE_IDX_TO_DAY = {idx: day for day, idx in CITE_DAY_TO_IDX.items()}


def download_citeseq_data(data_dir: Path = Path("data")) -> Path:
    """Download ``cite_pca50.csv`` if not present.

    Pulls the pre-computed 50-PC CSV from the VGFM repository
    (`DongyiWang-66/VGFM`, ``data/cite_pca50.csv``).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    filepath = data_dir / CITE_CSV_FILENAME

    if not filepath.exists():
        logger.info(f"Downloading CITE-seq data from {CITE_CSV_URL}...")
        urllib.request.urlretrieve(CITE_CSV_URL, filepath)
        logger.info(f"Data downloaded to {filepath}")

    return filepath


class MaxStdScaler:
    """Normalize by centering with mean and dividing by max std across features.

    Matches the normalization in Neklyudov et al. (2024) with ``whiten=False``.
    Unlike per-feature StandardScaler, this preserves relative scales between PCA
    components by using a single global divisor (max of per-feature stds).
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> "MaxStdScaler":
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.scale_ = float(X.std(axis=0).max())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.scale_ + self.mean_


def load_citeseq_data(
    data_dir: Path = Path("data"),
    pca_dim: int = 50,
    normalize: bool = True,
    ot_coupling: bool = False,
    ot_method: str = "emd",
    holdout_times: list[int] | None = None,
) -> dict:
    """
    Load and preprocess CITE-seq data from ``cite_pca50.csv``.

    The CSV has a ``samples`` column (timepoint index 0–3) and ``x1``–``x50``
    PCA columns, pre-computed following Tong et al. (2023).

    Args:
        data_dir: Directory containing/to download ``cite_pca50.csv``
        pca_dim: Number of PCA dimensions to use (max 50)
        normalize: Whether to normalize (center + divide by max std)
        ot_coupling: Whether to compute OT couplings
        ot_method: OT method ('emd' or 'sinkhorn')
        holdout_times: Time indices to hold out for evaluation (default: [1] = day 3)

    Returns:
        Dictionary with pcs, labels, marginals, scaler, train_times, etc.
    """
    data_path = download_citeseq_data(data_dir)

    df = pd.read_csv(data_path)
    logger.info(
        f"Loaded CITE-seq data: {len(df)} cells, {len(df.columns) - 1} PCs from {data_path}"
    )

    labels = df["samples"].values.astype(np.int64)
    unique_times = sorted(set(labels))
    logger.info(f"Unique time indices: {unique_times}")
    assert unique_times == [0, 1, 2, 3], f"Expected time indices [0,1,2,3], got {unique_times}"

    pc_cols = [f"x{i}" for i in range(1, pca_dim + 1)]
    pcs = df[pc_cols].values.astype(np.float32)
    logger.info(f"Using {pca_dim} PCs, shape: {pcs.shape}")

    scaler = None
    if normalize:
        scaler = MaxStdScaler()
        pcs = scaler.fit_transform(pcs).astype(np.float32)
        logger.info(f"Normalized with MaxStdScaler (scale={scaler.scale_:.4f})")

    marginals = {}
    for t in unique_times:
        mask = labels == t
        marginals[t] = torch.tensor(pcs[mask], dtype=torch.float32)
        logger.info(f"  Time idx {t} (day {CITE_IDX_TO_DAY[t]}): {marginals[t].shape[0]} cells")

    holdout_times = holdout_times if holdout_times is not None else [1]
    train_times = [t for t in unique_times if t not in holdout_times]
    logger.info(f"Train times: {train_times} (days {[CITE_IDX_TO_DAY[t] for t in train_times]})")
    logger.info(
        f"Holdout times: {holdout_times} (days {[CITE_IDX_TO_DAY[t] for t in holdout_times]})"
    )

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
    """Compute OT alignments between consecutive train time points."""
    try:
        import ot
    except ImportError:
        logger.warning("POT not installed, skipping OT alignment computation")
        return None

    alignments = {}

    for i in range(len(train_times) - 1):
        t_src, t_tgt = train_times[i], train_times[i + 1]
        logger.info(
            f"Computing OT alignment from t={t_src} (day {CITE_IDX_TO_DAY[t_src]}) "
            f"to t={t_tgt} (day {CITE_IDX_TO_DAY[t_tgt]})..."
        )

        source = pcs[labels == t_src]
        target = pcs[labels == t_tgt]

        cost = ot.dist(source, target, metric="sqeuclidean")
        a = np.ones(len(source)) / len(source)
        b = np.ones(len(target)) / len(target)

        if method == "emd":
            plan = ot.emd(a, b, cost)
        else:
            plan = ot.sinkhorn(a, b, cost, reg=reg)

        mapping = np.argmax(plan, axis=1)
        alignments[(t_src, t_tgt)] = mapping

    return alignments


class CiteSeqMultiMarginalDataset(Dataset):
    """
    PyTorch Dataset for CITE multi-marginal training.

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

        self.data_by_time = {}
        for t in self.train_times:
            mask = labels == t
            self.data_by_time[t] = torch.tensor(pcs[mask], dtype=torch.float32)

        if self.use_ot:
            self._build_ot_chains()
            self.dataset_size = len(self.data_by_time[self.train_times[0]])
        else:
            self.dataset_size = min(len(d) for d in self.data_by_time.values())
            self.indices = {t: torch.randperm(len(self.data_by_time[t])) for t in self.train_times}

    def _build_ot_chains(self):
        """Build OT coupling chains across training times."""
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


def create_citeseq_dataloaders(
    pcs: np.ndarray,
    labels: np.ndarray,
    holdout_times: list[int] | None = None,
    batch_size: int = 256,
    val_split: float = 0.2,
    ot_alignments: dict | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders for CITE dataset."""
    dataset = CiteSeqMultiMarginalDataset(
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
