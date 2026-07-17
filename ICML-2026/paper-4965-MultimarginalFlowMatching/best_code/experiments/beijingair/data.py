"""
Beijing air quality dataset loading and preprocessing.

Reference:
    - UCI ML Repository: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
    - 3MSBM: Theodoropoulos et al. "Momentum multi-marginal Schrödinger bridge matching" arXiv:2506.10168 (2025)

Author(s): Raghav Kansal
"""

import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)

BEIJING_DATA_URL = (
    "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
)
DIR_NAME = "PRSA_Data_20130301-20170228"
ALL_TIMES = list(range(13))
DEFAULT_HOLDOUT_TIMES = [2, 5, 8, 11]
DEFAULT_TRAIN_TIMES = [t for t in ALL_TIMES if t not in DEFAULT_HOLDOUT_TIMES]
METRIC_TIMES = [f"t{i}" for i in ALL_TIMES if i != 0] + ["rest"]


def download_beijing_data(data_dir: Path = Path("data/beijing")) -> Path:
    """Download Beijing air quality data from UCI repository."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = data_dir / DIR_NAME

    if extracted_dir.exists():
        return extracted_dir

    zip_path = data_dir / "beijing_air_quality.zip"
    logger.info("Downloading Beijing air quality data from UCI...")
    urllib.request.urlretrieve(BEIJING_DATA_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)
    zip_path.unlink(missing_ok=True)

    nested_zip_path = data_dir / f"{DIR_NAME}.zip"
    if nested_zip_path.exists():
        with zipfile.ZipFile(nested_zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        nested_zip_path.unlink(missing_ok=True)

    return extracted_dir


def load_beijing_data(
    data_dir: Path = Path("data/beijing"),
    station: str = "Dingling",
    normalize: bool = True,
    ot_coupling: bool = False,
    ot_method: str = "emd",
    train_times: list[int] | None = None,
    holdout_times: list[int] | None = None,
) -> dict:
    """
    Load and preprocess Beijing PM2.5 data.

    Args:
        data_dir: Directory containing data
        station: Station name (default: Dingling)
        normalize: Whether to normalize values
        ot_coupling: Whether to compute OT couplings
        ot_method: OT method
        train_times: Training time indices
        holdout_times: Holdout time indices

    Returns:
        Dictionary with marginals, scaler, train_times, etc.
    """
    data_dir = Path(data_dir)

    # Check for preprocessed file
    preprocessed_path = data_dir / f"{station.lower()}_pm25_13marginals.npy"
    if preprocessed_path.exists():
        data = np.load(preprocessed_path, allow_pickle=True)
        marginals_list = [arr.reshape(-1, 1).astype(np.float32) for arr in data]
    else:
        # Load from CSV
        csv_dir = data_dir / DIR_NAME
        if not csv_dir.exists():
            download_beijing_data(data_dir)

        marginals_list = _create_pm25_marginals(csv_dir, station, preprocessed_path)

    # Normalize
    scaler = None
    if normalize:
        all_data = np.concatenate(marginals_list, axis=0)
        scaler = StandardScaler()
        scaler.fit(all_data)
        marginals_list = [scaler.transform(m).astype(np.float32) for m in marginals_list]

    # Create marginals dict
    marginals = {i: torch.tensor(m, dtype=torch.float32) for i, m in enumerate(marginals_list)}

    # Default times
    holdout_times = holdout_times or DEFAULT_HOLDOUT_TIMES
    train_times = train_times or DEFAULT_TRAIN_TIMES

    # Compute OT alignments
    ot_alignments = None
    if ot_coupling:
        ot_alignments = compute_beijing_ot_alignments(marginals_list, train_times, ot_method)

    return {
        "marginals_list": marginals_list,
        "marginals": marginals,
        "scaler": scaler,
        "train_times": train_times,
        "holdout_times": holdout_times,
        "ot_alignments": ot_alignments,
        "dim": 1,
    }


def _create_pm25_marginals(csv_dir: Path, station: str, save_path: Path) -> list[np.ndarray]:
    """Create PM2.5 marginals from raw CSV data."""
    file_path = csv_dir / f"PRSA_Data_{station}_20130301-20170228.csv"
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["PM2.5"])
    df = df.sort_values(["year", "month", "day", "hour"])

    start_year, start_month = 2013, 3
    df["month_idx"] = (df["year"] - start_year) * 12 + (df["month"] - start_month)

    marginals = []
    month_indices = list(range(0, 25, 2))  # Every other month for 13 marginals

    for m_idx in month_indices:
        m_data = df[df["month_idx"] == m_idx]["PM2.5"].values
        if len(m_data) > 0:
            marginals.append(m_data.reshape(-1, 1).astype(np.float32))

    # Save preprocessed
    marginals_arr = np.empty(len(marginals), dtype=object)
    marginals_arr[:] = marginals
    np.save(save_path, marginals_arr, allow_pickle=True)

    return marginals


def compute_beijing_ot_alignments(
    marginals: list[np.ndarray],
    train_times: list[int],
    method: str = "emd",
    reg: float = 0.01,
) -> dict:
    """Compute OT alignments between consecutive training times."""
    try:
        import ot
    except ImportError:
        logger.warning("POT not installed, skipping OT alignment computation")
        return None

    alignments = {}
    for i in range(len(train_times) - 1):
        t_src, t_tgt = train_times[i], train_times[i + 1]
        logger.info(f"Computing OT alignment from t={t_src} to t={t_tgt}...")

        source = marginals[t_src]
        target = marginals[t_tgt]
        cost = ot.dist(source, target, metric="sqeuclidean")
        a = np.ones(len(source)) / len(source)
        b = np.ones(len(target)) / len(target)

        if method == "emd":
            plan = ot.emd(a, b, cost)
        else:
            plan = ot.sinkhorn(a, b, cost, reg=reg)

        alignments[(t_src, t_tgt)] = np.argmax(plan, axis=1)

    return alignments


class BeijingMultiMarginalDataset(Dataset):
    """PyTorch Dataset for Beijing PM2.5 multi-marginal training."""

    def __init__(
        self,
        marginals: list[np.ndarray],
        train_times: list[int] | None = None,
        ot_alignments: dict | None = None,
    ):
        self.all_times = list(range(len(marginals)))
        self.train_times = train_times or DEFAULT_TRAIN_TIMES
        self.ot_alignments = ot_alignments
        self.use_ot = ot_alignments is not None

        self.data_by_time = {
            t: torch.tensor(marginals[t], dtype=torch.float32) for t in self.train_times
        }

        if self.use_ot:
            self._build_ot_chains()
            self.dataset_size = len(self.data_by_time[self.train_times[0]])
        else:
            self.dataset_size = min(len(d) for d in self.data_by_time.values())
            self.indices = {t: torch.randperm(len(self.data_by_time[t])) for t in self.train_times}

    def _build_ot_chains(self):
        normalized = {(int(k[0]), int(k[1])): v for k, v in self.ot_alignments.items()}
        n_source = len(self.data_by_time[self.train_times[0]])
        n_times = len(self.train_times)
        self.ot_chains = np.zeros((n_source, n_times), dtype=np.int64)
        self.ot_chains[:, 0] = np.arange(n_source)

        for t_idx in range(n_times - 1):
            t_src, t_tgt = self.train_times[t_idx], self.train_times[t_idx + 1]
            mapping = normalized[(t_src, t_tgt)]
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
                samples.append(self.data_by_time[t][self.ot_chains[chain_idx, t_idx]])
        else:
            for t in self.train_times:
                samples.append(
                    self.data_by_time[t][self.indices[t][idx % len(self.data_by_time[t])]]
                )
        return samples

    def reshuffle(self):
        if self.use_ot:
            self.chain_perm = np.random.permutation(len(self.chain_perm))
        else:
            self.indices = {t: torch.randperm(len(self.data_by_time[t])) for t in self.train_times}

    def get_ot_aligned_samples(self, n_samples: int = 5) -> Tensor:
        if not self.use_ot:
            raise ValueError("OT coupling not enabled")
        n_times = len(self.train_times)
        n_samples = min(n_samples, len(self.ot_chains))
        dim = self.data_by_time[self.train_times[0]].shape[-1]
        samples = torch.zeros(n_samples, n_times, dim)
        for i in range(n_samples):
            for t_idx, t in enumerate(self.train_times):
                samples[i, t_idx] = self.data_by_time[t][self.ot_chains[i, t_idx]]
        return samples


def create_beijing_dataloaders(
    marginals: list[np.ndarray],
    batch_size: int = 128,
    holdout_times: list[int] | None = None,
    val_split: float = 0.0,
    ot_alignments: dict | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders for Beijing dataset."""
    train_times = [t for t in ALL_TIMES if t not in (holdout_times or DEFAULT_HOLDOUT_TIMES)]
    dataset = BeijingMultiMarginalDataset(
        marginals, train_times=train_times, ot_alignments=ot_alignments
    )

    if val_split > 0:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
    else:
        train_dataset = dataset
        val_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader
