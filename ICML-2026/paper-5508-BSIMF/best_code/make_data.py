# make_data.py
"""
Data loading interface and splitting utilities.

Users must implement get_dataloader() to load their own dataset and return it in
the format expected by the model. The concrete helpers provided here are:

  - ConnScoresDataset            : PyTorch Dataset wrapping the merged dict format
  - build_big_dict               : merge connectivity and score dicts into that format
  - split_subject_ids_holdout    : random subject-level holdout split
  - split_subject_ids_kfold      : subject-level K-fold split

Data format
-----------
The central data structure is a dict with one entry per subject:

    {
      subjectID: {
          "conn":          np.ndarray, shape (H, W)   — symmetric connectivity matrix
          "scores":        np.ndarray, shape (x_dim,) — behavioral score vector (NaN allowed)
          "assignedGroup": int                        — group label (used for evaluation only)
      },
      ...
    }

Model input/output shapes
-------------------------
  x : (batch, x_dim)     — behavioral scores; NaN entries are masked inside the model
  y : (batch, 1, H, W)   — connectivity matrix with a channel dimension added by the Dataset
  group : (batch,)        — integer label; never used as a training signal
"""

import os
import pickle
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Core Dataset
# ---------------------------------------------------------------------------

class ConnScoresDataset(Dataset):
    """
    PyTorch Dataset wrapping the merged subject dict.

    Each sample is returned as (x, y, group) by default, where:
      x     : behavioral score vector, shape (x_dim,), float32, may contain NaN
      y     : connectivity matrix, shape (1, H, W) unless flatten_conn=True
      group : integer group label (included for evaluation; never a training signal)

    If return_subject_id=True, samples are (subject_id, x, y, group).
    """

    def __init__(
        self,
        big_dict: Dict[str, Dict],
        subject_ids: Optional[List[str]] = None,
        flatten_conn: bool = False,
        upper_triangle_only: bool = False,
        add_channel_dim: bool = True,
        return_subject_id: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        self.data = big_dict
        self.subject_ids = subject_ids if subject_ids is not None else sorted(big_dict.keys())
        self.flatten_conn = flatten_conn
        self.upper_triangle_only = upper_triangle_only
        self.add_channel_dim = add_channel_dim
        self.return_subject_id = return_subject_id
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int):
        sid = self.subject_ids[idx]
        item = self.data[sid]

        y = np.asarray(item["conn"], dtype=np.float32)
        x = np.asarray(item["scores"], dtype=np.float32)
        group = int(item.get("assignedGroup", 0))

        y_t = torch.as_tensor(y, dtype=self.dtype)
        if self.flatten_conn:
            if self.upper_triangle_only:
                triu_idx = torch.triu_indices(y_t.shape[0], y_t.shape[1], offset=0)
                y_t = y_t[triu_idx[0], triu_idx[1]]
            else:
                y_t = y_t.reshape(-1)
        else:
            if self.add_channel_dim and y_t.ndim == 2:
                y_t = y_t.unsqueeze(0)  # (1, H, W)

        x_t = torch.as_tensor(x, dtype=self.dtype)
        group_t = torch.tensor(group, dtype=torch.long)

        if self.return_subject_id:
            return sid, x_t, y_t, group_t
        return x_t, y_t, group_t


# ---------------------------------------------------------------------------
# Dict builder
# ---------------------------------------------------------------------------

def build_big_dict(
    conn: Dict[str, np.ndarray],
    scores: Dict[str, np.ndarray],
    assigned_group: int = 0,
    group_map: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[str, Dict], Set[str], Set[str]]:
    """
    Merge connectivity and score dicts into the subject dict format.

    Parameters
    ----------
    conn : {subjectID: np.ndarray of shape (H, W)}
    scores : {subjectID: np.ndarray of shape (x_dim,)}  — NaN entries allowed
    assigned_group : default group label when group_map is absent or has no entry
    group_map : optional {subjectID: int} override per subject

    Returns
    -------
    big_dict : subjects with BOTH conn and a score row
    scores_only_ids : subjects in scores but missing a conn matrix
    conn_only_ids   : subjects in conn but missing a score row
    """
    conn_ids = set(conn.keys())
    score_ids = set(scores.keys())
    intersect_ids = conn_ids & score_ids

    big: Dict[str, Dict] = {}
    for sid in intersect_ids:
        grp = assigned_group if group_map is None else int(group_map.get(sid, assigned_group))
        big[sid] = {
            "conn": np.asarray(conn[sid], dtype=np.float32),
            "scores": np.asarray(scores[sid], dtype=np.float32),
            "assignedGroup": grp,
        }

    return big, score_ids - intersect_ids, conn_ids - intersect_ids


# ---------------------------------------------------------------------------
# Splitting utilities
# ---------------------------------------------------------------------------

def _shuffle_ids(ids: List[str], seed: int = 0, shuffle: bool = True) -> List[str]:
    ids = list(ids)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)
    return ids


def split_subject_ids_holdout(
    subject_ids: List[str],
    val_frac: float = 0.2,
    seed: int = 0,
    shuffle: bool = True,
) -> Tuple[List[str], List[str]]:
    """Random subject-level holdout split. Returns (train_ids, val_ids)."""
    ids = _shuffle_ids(subject_ids, seed=seed, shuffle=shuffle)
    n_val = int(round(len(ids) * val_frac))
    return ids[n_val:], ids[:n_val]


def split_subject_ids_kfold(
    subject_ids: List[str],
    n_splits: int = 5,
    fold_idx: int = 0,
    seed: int = 0,
    shuffle: bool = True,
) -> Tuple[List[str], List[str]]:
    """Subject-level K-fold split. Returns (train_ids, val_ids) for a given fold."""
    ids = _shuffle_ids(subject_ids, seed=seed, shuffle=shuffle)
    n = len(ids)
    fold_idx = int(fold_idx) % n_splits

    fold_sizes = [n // n_splits] * n_splits
    for i in range(n % n_splits):
        fold_sizes[i] += 1

    start = sum(fold_sizes[:fold_idx])
    end = start + fold_sizes[fold_idx]
    val_ids = ids[start:end]
    train_ids = ids[:start] + ids[end:]
    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Data loader stubs — implement these for your dataset
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Global cache for the merged data dict (loaded once)
# ---------------------------------------------------------------------------
_COMBINED_CACHE: Optional[Dict[str, Dict]] = None
_COMBINED_CACHE_PATH: Optional[str] = None


def _load_combined_dict(data_path: Optional[str] = None) -> Dict[str, Dict]:
    """Load (and cache) the merged subject dict from a pickle file."""
    global _COMBINED_CACHE, _COMBINED_CACHE_PATH

    if data_path is None:
        data_path = os.environ.get(
            "ABIDE_DATA_PKL",
            "/datasets/abide/abide_i_merged.pkl",
        )

    if _COMBINED_CACHE is not None and _COMBINED_CACHE_PATH == data_path:
        return _COMBINED_CACHE

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"ABIDE merged data not found at {data_path}. "
            f"Run prepare_abide_data.py first or set ABIDE_DATA_PKL env var."
        )

    with open(data_path, "rb") as f:
        _COMBINED_CACHE = pickle.load(f)
    _COMBINED_CACHE_PATH = data_path
    return _COMBINED_CACHE


def get_dataloader(
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    flatten_conn: bool = False,
    upper_triangle_only: bool = False,
    return_subject_id: bool = False,
    data_path: Optional[str] = None,
) -> tuple:
    """
    Load the full ABIDE dataset from a pickle file and return a DataLoader.

    Returns
    -------
    dataloader : DataLoader
    dataset    : ConnScoresDataset
    combined   : dict
    """
    combined = _load_combined_dict(data_path)
    subject_ids = sorted(combined.keys())

    ds = ConnScoresDataset(
        big_dict=combined,
        subject_ids=subject_ids,
        flatten_conn=flatten_conn,
        upper_triangle_only=upper_triangle_only,
        add_channel_dim=True,
        return_subject_id=return_subject_id,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return loader, ds, combined


def get_train_val_loaders(
    batch_size: int = 32,
    val_frac: float = 0.2,
    seed: int = 0,
    shuffle: bool = True,
    num_workers: int = 0,
    flatten_conn: bool = False,
    upper_triangle_only: bool = False,
    return_subject_id: bool = False,
    add_channel_dim: bool = True,
    data_path: Optional[str] = None,
) -> tuple:
    """
    Build separate train/val DataLoaders using a random holdout split.
    """
    _, _, combined = get_dataloader(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        flatten_conn=flatten_conn,
        upper_triangle_only=upper_triangle_only,
        return_subject_id=return_subject_id,
        data_path=data_path,
    )
    subject_ids = sorted(combined.keys())
    train_ids, val_ids = split_subject_ids_holdout(
        subject_ids, val_frac=val_frac, seed=seed, shuffle=shuffle
    )

    train_ds = ConnScoresDataset(
        big_dict=combined,
        subject_ids=train_ids,
        flatten_conn=flatten_conn,
        upper_triangle_only=upper_triangle_only,
        add_channel_dim=add_channel_dim,
        return_subject_id=return_subject_id,
    )
    val_ds = ConnScoresDataset(
        big_dict=combined,
        subject_ids=val_ids,
        flatten_conn=flatten_conn,
        upper_triangle_only=upper_triangle_only,
        add_channel_dim=add_channel_dim,
        return_subject_id=return_subject_id,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, val_loader, train_ds, val_ds, combined


def get_kfold_loaders(
    batch_size: int = 32,
    n_splits: int = 5,
    fold_idx: int = 0,
    seed: int = 0,
    shuffle: bool = True,
    num_workers: int = 0,
    flatten_conn: bool = False,
    upper_triangle_only: bool = False,
    return_subject_id: bool = False,
    add_channel_dim: bool = True,
    data_path: Optional[str] = None,
) -> tuple:
    """
    Build train/val DataLoaders for one fold of a subject-level K-fold split.
    """
    _, _, combined = get_dataloader(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        flatten_conn=flatten_conn,
        upper_triangle_only=upper_triangle_only,
        return_subject_id=return_subject_id,
        data_path=data_path,
    )
    subject_ids = sorted(combined.keys())
    train_ids, val_ids = split_subject_ids_kfold(
        subject_ids, n_splits=n_splits, fold_idx=fold_idx, seed=seed, shuffle=shuffle
    )

    train_ds = ConnScoresDataset(
        big_dict=combined,
        subject_ids=train_ids,
        flatten_conn=flatten_conn,
        upper_triangle_only=upper_triangle_only,
        add_channel_dim=add_channel_dim,
        return_subject_id=return_subject_id,
    )
    val_ds = ConnScoresDataset(
        big_dict=combined,
        subject_ids=val_ids,
        flatten_conn=flatten_conn,
        upper_triangle_only=upper_triangle_only,
        add_channel_dim=add_channel_dim,
        return_subject_id=return_subject_id,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, val_loader, train_ds, val_ds, combined


def get_test_loader(
    batch_size: int = 32,
    num_workers: int = 0,
    flatten_conn: bool = False,
    upper_triangle_only: bool = False,
    return_subject_id: bool = False,
    add_channel_dim: bool = True,
) -> tuple:
    """
    Build a deterministic DataLoader for held-out test subjects.

    Implement by loading test connectivity matrices and behavioral scores,
    merging them with build_big_dict(), wrapping in ConnScoresDataset with
    shuffle=False, and returning the loader.

    Returns
    -------
    test_loader : DataLoader
    test_ds     : ConnScoresDataset
    test_dict   : dict
    """
    raise NotImplementedError(
        "Implement get_test_loader() to load your held-out test subjects and "
        "return a non-shuffled DataLoader over them."
    )
