from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from wilds.common.grouper import CombinatorialGrouper

def _import_wilds_get_dataset():
    try:
        from wilds import get_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Missing dependency 'wilds'. Install with `pip install wilds` "
            "or `pip install -e \".[realworld]\"`."
        ) from e
    return get_dataset


def _resolve_split(dataset, candidates: Sequence[str]) -> str:
    for s in candidates:
        try:
            dataset.get_subset(s)
            return s
        except Exception:
            pass
    raise ValueError(f"Could not resolve split from {list(candidates)}")


@dataclass(frozen=True)
class WildsSplits:
    train: str
    val: str
    test: str


def load_wilds_dataset(wilds_name: str, root_dir: Path, download: bool = True):
    get_dataset = _import_wilds_get_dataset()
    ds = get_dataset(dataset=wilds_name, root_dir=str(root_dir), download=download)
    splits = WildsSplits(
        train=_resolve_split(ds, ["train"]),
        val=_resolve_split(ds, ["val", "id_val", "val_id"]),
        test=_resolve_split(ds, ["test", "id_test", "test_id"]),
    )
    return ds, splits


def make_train_cal_split(n: int, cal_fraction: float, split_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < cal_fraction < 1.0):
        raise ValueError(f"cal_fraction must be in (0,1), got {cal_fraction}")
    rng = np.random.default_rng(int(split_seed))
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_cal = int(np.floor(float(cal_fraction) * n))
    return idx[n_cal:], idx[:n_cal]


def _find_meta_field(dataset, needle: str) -> Tuple[int, str]:
    """Locate a metadata field index by substring match (case-insensitive)."""
    fields = getattr(dataset, "metadata_fields", None) or getattr(dataset, "metadata_field_names", None)
    if fields is None:
        raise RuntimeError(
            f"Cannot locate metadata field {needle!r}: dataset has no metadata_fields/metadata_field_names."
        )
    for i, name in enumerate(fields):
        if needle.lower() in str(name).lower():
            return int(i), str(name)
    raise RuntimeError(
        f"Cannot locate metadata field {needle!r}. Available metadata fields: {list(map(str, fields))}"
    )

def _groups_from_metadata(dataset, metadata_array: torch.Tensor, y_array: torch.Tensor, wilds_name: str) -> torch.Tensor:
    """
    Compute disjoint group labels.

    CivilComments (current pipeline default here):
        gid = 2*y + identity_any (4 groups)
    """
    name_l = wilds_name.lower()

    if name_l == "civilcomments":
        return _civilcomments_groups_identity_any(y_array, metadata_array)

    # ---- Generic WILDS fallback ----
    if hasattr(dataset, "eval_grouper") and dataset.eval_grouper is not None:
        g = dataset.eval_grouper.metadata_to_group(metadata_array)
        return g.long().view(-1)

    # Last resort: single group (keeps code running, but will be degenerate)
    return torch.zeros(metadata_array.shape[0], dtype=torch.long)

def _civilcomments_groups_identity_any(y: "torch.Tensor", metadata: "torch.Tensor") -> "torch.Tensor":
    """
    CivilComments groups for worst-group evaluation.

    WILDS CivilComments domain metadata corresponds to 8 identity dimensions
    (male, female, LGBTQ, Christian, Muslim, other religions, Black, White). 

    IMPORTANT:
    - In many WILDS versions, metadata values are floats in [0,1] (identity scores),
      not pre-binarised ints.
    - Therefore we must threshold in FLOAT space first, then cast to long.

    We build a disjoint 4-group partition:
      identity_any = 1 if ANY identity dimension is "present"
      g = 2*y + identity_any  in {0,1,2,3}
    """
    import torch

    y = torch.as_tensor(y).view(-1).to(dtype=torch.long)

    meta = metadata
    if not torch.is_tensor(meta):
        meta = torch.as_tensor(meta)

    # KEEP FLOAT until after thresholding
    meta = meta.to(dtype=torch.float32)

    if meta.ndim != 2 or meta.shape[1] < 9:
        raise ValueError(f"CivilComments metadata must be 2D with >=9 columns; got shape {tuple(meta.shape)}")

    identity_any = meta[:, 8].to(dtype=torch.long).view(-1)
    if identity_any.shape[0] != y.shape[0]:
        raise ValueError(f"y and metadata size mismatch: y={y.shape[0]}, identity_any={identity_any.shape[0]}")

    g = (2 * y + identity_any).to(dtype=torch.long)
    # HARD FAIL if still degenerate: otherwise we’ll silently get meaningless “worst-group” plots
    if int(torch.unique(g).numel()) <= 1:
        # print useful diagnostics
        mn = float(meta.min().item())
        mx = float(meta.max().item())
        mean = float(meta.mean().item())
        raise RuntimeError(
            "CivilComments group construction degenerated to a single group. "
            "This means your metadata has no positive identity signal under the chosen threshold.\n"
            f"metadata stats: min={mn:.6g}, max={mx:.6g}, mean={mean:.6g}\n"
            "Fix by inspecting metadata_array and choosing the correct thresholding rule."
        )

    return g



class IndexedWILDSSubset(Dataset):
    """Returns (x, y, g, idx_in_base_subset) for selected indices."""

    def __init__(self, base_subset: Dataset, indices: np.ndarray, y: torch.Tensor, g: torch.Tensor):
        super().__init__()
        self.base_subset = base_subset
        self.indices = np.asarray(indices, dtype=np.int64)
        self._y = y.detach().cpu().numpy().astype(np.int64)
        self._g = g.detach().cpu().numpy().astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        base_i = int(self.indices[i])
        x, _, _ = self.base_subset[base_i]
        return x, int(self._y[base_i]), int(self._g[base_i]), base_i


@dataclass(frozen=True)
class RealWorldSplits:
    train: Dataset
    cal: Dataset
    val: Dataset
    test: Dataset
    n_classes: int
    n_groups: int


def prepare_wilds_splits(
    wilds_name: str,
    root_dir: Path,
    cache_dir: Path,
    *,
    split_seed: int = 0,
    cal_fraction: float = 0.2,
    download: bool = True,
    transform=None,
) -> RealWorldSplits:
    ds, s = load_wilds_dataset(wilds_name, root_dir=root_dir, download=download)

    train_subset = ds.get_subset(s.train, transform=transform)
    y_train_full = train_subset.y_array
    if wilds_name.lower() == "civilcomments":
        g_train_full = _civilcomments_groups_identity_any(y_train_full, train_subset.metadata_array)
    else:
        g_train_full = _groups_from_metadata(ds, train_subset.metadata_array, y_train_full, wilds_name)

    cal_pct = int(round(100 * cal_fraction))
    idx_path = cache_dir / f"{wilds_name}__split{split_seed}__cal{cal_pct}__idx.npz"
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    if idx_path.exists():
        arr = np.load(idx_path)
        train_idx = arr["train_idx"].astype(np.int64)
        cal_idx = arr["cal_idx"].astype(np.int64)
    else:
        train_idx, cal_idx = make_train_cal_split(len(train_subset), cal_fraction=cal_fraction, split_seed=split_seed)
        np.savez(idx_path, train_idx=train_idx, cal_idx=cal_idx)

    train = IndexedWILDSSubset(train_subset, train_idx, y_train_full, g_train_full)
    cal = IndexedWILDSSubset(train_subset, cal_idx, y_train_full, g_train_full)

    val_subset = ds.get_subset(s.val, transform=transform)
    test_subset = ds.get_subset(s.test, transform=transform)

    y_val = val_subset.y_array
    y_test = test_subset.y_array
    if wilds_name.lower() == "civilcomments":
        g_val = _civilcomments_groups_identity_any(y_val, val_subset.metadata_array)
        g_test = _civilcomments_groups_identity_any(y_test, test_subset.metadata_array)
    else:
        g_val = _groups_from_metadata(ds, val_subset.metadata_array, y_val, wilds_name)
        g_test = _groups_from_metadata(ds, test_subset.metadata_array, y_test, wilds_name)


    val = IndexedWILDSSubset(val_subset, np.arange(len(val_subset), dtype=np.int64), y_val, g_val)
    test = IndexedWILDSSubset(test_subset, np.arange(len(test_subset), dtype=np.int64), y_test, g_test)

    y_all = np.concatenate(
        [
            y_train_full.detach().cpu().numpy().astype(np.int64),
            y_val.detach().cpu().numpy().astype(np.int64),
            y_test.detach().cpu().numpy().astype(np.int64),
        ]
    )
    g_all = np.concatenate(
        [
            g_train_full.detach().cpu().numpy().astype(np.int64),
            g_val.detach().cpu().numpy().astype(np.int64),
            g_test.detach().cpu().numpy().astype(np.int64),
        ]
    )

    return RealWorldSplits(
        train=train,
        cal=cal,
        val=val,
        test=test,
        n_classes=int(y_all.max()) + 1, #n_classes and n_groups are known constants
        n_groups=int(g_all.max()) + 1,
    )
