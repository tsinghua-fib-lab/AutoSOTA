"""
Ground Truth Query API for NAS-Bench-Tabular

Ported and adapted from VLDB_code/TRAILS query API.
Provides access to pre-computed training results and proxy scores
for 160K MLP architectures on Frappe, UCI Diabetes, and Criteo.

Data is loaded from the local NAS-Bench-Tabular release package.
"""

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

PTPROXY_KEY = "pTProxy"
LEGACY_PTPROXY_KEY = "express_flow"

# ============================================================
# Hardcoded data paths
# ============================================================

_PTNAS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPACE_MLP = os.path.join(_PTNAS_ROOT, "datasets", "nas_bench_tabular", "space_mlp")
_TRAIN_BASE = os.path.join(_SPACE_MLP, "training")
_PROXY_BASE = os.path.join(_SPACE_MLP, "proxy_score")

# Training data (ground truth: train/valid AUC per epoch)
_TRAIN_PATHS = {
    "frappe": os.path.join(_TRAIN_BASE, "all_train_baseline_frappe.csv"),
    "uci_diabetes": os.path.join(_TRAIN_BASE, "all_train_baseline_uci_160k_40epoch.csv"),
    "criteo": os.path.join(_TRAIN_BASE, "all_train_baseline_criteo.csv"),
}

# Baseline proxy scores (Frappe 用 120K 版本，和论文 Table 1 一致)
_SCORE_PATHS = {
    "frappe": os.path.join(_PROXY_BASE, "baseline", "score_frappe_batch_size_32_120kmodels_all_metrics.csv"),
    "uci_diabetes": os.path.join(_PROXY_BASE, "baseline", "score_uci_diabetes_batch_size_32_all_metrics.csv"),
    "criteo": os.path.join(_PROXY_BASE, "baseline", "score_criteo_batch_size_32.csv"),
}

# pTProxy scores (computed separately, full benchmark pool)
_PTPROXY_PATHS = {
    "frappe": os.path.join(_PROXY_BASE, "ptproxy", "score_mlp_sp_frappe_batch_size_32_cpu.csv"),
    "uci_diabetes": os.path.join(_PROXY_BASE, "ptproxy", "score_mlp_sp_uci_diabetes_batch_size_32_cpu.csv"),
    "criteo": os.path.join(_PROXY_BASE, "ptproxy", "score_mlp_sp_criteo_batch_size_32_cpu.csv"),
}

# Max training epochs available per dataset
_MAX_EPOCHS = {
    "frappe": 19,       # epochs 0-19
    "uci_diabetes": 39, # epochs 0-39
    "criteo": 9,        # epochs 0-9
}

# Ground truth epoch for baseline proxies (Table 1 SRCC)
_BEST_EPOCH = {
    "frappe": 19,
    "uci_diabetes": 0,
    "criteo": 9,
}

# Ground truth epoch for pTProxy scores
_BEST_EPOCH_PTPROXY = {
    "frappe": 13,
    "uci_diabetes": 0,
    "criteo": 9,
}

# Backward-compatible alias for older scripts.
_BEST_EPOCH_EXPRESSFLOW = _BEST_EPOCH_PTPROXY

# Pre-computed timing (seconds)
SCORE_TIME_PER_MODEL = {
    "cpu": {"frappe": 0.0212, "uci_diabetes": 0.0150, "criteo": 0.6824},
    "gpu": {"frappe": 0.0137, "uci_diabetes": 0.0082, "criteo": 0.6095},
}

TRAIN_TIME_PER_EPOCH = {
    "cpu": {"frappe": 5.12, "uci_diabetes": 4.16, "criteo": 422.0},
    "gpu": {"frappe": 2.8, "uci_diabetes": 1.4, "criteo": 125.0},
}


# ============================================================
# File loaders
# ============================================================

def _read_json(path: str) -> dict:
    """Load JSON file, return empty dict if file doesn't exist."""
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return {}
    print(f"Loading {path}...")
    with open(path, 'r') as f:
        return json.load(f)


def _float_or_nan(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _read_training_csv(path: str, dataset: str) -> dict:
    """Load training ground truth CSV into the historical nested API shape."""
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return {dataset: {}}
    print(f"Loading {path}...")
    result = {dataset: {}}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            arch_id = row["arch_id"]
            epoch = str(int(row["epoch"]))
            result[dataset].setdefault(arch_id, {})[epoch] = {
                "train_auc": _float_or_nan(row.get("train_auc")),
                "valid_auc": _float_or_nan(row.get("valid_auc")),
                "train_loss": _float_or_nan(row.get("train_loss")),
                "valid_loss": _float_or_nan(row.get("valid_loss")),
                "train_val_total_time": _float_or_nan(row.get("train_val_total_time")),
            }
    return result


def _read_score_csv(path: str) -> dict:
    """Load proxy score CSV as {arch_id: {proxy_name: score}}."""
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return {}
    print(f"Loading {path}...")
    result = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        proxy_names = [name for name in (reader.fieldnames or []) if name != "arch_id"]
        for row in reader:
            arch_id = row["arch_id"]
            result[arch_id] = {
                name: _float_or_nan(row.get(name))
                for name in proxy_names
                if row.get(name) not in (None, "")
            }
    return result


def _read_training_file(path: str, dataset: str) -> dict:
    if path.endswith(".csv"):
        return _read_training_csv(path, dataset)
    return _read_json(path)


def _read_score_file(path: str) -> dict:
    if path.endswith(".csv"):
        return _read_score_csv(path)
    return _read_json(path)


# ============================================================
# GTMLP: Ground Truth MLP Query API (Singleton)
# ============================================================

class GTMLP:
    """
    Singleton query API for NAS-Bench-Tabular ground truth.

    Usage:
        gt = GTMLP("frappe")
        valid_auc, time = gt.get_valid_auc("8-16-32-64", epoch_num=13)
        scores = gt.api_get_score("8-16-32-64")
    """

    _instances: Dict[str, "GTMLP"] = {}

    def __new__(cls, dataset: str):
        if dataset not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[dataset] = instance
        return cls._instances[dataset]

    def __init__(self, dataset: str):
        if self._initialized:
            return
        self._initialized = True

        if dataset not in _TRAIN_PATHS:
            raise ValueError(f"Unknown dataset: {dataset}. "
                             f"Choose from {list(_TRAIN_PATHS.keys())}")

        self.dataset = dataset

        # Load training ground truth
        self.mlp_train = _read_training_file(_TRAIN_PATHS[dataset], dataset)

        # Load zero-cost proxy scores
        self.mlp_score = _read_score_file(_SCORE_PATHS[dataset])

        # Merge pTProxy scores into mlp_score.
        ptproxy_data = _read_score_file(_PTPROXY_PATHS[dataset])
        for arch_id, scores in ptproxy_data.items():
            if LEGACY_PTPROXY_KEY in scores and PTPROXY_KEY not in scores:
                scores[PTPROXY_KEY] = scores.pop(LEGACY_PTPROXY_KEY)
            if arch_id in self.mlp_score:
                self.mlp_score[arch_id].update(scores)
            else:
                self.mlp_score[arch_id] = scores

        print(f"[GTMLP] {dataset}: "
              f"{len(self.mlp_train.get(dataset, {}))} trained models, "
              f"{len(self.mlp_score)} scored models")

    # ----------------------------------------------------------
    # Training ground truth queries
    # ----------------------------------------------------------

    def get_valid_auc(self, arch_id: str, epoch_num: Optional[int] = None) -> Tuple[float, float]:
        """
        Get validation AUC and training time for an architecture.

        Args:
            arch_id: e.g. "8-16-32-64"
            epoch_num: training epoch (0-indexed). If None, uses best epoch.

        Returns:
            (valid_auc, train_val_total_time)
        """
        best_epoch = _BEST_EPOCH[self.dataset]
        max_epoch = _MAX_EPOCHS[self.dataset]

        if epoch_num is None or epoch_num > max_epoch:
            epoch_num = best_epoch

        data = self.mlp_train[self.dataset][arch_id]
        epoch_data = data[str(epoch_num)]

        valid_auc = epoch_data["valid_auc"]
        time_usage = epoch_data["train_val_total_time"]
        return valid_auc, time_usage

    def get_best_valid_auc(self, arch_id: str) -> float:
        """Get best validation AUC across all epochs."""
        data = self.mlp_train[self.dataset][arch_id]
        best_auc = max(
            float(data[str(e)]["valid_auc"])
            for e in range(min(_MAX_EPOCHS[self.dataset] + 1, len(data)))
            if str(e) in data
        )
        return best_auc

    def get_all_trained_model_ids(self) -> List[str]:
        """Get all architecture IDs with training data."""
        return list(self.mlp_train[self.dataset].keys())

    # ----------------------------------------------------------
    # Proxy score queries
    # ----------------------------------------------------------

    def api_get_score(self, arch_id: str) -> Dict[str, float]:
        """Get all proxy scores for an architecture."""
        return self.mlp_score.get(arch_id, {})

    def get_proxy_score(self, arch_id: str, proxy_name: str) -> Optional[float]:
        """Get a specific proxy score for an architecture."""
        scores = self.mlp_score.get(arch_id, {})
        if proxy_name == PTPROXY_KEY:
            return scores.get(PTPROXY_KEY, scores.get(LEGACY_PTPROXY_KEY))
        return scores.get(proxy_name)

    def get_all_scored_model_ids(self) -> List[str]:
        """Get all architecture IDs with proxy scores."""
        return list(self.mlp_score.keys())

    # ----------------------------------------------------------
    # Timing queries
    # ----------------------------------------------------------

    def get_score_one_model_time(self, device: str = "cpu") -> float:
        return SCORE_TIME_PER_MODEL[device][self.dataset]

    def get_train_one_epoch_time(self, device: str = "gpu") -> float:
        return TRAIN_TIME_PER_EPOCH[device][self.dataset]

    # ----------------------------------------------------------
    # Convenience: get ground truth ranking
    # ----------------------------------------------------------

    def get_all_ground_truth_aucs(self, epoch_num: Optional[int] = None) -> Dict[str, float]:
        """
        Get {arch_id: valid_auc} for all trained architectures.
        Useful for computing SRCC against proxy scores.
        """
        result = {}
        for arch_id in self.mlp_train[self.dataset]:
            try:
                auc, _ = self.get_valid_auc(arch_id, epoch_num)
                result[arch_id] = auc
            except (KeyError, TypeError):
                continue
        return result
