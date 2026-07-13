from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from basicts.utils.data_utils import parse_and_filter_indices
from tsl.data import SpatioTemporalDataset


def _normalize_index_values(index: pd.Index, values: np.ndarray) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        return pd.to_datetime(values).to_numpy()
    return np.asarray(values)


def _parse_residuals_df(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, object]]:
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        nodes = []
        horizons = []
        for node, feat in cols:
            if node not in nodes:
                nodes.append(node)
            try:
                h = int(str(feat).split("_")[-1])
            except Exception:
                h = 0
            if h not in horizons:
                horizons.append(h)
        horizons = sorted(horizons)
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        h_to_idx = {h: i for i, h in enumerate(horizons)}
        arr = np.zeros((len(df), len(horizons), len(nodes)), dtype=np.float32)
        for (node, feat) in cols:
            try:
                h = int(str(feat).split("_")[-1])
            except Exception:
                h = 0
            arr[:, h_to_idx[h], node_to_idx[node]] = df[(node, feat)].to_numpy()
        meta = {"nodes": nodes, "horizons": horizons}
        return arr, meta
    values = df.to_numpy()
    if values.ndim == 1:
        values = values.reshape(-1, 1, 1)
    elif values.ndim == 2:
        values = values.reshape(values.shape[0], 1, values.shape[1])
    meta = {"nodes": list(range(values.shape[2])), "horizons": [0]}
    return values.astype(np.float32), meta


def _get_dataset_class(name: str):
    import importlib
    ds = importlib.import_module("tsl.datasets")
    candidates = {
        "la": ["MetrLA"],
        "pems03": ["PeMS03", "Pems03", "PEMS03"],
        "pems04": ["PeMS04", "Pems04", "PEMS04"],
        "pems07": ["PeMS07", "Pems07", "PEMS07"],
        "pems08": ["PeMS08", "Pems08", "PEMS08"],
        "pems_bay": ["PemsBay", "PeMSBay", "PEMSBay"],
        "large_st": ["LargeST"],
    }
    for cls_name in candidates.get(name, []):
        if hasattr(ds, cls_name):
            return getattr(ds, cls_name)
    raise ImportError(f"Dataset class for '{name}' not found in tsl.datasets.")


def _make_dataset(ds_cls, *args, **kwargs):
    try:
        from tsl import config as tsl_config
        root = Path(__file__).resolve().parents[1]
        tsl_config.data_dir = str(root / "datasets")
    except Exception:
        pass
    return ds_cls(*args, **kwargs)


def _get_dataset(dataset_cfg):
    name = dataset_cfg["name"]
    if name in {"pems03", "pems04", "pems07", "pems08", "pems_bay"}:
        return _make_dataset(_get_dataset_class(name))
    if name == "large_st":
        hparams = dict(dataset_cfg.get("hparams", {}))
        root = hparams.pop("root", None) or str(Path(__file__).resolve().parents[1] / "datasets" / "large_st")
        return _make_dataset(_get_dataset_class(name), root=root, **hparams)
    if name == "la":
        return _make_dataset(_get_dataset_class("la"))
    if name == "air":
        from basicts.data.air_quality import AirQuality
        return AirQuality()
    if name == "gpvar":
        from basicts.data.gpvar import GPVARDataset
        return GPVARDataset(**dataset_cfg["hparams"], p_max=0)
    raise ValueError(f"Dataset {name} not available.")


def _build_index_dataset(dataset, src_config):
    return SpatioTemporalDataset(
        index=dataset.index,
        target=dataset.dataframe(),
        mask=dataset.mask,
        window=src_config["window"],
        horizon=src_config["horizon"],
        stride=src_config["stride"],
        delay=src_config.get("delay", 0),
    )


def _resolve_time_indices(
    src_dir: Path, df_index: pd.Index, indices: np.lib.npyio.NpzFile
) -> Tuple[np.ndarray, np.ndarray]:
    calib_time = indices.get("calib_time_index")
    test_time = indices.get("test_time_index")
    if calib_time is not None and test_time is not None:
        return (
            _normalize_index_values(df_index, calib_time),
            _normalize_index_values(df_index, test_time),
        )

    calib_indices = indices.get("calib_indices")
    test_indices = indices.get("test_indices")
    if calib_indices is None or test_indices is None:
        raise ValueError("indices.npz must contain calib/test indices or time indices.")

    cfg_path = src_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError("config.yaml missing; cannot map indices to timestamps.")
    with open(cfg_path, "r") as fp:
        src_config = yaml.load(fp, Loader=yaml.FullLoader)
    dataset = _get_dataset(src_config["dataset"])
    target_dataset = _build_index_dataset(dataset, src_config)
    calib_indices, test_indices = parse_and_filter_indices(target_dataset, indices)

    calib_ts = target_dataset.data_timestamps(calib_indices)["horizon"][:, 0]
    test_ts = target_dataset.data_timestamps(test_indices)["horizon"][:, 0]
    return calib_ts, test_ts


def load_residuals_split(src_dir: str | Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    src_dir = Path(src_dir)
    h5_path = src_dir / "residuals.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"residuals.h5 not found in {src_dir}")
    idx_path = src_dir / "indices.npz"
    if not idx_path.exists():
        raise FileNotFoundError(f"indices.npz not found in {src_dir}")

    target_df: pd.DataFrame = pd.read_hdf(h5_path, key="target")
    indices = np.load(idx_path)
    calib_time, test_time = _resolve_time_indices(src_dir, target_df.index, indices)
    cal_df = target_df.loc[calib_time]
    test_df = target_df.loc[test_time]

    cal_arr, meta = _parse_residuals_df(cal_df)
    test_arr, _ = _parse_residuals_df(test_df)
    return cal_arr, test_arr, meta


def load_residuals_input_split(src_dir: str | Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    src_dir = Path(src_dir)
    h5_path = src_dir / "residuals.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"residuals.h5 not found in {src_dir}")
    idx_path = src_dir / "indices.npz"
    if not idx_path.exists():
        raise FileNotFoundError(f"indices.npz not found in {src_dir}")

    input_df: pd.DataFrame = pd.read_hdf(h5_path, key="input")
    indices = np.load(idx_path)
    calib_time, test_time = _resolve_time_indices(src_dir, input_df.index, indices)
    cal_df = input_df.loc[calib_time]
    test_df = input_df.loc[test_time]
    cal_arr, meta = _parse_residuals_df(cal_df)
    test_arr, _ = _parse_residuals_df(test_df)
    return cal_arr, test_arr, meta


__all__ = ["load_residuals_split", "load_residuals_input_split"]
