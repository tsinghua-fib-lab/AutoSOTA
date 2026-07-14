import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _fs_num(x: float) -> str:
    """Filesystem-safe number: 0.2 -> 0p2, -1 -> m1."""
    s = format(float(x), ".8g")
    s = s.replace("-", "m").replace(".", "p")
    return re.sub(r"[^0-9mp]", "", s) or "0"


def dataset_group_dirname(
    encoded_dim: int,
    length_scale: float,
    domain_bounds: np.ndarray,
    bundle_type: str = "hexagonal",
) -> str:
    """
    Human-readable folder under ``data_root``:

    ``{bundle}_dim{dim}_ls{length_scale}_bounds_{axis0_lo_hi}__{axis1_lo_hi}...``
    """
    b = np.asarray(domain_bounds, dtype=float)
    axes = []
    for row in b:
        axes.append(f"{_fs_num(row[0])}_{_fs_num(row[1])}")
    dom = "__".join(axes)
    bt = str(bundle_type).lower()
    if bt in ("hexagonal", "hex"):
        bt = "hex"
    elif bt == "random":
        bt = "rnd"
    else:
        bt = re.sub(r"[^a-z0-9]+", "_", bt).strip("_")[:24] or "ssp"
    ls = _fs_num(length_scale)
    return f"{bt}_dim{int(encoded_dim)}_ls{ls}_bounds_{dom}"


@dataclass
class DatasetSpec:
    data_root: Path
    dataset_type: str
    encoded_dim: int
    length_scale: float
    train_samples: int
    test_samples: int
    sampling_method: str = "sobol"
    train_subdir: str = "train"
    test_subdir: str = "test"


def _stable_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _config_payload(spec: DatasetSpec, ssp_cfg: dict, ssp_space) -> dict:
    payload = {
        "dataset_type": spec.dataset_type,
        "encoded_dim": spec.encoded_dim,
        "length_scale": spec.length_scale,
        "train_samples": spec.train_samples,
        "test_samples": spec.test_samples,
        "sampling_method": spec.sampling_method,
        "ssp_class": type(ssp_space).__name__,
        "ssp_config": {
            "n_rotates": ssp_cfg.get("n_rotates"),
            "n_scales": ssp_cfg.get("n_scales"),
            "domain_bounds": np.asarray(getattr(ssp_space, "domain_bounds", [])).tolist(),
        },
    }
    return payload


def _count_npy(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.glob("*.npy"))


def _save_targets(ssp_space, out_dir: Path, total: int, sampling_method: str, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ssps, _ = ssp_space.get_sample_pts_and_ssps(total, method=sampling_method)
    for i in range(total):
        np.save(out_dir / f"{prefix}_{i:06d}.npy", ssps[i])


def _read_meta(meta_path: Path):
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _dataset_ready(dataset_dir: Path, spec: DatasetSpec, meta_path: Path) -> bool:
    if not meta_path.exists():
        return False
    train_dir = dataset_dir / spec.train_subdir
    test_dir = dataset_dir / spec.test_subdir
    return (
        _count_npy(train_dir) >= spec.train_samples
        and _count_npy(test_dir) >= spec.test_samples
    )


def ensure_target_dataset(spec: DatasetSpec, ssp_cfg: dict, ssp_space) -> dict:
    """
    Layout (new runs):

    * ``{data_root}/{group_dir}/train/`` — training target ``.npy`` files
    * ``{data_root}/{group_dir}/test/`` — held-out targets
    * ``{data_root}/{group_dir}/A_matrix.npy`` — same axis matrix used when
      encoding those targets (train and test share one space / one ``A``).

    ``group_dir`` encodes bundle type, encoded dimension, length scale, and
    domain bounds (see :func:`dataset_group_dirname`). A full config hash
    ``dataset_id`` is stored in ``dataset_meta.json``; if that folder already
    exists with a different id, a :class:`ValueError` is raised so two
    incompatible configs do not share one directory.

    Reuse order (backward compatibility):

    1. ``{data_root}/{group_dir}/`` (flat)
    2. ``{data_root}/{group_dir}/dataset_{hash}/`` (older nested layout)
    3. ``{data_root}/dataset_{hash}/`` (legacy, no group prefix)

    Training and evaluation both read targets from disk. :class:`SSPDataset`
    always draws **z0** as pure ``noise_type`` noise; it matches the trainer's
    ``noise_type`` / ``target_type`` when eval passes the same settings.
    Signal-strength sweeps in eval blend noise and target only when forming the
    model's initial state (see ``utils.evaluation``), not inside the dataset.
    """
    spec.data_root.mkdir(parents=True, exist_ok=True)
    payload = _config_payload(spec, ssp_cfg, ssp_space)
    dataset_id = _stable_hash(payload)

    bounds = np.asarray(getattr(ssp_space, "domain_bounds", []), dtype=float)
    group_dir = dataset_group_dirname(
        spec.encoded_dim,
        float(spec.length_scale),
        bounds,
        bundle_type=str(ssp_cfg.get("bundle_type", "hexagonal")),
    )

    flat_dir = spec.data_root / group_dir
    nested_dir = spec.data_root / group_dir / f"dataset_{dataset_id}"
    legacy_dir = spec.data_root / f"dataset_{dataset_id}"

    for dataset_dir, meta_path in (
        (flat_dir, flat_dir / "dataset_meta.json"),
        (nested_dir, nested_dir / "dataset_meta.json"),
        (legacy_dir, legacy_dir / "dataset_meta.json"),
    ):
        meta = _read_meta(meta_path)
        if meta is None or meta.get("dataset_id") != dataset_id:
            continue
        if not _dataset_ready(dataset_dir, spec, meta_path):
            continue
        train_dir = dataset_dir / spec.train_subdir
        test_dir = dataset_dir / spec.test_subdir
        return {
            "dataset_id": dataset_id,
            "dataset_group": group_dir,
            "dataset_dir": str(dataset_dir),
            "train_dir": str(train_dir),
            "test_dir": str(test_dir),
            "created": False,
        }

    dataset_dir = flat_dir
    meta_path = dataset_dir / "dataset_meta.json"
    amat_path = dataset_dir / "A_matrix.npy"
    train_dir = dataset_dir / spec.train_subdir
    test_dir = dataset_dir / spec.test_subdir

    existing = _read_meta(meta_path)
    if existing is not None and existing.get("dataset_id") != dataset_id:
        raise ValueError(
            f"Data folder {dataset_dir} already exists with dataset_id "
            f"{existing.get('dataset_id')!r}, but the current config hashes to "
            f"{dataset_id!r} (different sample counts, sampling method, or SSP settings). "
            "Remove that folder or align config with the existing dataset."
        )

    if dataset_dir.exists() and not meta_path.exists():
        if any(dataset_dir.rglob("*.npy")):
            raise ValueError(
                f"{dataset_dir} contains .npy files but no dataset_meta.json. "
                "Remove stray files or restore metadata."
            )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    _save_targets(ssp_space, train_dir, spec.train_samples, spec.sampling_method, prefix="target")
    _save_targets(ssp_space, test_dir, spec.test_samples, spec.sampling_method, prefix="target")

    axis_matrix = np.asarray(getattr(ssp_space, "axis_matrix", np.array([])))
    np.save(amat_path, axis_matrix)

    meta = {
        "dataset_id": dataset_id,
        "dataset_group": group_dir,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": payload,
        "paths": {
            "train_dir": str(train_dir),
            "test_dir": str(test_dir),
            "A_matrix": str(amat_path),
        },
        "counts": {
            "train_files": _count_npy(train_dir),
            "test_files": _count_npy(test_dir),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "dataset_id": dataset_id,
        "dataset_group": group_dir,
        "dataset_dir": str(dataset_dir),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "created": True,
    }
