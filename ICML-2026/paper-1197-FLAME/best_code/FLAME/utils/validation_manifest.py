"""Utilities for fixed validation sample manifests.

The training configs often use ``sample_ratio`` for validation subsets.  That is
convenient but fragile because the chosen subset depends on Python's global RNG
state and on dataset-construction order.  A validation manifest stores the exact
sample identities so all reload/evaluation/training runs compare on the same
samples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


_ID_KEYS = (
    "type",
    "base_name",
    "target_file",
    "target_path",
    "mask_path",
    "source_path",
    "auth_file",
    "source_dir",
    "is_nested",
)


def sample_identity(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable JSON-safe identity for a LocalForgeryDataset sample."""
    identity: dict[str, Any] = {}
    for key in _ID_KEYS:
        if key in sample:
            value = sample[key]
            if isinstance(value, Path):
                value = str(value)
            identity[key] = value
    if "type" not in identity:
        identity["type"] = "forgery"
    if "base_name" not in identity:
        raise KeyError(f"sample has no base_name: {sample}")
    return identity


def _identity_key(identity: Mapping[str, Any]) -> str:
    normalized = sample_identity(identity)
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def apply_manifest_to_dataset(
    dataset: Any,
    manifest_entries: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str,
) -> Any:
    """Filter and reorder ``dataset.all_samples`` according to manifest entries.

    The function mutates and returns ``dataset`` because ``LocalForgeryDataset``
    stores sampling state in ``all_samples``.  Missing manifest entries are hard
    errors: a silent fallback would invalidate the fixed-evaluation contract.
    """
    available: dict[str, Mapping[str, Any]] = {}
    for sample in getattr(dataset, "all_samples"):
        available[_identity_key(sample)] = sample

    selected = []
    missing = []
    for entry in manifest_entries:
        key = _identity_key(entry)
        sample = available.get(key)
        if sample is None:
            missing.append(sample_identity(entry))
        else:
            selected.append(sample)

    if missing:
        preview = missing[:3]
        raise KeyError(
            f"Validation manifest for {dataset_name!r} references {len(missing)} missing samples; "
            f"first missing entries: {preview}"
        )

    dataset.all_samples = list(selected)
    return dataset


def make_manifest_dataset_entry(dataset_name: str, dataset: Any) -> dict[str, Any]:
    """Create one dataset entry for a validation manifest."""
    return {
        "name": dataset_name,
        "root": str(getattr(dataset, "root", "")),
        "num_samples": len(getattr(dataset, "all_samples", [])),
        "samples": [sample_identity(sample) for sample in getattr(dataset, "all_samples", [])],
    }


def manifest_entries_by_name(manifest: Mapping[str, Any]) -> dict[str, Sequence[Mapping[str, Any]]]:
    """Normalize supported manifest formats to ``{dataset_name: sample_entries}``."""
    datasets = manifest.get("datasets", {})
    if isinstance(datasets, Mapping):
        return {str(name): list(entries) for name, entries in datasets.items()}
    out: dict[str, Sequence[Mapping[str, Any]]] = {}
    for entry in datasets:
        out[str(entry["name"])] = list(entry.get("samples", []))
    return out


def load_validation_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_validation_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
