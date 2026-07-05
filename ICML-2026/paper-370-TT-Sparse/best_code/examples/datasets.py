"""Dataset registry for examples. Fetches a DataFrame with a 'target' column on demand.

Sources:
- 'openml': sklearn.datasets.fetch_openml(data_id=...)
- 'sklearn_wine', 'sklearn_california': built-in sklearn loaders
- 'pmlb': pmlb.fetch_data(name)              [pip install pmlb]
- 'uci':  ucimlrepo.fetch_ucirepo(id=...)    [pip install ucimlrepo]
"""

from __future__ import annotations

import pandas as pd

DATASETS: dict[str, dict] = {
    # Binary
    "compas":              {"task": "binary",     "src": "openml", "id": 45039},
    "covertype":           {"task": "binary",     "src": "openml", "id": 44159},
    "credit_card_default": {"task": "binary",     "src": "openml", "id": 45036},
    "electricity":         {"task": "binary",     "src": "openml", "id": 44156},
    "eye_movements":       {"task": "binary",     "src": "openml", "id": 44157},
    "road_safety":         {"task": "binary",     "src": "openml", "id": 44161},
    "creditg":             {"task": "binary",     "src": "openml", "id": 31},
    "diabetes":            {"task": "binary",     "src": "openml", "id": 37},
    "blood":               {"task": "binary",     "src": "openml", "id": 1464},
    "income":              {"task": "binary",     "src": "openml", "id": 1590},
    "bank":                {"task": "binary",     "src": "openml", "id": 1461},
    "heart":               {"task": "binary",     "src": "openml", "id": 43672},
    # Multiclass
    "satimage": {"task": "multiclass", "src": "pmlb", "name": "satimage"},
    "penguins": {"task": "multiclass", "src": "pmlb", "name": "penguins"},
    "iris":     {"task": "multiclass", "src": "pmlb", "name": "iris"},
    "car":      {"task": "multiclass", "src": "pmlb", "name": "car_evaluation"},
    "ecoli":    {"task": "multiclass", "src": "pmlb", "name": "ecoli"},
    "yeast":    {"task": "multiclass", "src": "pmlb", "name": "yeast"},
    "wine":     {"task": "multiclass", "src": "sklearn_wine"},
    "jungle":   {"task": "multiclass", "src": "openml", "id": 41027},
    # Regression
    "wine_reg":   {"task": "regression", "src": "uci", "id": 186},
    "abalone":    {"task": "regression", "src": "uci", "id": 1},
    "california": {"task": "regression", "src": "sklearn_california"},
    "bike":       {"task": "regression", "src": "uci", "id": 275},
    "crime":      {"task": "regression", "src": "uci", "id": 183},
    "parkinsons": {"task": "regression", "src": "uci", "id": 189},
}


def list_datasets(task: str | None = None) -> list[str]:
    """Return all dataset names, optionally filtered by task."""
    if task is None:
        return list(DATASETS)
    return [k for k, v in DATASETS.items() if v["task"] == task]


def load_dataset(name: str) -> tuple[pd.DataFrame, str]:
    """Load a dataset by name. Returns (df with 'target' column, task_type)."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(DATASETS)}")
    cfg = DATASETS[name]
    src = cfg["src"]

    if src == "openml":
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(data_id=cfg["id"], as_frame=True)
        df = ds.frame.copy()
        target_col = ds.target_names[0] if getattr(ds, "target_names", None) else df.columns[-1]
        if target_col != "target":
            df = df.rename(columns={target_col: "target"})

    elif src == "sklearn_wine":
        from sklearn.datasets import load_wine
        ds = load_wine(as_frame=True)
        df = pd.concat([ds.data, ds.target.rename("target")], axis=1)

    elif src == "sklearn_california":
        from sklearn.datasets import fetch_california_housing
        ds = fetch_california_housing(as_frame=True)
        df = pd.concat([ds.data, ds.target.rename("target")], axis=1)

    elif src == "pmlb":
        from pmlb import fetch_data
        df = fetch_data(cfg["name"])
        if "target" not in df.columns:
            df = df.rename(columns={df.columns[-1]: "target"})

    elif src == "uci":
        from ucimlrepo import fetch_ucirepo
        repo = fetch_ucirepo(id=cfg["id"])
        df = pd.concat([repo.data.features, repo.data.targets], axis=1)
        df = df.rename(columns={df.columns[-1]: "target"})
        if name == "bike" and "dteday" in df.columns:
            df = df.drop(columns=["dteday"])

    else:
        raise ValueError(f"Unknown source {src!r} for dataset {name!r}")

    return df, cfg["task"]
