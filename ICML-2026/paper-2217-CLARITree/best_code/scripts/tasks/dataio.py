from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import split_regression 

def load_xy_pystreed(csv_path: Path, target_pos: str = "last") -> Tuple[np.ndarray, np.ndarray]:
    """
    Loader for STreeD (pandas): returns (X, y) where y is last (or first) column.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")
    df = pd.read_csv(csv_path, sep=None, engine="python")
    if target_pos == "first":
        y = df.iloc[:, 0].to_numpy()
        X = df.iloc[:, 1:].to_numpy()
    else:
        y = df.iloc[:, -1].to_numpy()
        X = df.iloc[:, :-1].to_numpy()
    return X, y


def load_xy_split_regression(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loader for Cholickety (C++): uses split_regression .read_csv, which auto-adds an intercept to X.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")
    X, y = split_regression.read_csv(str(csv_path), has_header=True, delimiter=",")
    return np.array(X), np.array(y)
