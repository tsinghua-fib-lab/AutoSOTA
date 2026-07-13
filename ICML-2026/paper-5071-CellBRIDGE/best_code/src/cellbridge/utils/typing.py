from typing import Any

import anndata as ad
import pandas as pd
from pandas import Index


def as_index(obj: Any) -> Index:
    """Convert an object to a pandas Index."""
    return obj if isinstance(obj, pd.Index) else pd.Index(obj)


def ensure_upper_var(adata: ad.AnnData, var_col: str = "UP") -> pd.Index:
    """
    Ensure an uppercase symbol column exists; returns that index.
    Uses var_names by default.
    """
    if var_col not in adata.var:
        adata.var[var_col] = adata.var_names.str.upper().astype(str)
    return as_index(adata.var[var_col])
