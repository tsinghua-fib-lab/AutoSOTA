from typing import Dict, Iterable, List, Optional, Any, Tuple, Union
from dataclasses import dataclass

import pandas as pd


@dataclass (frozen=True, order=True)
class TimeVar:
    name: str
    time: Union[int, float]

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError(f"name must be str, got {type(self.name).__name__}")

        if not isinstance(self.time, int) and not isinstance(self.time, float):
            raise TypeError(f"time must be int or float, got {type(self.time).__name__}")

    def get_name(self):
        return self.name
    
    def get_time(self):
        return self.time


@dataclass (frozen=True, order=True)
class DTimeVar(TimeVar):
    time: int

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.time, int):
            raise TypeError(f"time must be int, got {type(self.time).__name__}")


@dataclass(frozen=True)
class CTimeVar(TimeVar):
    time: float

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.time, float):
            raise TypeError(f"time must be float, got {type(self.time).__name__}")


def time_var_to_str(time_var: TimeVar) -> str:
    return f"{time_var.name}_{time_var.time}"


# def wide_row_to_df_ts(df):
#     row = df.iloc[0]
#     records = []
#
#     for col, val in row.items():
#         records.append({"time": col.time, "variable": col.name, "value": val})
#
#     long_df = pd.DataFrame(records)
#     ts_df = long_df.pivot(index="time", columns="variable", values="value").sort_index()
#     return ts_df


def wide_timevar_to_ts_df(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a one-row wide DataFrame with DTimeVar columns into a time-indexed
    trajectory DataFrame with string variable columns.

    Example input columns:
        DTimeVar("X", 1), DTimeVar("X", 2), DTimeVar("Y", 1), ...

    Output:
            X    Y
        1  ...
        2  ...
    """
    print(len(df_wide.columns))
    print(len(df_wide))
    if len(df_wide) != 1:
        raise ValueError("Expected a one-row DataFrame representing one trajectory.")

    row = df_wide.iloc[0]

    records = []
    for col, val in row.items():
        if not isinstance(col, DTimeVar):
            raise TypeError("All columns must be DTimeVar objects.")
        records.append({"time": col.time, "variable": col.name, "value": val})

    df_long = pd.DataFrame(records)
    df_ts = df_long.pivot(index="time", columns="variable", values="value").sort_index()
    return df_ts


def ts_to_lagged_df(
    df_ts: pd.DataFrame,
    window_size: int,
    targets: Optional[Iterable[str]] = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Convert a time-indexed trajectory DataFrame into a lagged tabular DataFrame
    using DTimeVar columns.

    Parameters
    ----------
    df_ts : pd.DataFrame
        DataFrame indexed by time, with columns equal to variable names (strings),
        e.g. columns ['X', 'Y'].
    max_lag : int
        Maximum lag to include.
    targets : iterable of str, optional
        Variables to include at lag 0 as targets. If None, all columns are used.
    dropna : bool
        Whether to drop rows with missing values induced by shifting.

    Returns
    -------
    pd.DataFrame
        Lagged dataframe with DTimeVar columns:
        - DTimeVar(var, -k) for lagged predictors
        - DTimeVar(var, 0) for targets
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    if targets is None:
        targets = list(df_ts.columns)
    else:
        targets = list(targets)

    out = {}

    # lagged predictors
    for col in df_ts.columns:
        for lag in range(1, window_size):
            out[DTimeVar(col, -lag)] = df_ts[col].shift(lag)

    # current-time targets
    for col in targets:
        out[DTimeVar(col, 0)] = df_ts[col]

    df_lagged = pd.DataFrame(out, index=df_ts.index)

    if dropna:
        df_lagged = df_lagged.dropna()

    return df_lagged


def wide_timevar_to_lagged_df(df_wide: pd.DataFrame, window_size, dropna: bool = True,) -> pd.DataFrame:
    """
    Convert a one-row wide DataFrame with DTimeVar columns into a lagged tabular
    DataFrame with DTimeVar columns.

    Example input columns:
        DTimeVar("X", 1), DTimeVar("X", 2), DTimeVar("Y", 1), ...

    Output:
        DTimeVar("X", -1)  DTimeVar("X", -2)  DTimeVar("Y", -1)  DTimeVar("Y", -2)  DTimeVar("X", 0)  DTimeVar("Y", 0)
        ...
    """
    df_ts = wide_timevar_to_ts_df(df_wide)
    df_lagged = ts_to_lagged_df(df_ts, window_size=window_size, dropna=dropna)
    return df_lagged
