import numpy as np
import pandas as pd


def analyze_column(input_series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(input_series):
        return 'numeric'
    else:
        return 'categorical'
