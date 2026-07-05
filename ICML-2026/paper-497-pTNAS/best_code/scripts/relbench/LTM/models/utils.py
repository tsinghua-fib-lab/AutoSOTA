"""
Utility functions for LTM package.

Shared helper functions used across different modules.
"""

import json
import pandas as pd
from pathlib import Path


def dataframe_to_texts(df: pd.DataFrame, feature_names_map=None, prefix=None):
    """
    Convert DataFrame rows to text strings.
    
    Args:
        df: Input DataFrame
        feature_names_map: Optional dict mapping original column names to standardized names
        prefix: Optional prefix to add to each text (e.g., "classification:")
    
    Returns:
        List of text strings, one per row
    """
    if feature_names_map:
        col_names = [feature_names_map.get(c, c) for c in df.columns]
    else:
        col_names = list(df.columns)

    texts = []
    for _, row in df.iterrows():
        items = [
            f"{k}: {v}"
            for k, v in zip(col_names, row.values)
            if pd.notna(v)
        ]
        text = ", ".join(items)
        if prefix:
            text = f"{prefix} {text}"
        texts.append(text)
    return texts


def generate_feature_names(df: pd.DataFrame, output_file: Path, has_label: bool = True):
    """
    Generate feature_names.json from DataFrame.
    
    Args:
        df: Input DataFrame
        output_file: Path to output JSON file
        has_label: Whether DataFrame has a label column (last column)
    """
    feature_name_dict = {}

    # Determine which columns are features (skip label if present)
    feature_cols = df.columns[:-1] if has_label else df.columns
    
    for col in feature_cols:
        temp = col
        # Handle underscores
        if '_' in temp:
            temp = ' '.join(temp.lower().split('_'))
        # Handle dots
        if '.' in temp:
            temp = ' '.join(temp.lower().split('.'))
        # Handle hyphens
        if '-' in temp:
            temp = ' '.join(temp.lower().split('-'))

        feature_name_dict[col] = temp.lower()

    with open(output_file, 'w') as f:
        json.dump(feature_name_dict, f, indent=4)

