"""
Data loading utilities for the Disagreement-Aware Evaluation Pipeline.

This module provides functions to load and validate annotation data from CSV files.
Expected CSV format: item_id, agent_id, annotator_id, label
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple
import glob


def load_single_csv(filepath: Union[str, Path], convert_to_response: bool = True) -> pd.DataFrame:
    """
    Load a single CSV file with annotation data.
    
    Args:
        filepath: Path to the CSV file
        convert_to_response: If True, convert to response-level format where
                            item_id becomes (original_item_id + agent_id)
        
    Returns:
        DataFrame with columns: item_id, agent_id, annotator_id, label
        If convert_to_response=True, also includes original_item_id column
        
    Raises:
        ValueError: If required columns are missing
    """
    df = pd.read_csv(filepath)
    required_columns = {'item_id', 'agent_id', 'annotator_id', 'label'}
    
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Ensure consistent types
    df['item_id'] = df['item_id'].astype(str)
    df['agent_id'] = df['agent_id'].astype(str)
    df['annotator_id'] = df['annotator_id'].astype(str)
    df['label'] = df['label'].astype(int)
    
    df = df[['item_id', 'agent_id', 'annotator_id', 'label']]
    
    # Convert to response-level format
    if convert_to_response:
        df = convert_to_response_level(df)
    
    return df


def load_all_data(data_dir: Union[str, Path], convert_to_response: bool = True) -> pd.DataFrame:
    """
    Load all CSV files from a directory and concatenate them.
    
    Args:
        data_dir: Path to directory containing CSV files
        convert_to_response: If True, convert to response-level format
        
    Returns:
        Combined DataFrame with all annotation data
    """
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    dfs = []
    for filepath in csv_files:
        try:
            df = load_single_csv(filepath, convert_to_response=False)  # Convert after combining
            df['source_file'] = filepath.name
            dfs.append(df)
            print(f"Loaded {len(df)} annotations from {filepath.name}")
        except Exception as e:
            print(f"Warning: Could not load {filepath.name}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(combined)} annotations loaded")
    
    # Convert to response-level format
    if convert_to_response:
        combined = convert_to_response_level(combined)
    
    return combined


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for the annotation data.
    
    Args:
        df: DataFrame with annotation data
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'n_annotations': len(df),
        'n_items': df['item_id'].nunique(),
        'n_agents': df['agent_id'].nunique(),
        'n_annotators': df['annotator_id'].nunique(),
        'label_distribution': df['label'].value_counts().to_dict(),
        'annotations_per_item': df.groupby('item_id').size().describe().to_dict(),
        'annotations_per_annotator': df.groupby('annotator_id').size().describe().to_dict(),
        'items_per_agent': df.groupby('agent_id')['item_id'].nunique().describe().to_dict(),
    }
    
    return summary


def print_data_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary of the annotation data."""
    summary = get_data_summary(df)
    
    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Total annotations: {summary['n_annotations']:,}")
    print(f"Unique items: {summary['n_items']:,}")
    print(f"Unique agents: {summary['n_agents']:,}")
    print(f"Unique annotators: {summary['n_annotators']:,}")
    print()
    print("Label distribution:")
    for label, count in sorted(summary['label_distribution'].items()):
        pct = 100 * count / summary['n_annotations']
        print(f"  Label {label}: {count:,} ({pct:.1f}%)")
    print()
    print(f"Annotations per item: mean={summary['annotations_per_item']['mean']:.2f}, "
          f"min={summary['annotations_per_item']['min']:.0f}, "
          f"max={summary['annotations_per_item']['max']:.0f}")
    print("=" * 60)


def create_item_agent_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a mapping from item_id to agent_id.
    
    Args:
        df: DataFrame with annotation data
        
    Returns:
        DataFrame with item_id and agent_id columns (unique pairs)
    """
    return df[['item_id', 'agent_id']].drop_duplicates()


def get_label_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Convert annotation data to a matrix format.
    
    Args:
        df: DataFrame with annotation data
        
    Returns:
        Tuple of (label_matrix, item_ids, annotator_ids)
        label_matrix has items as rows, annotators as columns
        NaN indicates missing annotations
    """
    pivot = df.pivot_table(
        index='item_id',
        columns='annotator_id',
        values='label',
        aggfunc='first'  # In case of duplicates, take first
    )
    
    return pivot, list(pivot.index), list(pivot.columns)


def detect_data_structure(df: pd.DataFrame) -> str:
    """
    Detect whether the data has one agent per item or multiple agents per item.
    
    Args:
        df: DataFrame with annotation data
        
    Returns:
        'single_agent' if each item has one agent
        'multi_agent' if items have multiple agents (need response-level scoring)
    """
    agents_per_item = df.groupby('item_id')['agent_id'].nunique()
    max_agents = agents_per_item.max()
    
    if max_agents > 1:
        return 'multi_agent'
    return 'single_agent'


def convert_to_response_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert data to response-level format by creating composite item IDs.
    
    For datasets where multiple agents respond to the same item (e.g., MT-Bench),
    this creates a unique response_id = item_id + agent_id so each agent's 
    response is scored independently.
    
    Args:
        df: DataFrame with columns item_id, agent_id, annotator_id, label
        
    Returns:
        DataFrame with response_id replacing item_id, original item_id preserved
    """
    df = df.copy()
    df['original_item_id'] = df['item_id']
    df['item_id'] = df['original_item_id'].astype(str) + '_' + df['agent_id'].astype(str)
    
    return df


if __name__ == "__main__":
    # Test with sample data
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data/processed"
    
    try:
        df = load_all_data(data_dir)
        print_data_summary(df)
    except FileNotFoundError as e:
        print(f"Error: {e}")
