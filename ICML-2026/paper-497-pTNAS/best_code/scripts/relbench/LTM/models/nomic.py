"""
Nomic Embedding Model

Uses nomic-ai/nomic-embed-text-v1.5 for text embeddings.
"""

import json
import numpy as np
import pandas as pd
import torch
from typing import Optional, Union
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from LTM.models.utils import dataframe_to_texts


def get_nomic_embeddings(
        df: pd.DataFrame,
        task_prefix: str = "classification",
        batch_size: int = 32,
        device: Optional[Union[str, torch.device]] = None,
        feature_names_file: Optional[str] = None,
) -> np.ndarray:
    """
    Get embeddings using nomic-embed-text-v1.5 model.
    
    Args:
        df: Input DataFrame
        task_prefix: Task instruction prefix for nomic model
            Options: "search_document", "search_query", "clustering", "classification"
        batch_size: Batch size for encoding
        device: Device to use ("cuda", "cpu", or torch.device)
        feature_names_file: Optional path to feature_names.json. If provided, uses standardized
                           feature names as keys instead of original column names.
    
    Returns:
        numpy array of embeddings [N, embedding_dim]
    """
    # Convert device string to torch.device if needed
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Load feature names map if provided
    feature_names_map = None
    if feature_names_file and Path(feature_names_file).exists():
        with open(feature_names_file, 'r') as f:
            feature_names_map = json.load(f)

    # Load model
    print("  Loading nomic model...")
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        device=str(device)
    )
    print("  Model loaded. Converting DataFrame to texts...")

    # Convert DataFrame to texts with prefix
    prefix = f"{task_prefix}:"
    texts = dataframe_to_texts(df, prefix=prefix, feature_names_map=feature_names_map)
    print(f"  Converted {len(texts)} rows to texts. Starting encoding...")

    # Encode all texts directly with specified batch_size
    # model.encode() will handle batching internally
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=batch_size,  # Single batch_size parameter
        show_progress_bar=True  # Use built-in progress bar
    )

    return embeddings
