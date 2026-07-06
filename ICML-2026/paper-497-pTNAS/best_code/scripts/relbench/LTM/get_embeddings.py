"""
Unified Text Embedding Module

Main interface for extracting embeddings from DataFrames.
Supports multiple embedding models via models/ submodule.
"""

import hashlib
import numpy as np
import pandas as pd
from typing import Optional, Union
import torch

from LTM.models.nomic import get_nomic_embeddings
from LTM.models.bge import get_bge_embeddings
from LTM.models.tpberta import get_tpberta_embeddings


def get_embeddings(
        df: pd.DataFrame,
        model: str = "tpberta",
        dataset_name: Optional[str] = None,
        pretrain_dir: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        has_label: bool = False,
        # TP-BERTa specific
        feature_names_file: Optional[str] = None,
        # Text model specific
        batch_size: int = 32,
        # Nomic specific
        task_prefix: str = "classification",
        # BGE specific (no additional params needed)
) -> np.ndarray:
    """
    Get embeddings from DataFrame (runtime function, returns numpy array).
    
    This is the main interface for runtime embedding extraction. It automatically handles
    feature_names.json generation and cleanup for TP-BERTa. For nomic/bge, it can optionally
    use standardized feature names.
    
    Args:
        df: Input DataFrame
        model: Model to use ("tpberta", "nomic", or "bge")
        dataset_name: Optional dataset name for feature_names.json generation (auto-generated if None)
        pretrain_dir: Path to TP-BERTa pretrained model (required for "tpberta")
        device: Device to use ("cuda", "cpu", or torch.device)
        has_label: Whether DataFrame has a label column (for TP-BERTa)
        feature_names_file: Optional path to feature_names.json. If None, will auto-generate
                           for TP-BERTa (temporary file, cleaned up after use).
                           For nomic/bge, if provided, uses standardized feature names as keys.
        batch_size: Batch size for text models
        task_prefix: Task prefix for nomic model ("classification", "search_query", etc.)
    
    Returns:
        numpy array of embeddings [N, embedding_dim]
    
    Examples:
        # TP-BERTa (auto-generates and cleans up feature_names.json)
        embeddings = get_embeddings(df, model="tpberta", pretrain_dir="...")
        
        # Nomic
        embeddings = get_embeddings(df, model="nomic", task_prefix="classification")
        
        # BGE
        embeddings = get_embeddings(df, model="bge")
    """
    # Auto-generate dataset_name if not provided
    if dataset_name is None:
        # Generate a unique name based on DataFrame columns and shape
        df_hash = hashlib.md5(str(df.columns.tolist()).encode()).hexdigest()[:8]
        dataset_name = f"dataset_{df_hash}"

    if model.lower() == "tpberta":
        if pretrain_dir is None:
            raise ValueError("pretrain_dir is required for TP-BERTa model")

        # For TP-BERTa, auto-generate feature_names.json if not provided
        # It will be created in temp dir and cleaned up automatically
        return get_tpberta_embeddings(
            df=df,
            pretrain_dir=pretrain_dir,
            feature_names_file=feature_names_file,
            device=device,
            has_label=has_label,
            dataset_name=dataset_name,
        )
    elif model.lower() == "nomic":
        return get_nomic_embeddings(
            df=df,
            task_prefix=task_prefix,
            batch_size=batch_size,
            device=device,
            feature_names_file=feature_names_file,
        )
    elif model.lower() == "bge":
        return get_bge_embeddings(
            df=df,
            batch_size=batch_size,
            device=device,
            feature_names_file=feature_names_file,
        )
    else:
        raise ValueError(
            f"Unknown model: {model}. "
            "Supported models: 'tpberta', 'nomic', 'bge'"
        )
