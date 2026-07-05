"""
TP-BERTa Embedding Model

Uses TP-BERTa (table-specific transformer) for embeddings.
"""

import numpy as np
import pandas as pd
import torch
import tempfile
import shutil
from typing import Optional, Union
from pathlib import Path

# TP-BERTa imports
from bin import build_default_model
from lib import DataConfig
from lib.data_utils import prepare_tpberta_loaders

from LTM.models.utils import generate_feature_names


class ModelArgs:
    """Arguments class for building TP-BERTa model."""
    def __init__(self, pretrain_dir, max_position_embeddings, max_feature_length,
                 max_numerical_token, max_categorical_token, feature_map, batch_size):
        self.base_model_dir = None
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = 5
        self.max_seq_length = 512
        self.max_feature_length = max_feature_length
        self.max_numerical_token = max_numerical_token
        self.max_categorical_token = max_categorical_token
        self.feature_map = feature_map
        self.batch_size = batch_size
        self.pretrain_dir = str(pretrain_dir)
        self.model_suffix = "pytorch_models/best"


def get_tpberta_embeddings(
        df: pd.DataFrame,
        pretrain_dir: str,
        feature_names_file: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        has_label: bool = True,
        dataset_name: str = "temp_dataset",
) -> np.ndarray:
    """
    Get TP-BERTa embeddings for a DataFrame.
    
    Args:
        df: DataFrame with features. If has_label=True, label should be the last column.
        pretrain_dir: Path to pre-trained TP-BERTa model
        feature_names_file: Path to feature_names.json
        device: Device to use (can be torch.device or str like "cuda"/"cpu")
        has_label: Whether the DataFrame has a label column (default: True)
        dataset_name: Name for temporary files
    
    Returns:
        numpy array of embeddings [N, hidden_size]
    
    Note:
        temp_dir is used because TP-BERTa's data loaders require reading CSV and 
        feature_names.json from the filesystem. We create a temporary directory,
        save the DataFrame as CSV there, process it, then clean up.
        
        If has_label=False, a dummy label column will be added (TP-BERTa requires it).
    """
    # Convert device string to torch.device if needed
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Create temporary directory for TP-BERTa processing
    # TP-BERTa data loaders need to read from filesystem, so we save DataFrame to temp CSV
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Prepare DataFrame: TP-BERTa requires label column as last column
        df_to_save = df.copy()
        if not has_label:
            # Add dummy label column (TP-BERTa data loader requires it)
            df_to_save['dummy_label'] = 0
        
        # Truncate very long strings to prevent memory issues when TP-BERTa reads CSV
        # TP-BERTa's CSV reader can allocate huge memory if strings are too long
        # Limit string columns to max 1000 characters (reasonable for tokenization)
        MAX_STRING_LENGTH = 1000
        print(f"  Processing {len(df_to_save.columns)} columns, truncating long strings (max: {MAX_STRING_LENGTH} chars)...")
        
        # Process ALL columns: convert to string and truncate if needed
        # This ensures no column will cause memory issues when TP-BERTa reads the CSV
        truncated_cols = []
        for col in df_to_save.columns:
            # Convert to string first (handles mixed types)
            df_to_save[col] = df_to_save[col].astype(str)
            
            # Check if any string is too long
            max_len = df_to_save[col].str.len().max()
            if pd.notna(max_len) and max_len > MAX_STRING_LENGTH:
                truncated_cols.append((col, max_len))
                # Truncate long strings
                df_to_save[col] = df_to_save[col].apply(
                    lambda x: x[:MAX_STRING_LENGTH] if len(str(x)) > MAX_STRING_LENGTH else x
                )
        
        if truncated_cols:
            print(f"  Truncated {len(truncated_cols)} columns:")
            for col, max_len in truncated_cols:
                print(f"    - '{col}': {max_len} -> {MAX_STRING_LENGTH} chars")
        else:
            print(f"  No columns needed truncation")
        
        # Save DataFrame as CSV
        # All columns are now strings with max length 1000, preventing fixed-length Unicode inference
        csv_path = temp_dir / f"{dataset_name}.csv"
        df_to_save.to_csv(csv_path, index=False)

        # Handle feature_names.json: TP-BERTa MUST read from filesystem
        # Auto-generate if not provided (will be cleaned up with temp_dir)
        temp_feature_names_file = temp_dir / "feature_names.json"
        
        if feature_names_file is None or not Path(feature_names_file).exists():
            # Auto-generate from DataFrame (temporary, will be cleaned up)
            generate_feature_names(df, temp_feature_names_file, has_label=has_label)
        else:
            # File path provided: copy to temp dir
            shutil.copy(feature_names_file, temp_feature_names_file)

        # Load pre-trained model
        pretrain_path = Path(pretrain_dir)
        data_config = DataConfig.from_pretrained(
            pretrain_path,
            data_dir=temp_dir,
            batch_size=32,
            train_ratio=1.0,  # Use all data
            preproc_type='lm',
            pre_train=False
        )

        # For embedding extraction, task_type doesn't matter (we only use encoder, not head)
        # TP-BERTa data loader requires a valid task_type, but it won't affect embeddings
        # Prepare data loaders
        data_loaders, datasets = prepare_tpberta_loaders(
            [dataset_name],
            data_config,
            tt="binclass"  # Default task_type (required by TP-BERTa, but doesn't affect embeddings)
        )

        if len(data_loaders) == 0:
            raise ValueError("Failed to prepare data loaders")

        data_loader, _ = data_loaders[0]
        dataset = datasets[0]

        # Build model (encoder only, no head needed for embeddings)
        args = ModelArgs(
            pretrain_path,
            max_position_embeddings=64,
            max_feature_length=8,
            max_numerical_token=256,
            max_categorical_token=16,
            feature_map="feature_names.json",
            batch_size=32
        )

        model_config, model = build_default_model(
            args, data_config, dataset.n_classes, device, pretrain=True
        )

        # Handle DataParallel: if model is wrapped, access via .module
        actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model

        # Extract embeddings (use CLS token from encoder output)
        actual_model.eval()
        all_embeddings = []

        with torch.no_grad():
            for batch in data_loader['train']:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop('labels', None)

                # Get encoder output
                outputs = actual_model.tpberta(**batch)
                # Extract CLS token (first token) from sequence output
                # Note: TPBertaForClassification and TPBertaForMTLPretrain both set
                # add_pooling_layer=False, so pooler_output is always None.
                # We directly use the CLS token from last_hidden_state.
                embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

                all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all embeddings
        final_embeddings = np.vstack(all_embeddings)

        return final_embeddings

    finally:
        # Cleanup temporary directory and all files (including feature_names.json)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

