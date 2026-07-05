"""
Table Preprocessing Module

Converts CSV files to embedding format with multiple model support.
Supports: tpberta, nomic, bge
"""

import os
import argparse
import traceback
from pathlib import Path
from typing import Optional
import pandas as pd
import torch

from LTM.get_embeddings import get_embeddings
from LTM.models.utils import generate_feature_names


def process_split(
        df: pd.DataFrame,
        split_name: str,
        target_col: str,
        model: str,
        pretrain_dir: Optional[str],
        feature_names_file: str,
        device: str,
        batch_size: int,
        task_prefix: str,
):
    """Process a single split and return embeddings + targets."""
    print(f"   Processing {split_name} split ({len(df)} rows)...")

    # Prepare feature DataFrame (without target column)
    feature_df = df.drop(columns=[target_col])

    # Get embeddings using unified interface
    embeddings = get_embeddings(
        df=feature_df,
        model=model,
        pretrain_dir=pretrain_dir,
        feature_names_file=feature_names_file,
        device=device,
        has_label=False,  # We already separated features from target
        batch_size=batch_size,
        task_prefix=task_prefix,
    )

    # Convert embeddings to comma-separated strings
    embedding_strings = []
    for emb in embeddings:
        emb_str = ",".join([str(x) for x in emb.flatten()])
        embedding_strings.append(emb_str)

    # Get targets
    targets = df[target_col].values

    return embedding_strings, targets


def preprocess(
        input_dir: str,
        output_dir: str,
        model: str = "tpberta",
        device: Optional[str] = None,
        batch_size: int = 32,
        task_prefix: str = "classification",
) -> str:
    """
    Offline preprocessing: Convert TableData format to embedding format.
    
    This function performs offline batch processing:
    1. Reads train/val/test CSV files
    2. Generates embeddings for each row using specified model
    3. Saves feature_names.json (persistent, not cleaned up)
    4. Outputs train.csv, val.csv, test.csv with embeddings (2-column: embedding, target)
    
    Args:
        input_dir: Directory containing train.csv, val.csv, test.csv, and target_col.txt
        output_dir: Output directory for embedding format files
        model: Embedding model to use ("tpberta", "nomic", or "bge")
        device: Device to use (default: "cuda" if available, else "cpu")
        batch_size: Batch size for text models (nomic, bge)
        task_prefix: Task prefix for nomic model ("classification", "search_query", etc.)
    
    Returns:
        Path to output directory
    
    Note:
        This is for offline batch processing. For runtime embedding extraction,
        use get_embeddings() instead.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get pretrain_dir from environment variable (only needed for TP-BERTa)
    pretrain_dir = None
    if model.lower() == "tpberta":
        pretrain_dir = os.environ.get("TPBERTA_PRETRAIN_DIR")
        if pretrain_dir is None:
            raise ValueError(
                "TPBERTA_PRETRAIN_DIR environment variable not set. "
                "Please set TPBERTA_PRETRAIN_DIR to the path of pre-trained TP-BERTa model."
            )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load target column info from target_col.txt (assumed to exist)
    target_col_file = input_dir / "target_col.txt"
    with open(target_col_file, 'r') as f:
        lines = f.readlines()
        target_col = lines[0].strip()
        task_type = lines[1].strip() if len(lines) > 1 else "BINARY_CLASSIFICATION"

    # Load all splits
    train_df = pd.read_csv(input_dir / "train.csv")
    val_df = pd.read_csv(input_dir / "val.csv")
    test_df = pd.read_csv(input_dir / "test.csv")

    # Verify target column exists
    if target_col not in train_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in CSV. "
            f"Available columns: {list(train_df.columns)}"
        )

    # Make sure target column is last
    all_columns = [col for col in train_df.columns if col != target_col] + [target_col]
    train_df = train_df[all_columns]
    val_df = val_df[all_columns]
    test_df = test_df[all_columns]

    # Generate feature_names.json (needed for embedding generation)
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    feature_names_file = output_dir / "feature_names.json"
    generate_feature_names(combined_df, feature_names_file)

    print(f"🔄 Generating embeddings using {model} model...")
    print(f"   Device: {device}")
    if pretrain_dir:
        print(f"   Pretrain dir: {pretrain_dir}")

    # Process all splits separately
    train_embeddings, train_targets = process_split(
        df=train_df,
        split_name="train",
        target_col=target_col,
        model=model,
        pretrain_dir=pretrain_dir,
        feature_names_file=str(feature_names_file),
        device=device,
        batch_size=batch_size,
        task_prefix=task_prefix,
    )
    val_embeddings, val_targets = process_split(
        df=val_df,
        split_name="val",
        target_col=target_col,
        model=model,
        pretrain_dir=pretrain_dir,
        feature_names_file=str(feature_names_file),
        device=device,
        batch_size=batch_size,
        task_prefix=task_prefix,
    )
    test_embeddings, test_targets = process_split(
        df=test_df,
        split_name="test",
        target_col=target_col,
        model=model,
        pretrain_dir=pretrain_dir,
        feature_names_file=str(feature_names_file),
        device=device,
        batch_size=batch_size,
        task_prefix=task_prefix,
    )

    # Save each split as separate CSV file
    dataset_name = input_dir.name

    train_output_df = pd.DataFrame({
        'embedding': train_embeddings,
        'target': train_targets
    })
    train_csv = output_dir / "train.csv"
    train_output_df.to_csv(train_csv, index=False)

    val_output_df = pd.DataFrame({
        'embedding': val_embeddings,
        'target': val_targets
    })
    val_csv = output_dir / "val.csv"
    val_output_df.to_csv(val_csv, index=False)

    test_output_df = pd.DataFrame({
        'embedding': test_embeddings,
        'target': test_targets
    })
    test_csv = output_dir / "test.csv"
    test_output_df.to_csv(test_csv, index=False)

    print(f"✅ Converted to embedding format with {model} model:")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Model: {model}")
    print(f"   Train CSV: {train_csv} ({len(train_output_df)} rows, 2 columns: embedding, target)")
    print(f"   Val CSV:   {val_csv} ({len(val_output_df)} rows, 2 columns: embedding, target)")
    print(f"   Test CSV:  {test_csv} ({len(test_output_df)} rows, 2 columns: embedding, target)")
    print(f"   Feature names: {feature_names_file}")
    print(f"   Target column: {target_col}")
    print(f"   Task type: {task_type}")
    print(f"   Total rows: {len(train_output_df) + len(val_output_df) + len(test_output_df)}")

    return str(train_csv)


def main():
    """Main function to convert TableData format to embedding format."""

    parser = argparse.ArgumentParser(
        description="Convert TableData format to embedding format with multiple model support"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing train.csv, val.csv, test.csv, and target_col.txt"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for embedding format files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tpberta",
        choices=["tpberta", "nomic", "bge"],
        help="Embedding model to use: tpberta, nomic, or bge (default: tpberta)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for text models (nomic, bge, default: 32)"
    )
    parser.add_argument(
        "--task_prefix",
        type=str,
        default="classification",
        help="Task prefix for nomic model (default: classification)"
    )

    args = parser.parse_args()

    try:
        output_csv = preprocess(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model=args.model,
            device=args.device,
            batch_size=args.batch_size,
            task_prefix=args.task_prefix,
        )
        print(f"\n✅ Success! Output CSV: {output_csv}")
        print(f"   Format: 2 columns (embedding, target)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
