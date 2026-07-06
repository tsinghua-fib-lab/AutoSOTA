"""
TP-BERTa Training Module

Trains prediction head on preprocessed embedding data.
"""

import sys
from pathlib import Path
from typing import Optional, List
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.metrics import roc_auc_score, mean_absolute_error


class EmbeddingDataset(Dataset):
    """Dataset for loading embeddings and labels."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        """
        Args:
            embeddings: Pre-parsed numpy array of embeddings (shape: [N, embedding_dim])
            labels: Array of labels (shape: [N])
        """
        self.embeddings = embeddings.astype(np.float32)  # Ensure float32 for efficiency
        self.labels = labels.astype(np.float32)
        self.embedding_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else embeddings.shape[0]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        # Direct indexing - no string parsing needed!
        embedding = self.embeddings[idx]
        label_value = float(self.labels[idx])
        
        return {
            'embedding': torch.FloatTensor(embedding),
            'label': torch.FloatTensor([label_value])
        }


class TPBertaHead(nn.Module):
    """TP-BERTa prediction head (matches original TPBertaHead structure)."""

    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        # TP-BERTa: Linear(input_dim -> input_dim) -> Tanh -> Linear(input_dim -> output_dim)
        self.dense = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)  # Use single dropout layer (matches original)
        self.out_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)  # Reuse same dropout layer (matches original)
        x = self.out_proj(x)
        return x


def set_random_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_task_type(target_col_txt_path: str) -> str:
    """Load task type from target_col.txt."""
    target_col_txt_path = Path(target_col_txt_path)
    
    with open(target_col_txt_path, 'r') as f:
        lines = f.readlines()
        task_type_str = lines[1].strip()
    task_type_map = {
        "BINARY_CLASSIFICATION": "binclass",
        "REGRESSION": "regression"
    }
    task_type = task_type_map.get(task_type_str, "binclass")
    print(f"Detected task type from target_col.txt: {task_type}")
    return task_type


def load_embedding_data(data_dir: Path):
    """Load train/val/test embedding data from CSV files."""
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found in {data_dir}")
    if not val_csv.exists():
        raise FileNotFoundError(f"val.csv not found in {data_dir}")
    if not test_csv.exists():
        raise FileNotFoundError(f"test.csv not found in {data_dir}")

    # Load and parse embeddings in one go (much faster than parsing in __getitem__)
    def parse_embeddings(embedding_strings: List[str]) -> np.ndarray:
        """Parse comma-separated embedding strings to numpy array."""
        print(f"  Parsing {len(embedding_strings)} embeddings...")
        embeddings = []
        for emb_str in embedding_strings:
            embeddings.append([float(x) for x in emb_str.split(",")])
        return np.array(embeddings, dtype=np.float32)
    
    train_df = pd.read_csv(train_csv)
    train_embeddings = parse_embeddings(train_df['embedding'].tolist())
    train_labels = train_df['target'].values.astype(np.float32)

    val_df = pd.read_csv(val_csv)
    val_embeddings = parse_embeddings(val_df['embedding'].tolist())
    val_labels = val_df['target'].values.astype(np.float32)

    test_df = pd.read_csv(test_csv)
    test_embeddings = parse_embeddings(test_df['embedding'].tolist())
    test_labels = test_df['target'].values.astype(np.float32)

    print(f"  Train: {len(train_embeddings)} rows, embedding_dim: {train_embeddings.shape[1]}")
    print(f"  Val: {len(val_embeddings)} rows")
    print(f"  Test: {len(test_embeddings)} rows")
    print(f"Data split: Train={len(train_embeddings)}, Val={len(val_embeddings)}, Test={len(test_embeddings)}")

    return (train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels)


def check_data_distribution(train_labels, val_labels, test_labels, task_type: str):
    """Check and print data distribution information."""
    all_labels = np.concatenate([train_labels, val_labels, test_labels])
    
    print(f"\nData distribution:")
    print(f"  Total samples: {len(all_labels)}")
    
    if task_type == "regression":
        # For regression: show statistical summary
        print(f"  Label statistics:")
        print(f"    Mean: {np.mean(all_labels):.6f}")
        print(f"    Std: {np.std(all_labels):.6f}")
        print(f"    Min: {np.min(all_labels):.6f}")
        print(f"    Max: {np.max(all_labels):.6f}")
        print(f"    25th percentile: {np.percentile(all_labels, 25):.6f}")
        print(f"    50th percentile (median): {np.percentile(all_labels, 50):.6f}")
        print(f"    75th percentile: {np.percentile(all_labels, 75):.6f}")
        
        # Also show split-specific stats
        print(f"\n  Split-specific statistics:")
        print(f"    Train: mean={np.mean(train_labels):.6f}, std={np.std(train_labels):.6f}, range=[{np.min(train_labels):.6f}, {np.max(train_labels):.6f}]")
        print(f"    Val: mean={np.mean(val_labels):.6f}, std={np.std(val_labels):.6f}, range=[{np.min(val_labels):.6f}, {np.max(val_labels):.6f}]")
        print(f"    Test: mean={np.mean(test_labels):.6f}, std={np.std(test_labels):.6f}, range=[{np.min(test_labels):.6f}, {np.max(test_labels):.6f}]")
    else:
        # For classification: show class distribution
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  Label {label}: {count} ({count / len(all_labels) * 100:.1f}%)")
        
        if task_type == "binclass" and len(unique_labels) == 2:
            pos_ratio = counts[1] / len(all_labels) if len(counts) > 1 else 0
            print(f"  Positive class ratio: {pos_ratio:.3f}")
            if pos_ratio < 0.1 or pos_ratio > 0.9:
                print(f"  ⚠️  Warning: Highly imbalanced dataset! This may affect AUC.")


def train_prediction_head(
        data_dir: str,
        output_dir: str,
        target_col_txt_path: str,
        dropout: float = 0.1,
        batch_size: int = 256,
        learning_rate: float = 0.005,
        num_epochs: int = 200,
        early_stop: int = 10,
        max_round_epoch: int = 20,
        device: Optional[str] = None,
        seed: int = 42,
) -> dict:
    """
    Train prediction head on preprocessed embedding data.
    
    This function loads embeddings from separate train/val/test CSV files (generated by process_tables.py) 
    and trains a simple MLP prediction head. No TP-BERTa model is loaded - only the embeddings.
    
    Args:
        data_dir: Directory containing train.csv, val.csv, test.csv (each with columns: embedding, target)
        output_dir: Directory to save results
        target_col_txt_path: Path to target_col.txt (required)
        dropout: Dropout rate (default: 0.1)
        batch_size: Batch size
        learning_rate: Learning rate
        num_epochs: Maximum number of epochs
        early_stop: Early stopping patience
        max_round_epoch: Maximum number of batches per epoch (default: 20)
        device: Device to use (default: "cuda" if available)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Dictionary with training results and metrics
    """
    # Set random seeds for reproducibility
    set_random_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(data_dir)

    # Load task type (target_col_txt_path must be provided)
    if target_col_txt_path is None:
        raise ValueError("target_col_txt_path must be provided")
    task_type = load_task_type(target_col_txt_path)
    
    # Load data from separate CSV files
    print(f"Loading embedding data from {data_dir}...")
    train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels = load_embedding_data(data_dir)

    # Create datasets (embeddings are already parsed)
    train_dataset = EmbeddingDataset(train_embeddings, train_labels)
    val_dataset = EmbeddingDataset(val_embeddings, val_labels)
    test_dataset = EmbeddingDataset(test_embeddings, test_labels)

    # Use detected embedding dimension
    embedding_dim = train_dataset.embedding_dim
    print(f"Detected embedding dimension: {embedding_dim}")

    # Check data distribution
    check_data_distribution(train_labels, val_labels, test_labels, task_type)

    # Set generator for DataLoader reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    # For regression, set dropout to 0 to match dnn_baseline behavior
    model_dropout = 0.0 if task_type == "regression" else dropout
    model = TPBertaHead(
        input_dim=embedding_dim,
        output_dim=1,
        dropout=model_dropout
    ).to(device)
    print(f"Model architecture: TP-BERTa head (Input={embedding_dim} -> {embedding_dim} -> 1)")
    if task_type == "regression":
        print("  Note: Dropout disabled for regression task (consistent with dnn_baseline)")

    # Setup loss and optimizer
    if task_type == "binclass":
        criterion = nn.BCEWithLogitsLoss()
        metric_fn = roc_auc_score
        higher_is_better = True
        is_regression = False
    else:  # regression
        criterion = nn.L1Loss()  # Use L1Loss for regression
        metric_fn = mean_absolute_error  # Use MAE for regression
        higher_is_better = False
        is_regression = True

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_metric = -np.inf if higher_is_better else np.inf
    best_model_state = None
    no_improvement = 0
    train_losses = []
    val_metrics = []

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Task: {task_type}, Device: {device}")
    print("=" * 60)

    for epoch in range(num_epochs):
        # Training
        model.train()
        loss_accum = 0.0
        count_accum = 0
        for idx, batch in enumerate(train_loader):
            if idx > max_round_epoch:
                break
                
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].view(-1).to(device)  # Use view(-1) instead of squeeze()

            optimizer.zero_grad()
            logits = model(embeddings).view(-1)  # Use view(-1) instead of squeeze()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_accum += loss.item()
            count_accum += 1

        train_loss = loss_accum / count_accum if count_accum > 0 else 0.0
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embedding'].to(device)
                labels = batch['label'].view(-1).cpu().numpy()  # Use view(-1) to avoid 0-d array issue

                logits = model(embeddings).view(-1).detach().cpu().numpy()  # Use view(-1) + detach()
                
                # sigmoid for classification, logits for regression
                if task_type == "binclass":
                    pred_probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
                    val_preds.extend(pred_probs.tolist())  # Use tolist() for safety
                else:  # regression
                    val_preds.extend(logits.tolist())  # Use tolist() for safety

                val_targets.extend(labels.tolist())  # Use tolist() for safety

        # Check if we have both classes in validation set
        if task_type == "binclass":
            val_targets_array = np.array(val_targets)
            unique_val_labels = np.unique(val_targets_array)
            if len(unique_val_labels) < 2:
                print(f"  ⚠️  Warning: Validation set has only one class ({unique_val_labels[0]}). AUC cannot be calculated.")
                val_metric = 0.5  # Default to 0.5 if only one class
            else:
                val_metric = metric_fn(val_targets_array, np.array(val_preds))
                # Debug info for first few epochs
                if epoch < 3:
                    print(f"    Debug: Val labels unique: {unique_val_labels}, "
                          f"Val preds range: [{np.min(val_preds):.4f}, {np.max(val_preds):.4f}], "
                          f"Val preds mean: {np.mean(val_preds):.4f}")
        else:
            val_metric = metric_fn(np.array(val_targets), np.array(val_preds))
        
        # Convert to Python native float for JSON serialization
        val_metric = float(val_metric)
        val_metrics.append(val_metric)

        # Check for improvement
        is_better = (higher_is_better and val_metric > best_val_metric) or \
                    (not higher_is_better and val_metric < best_val_metric)

        if is_better:
            best_val_metric = val_metric
            best_model_state = model.state_dict().copy()
            no_improvement = 0
            print(f"Epoch {epoch + 1:3d} | Loss: {train_loss:.6f} | Val {metric_fn.__name__}: {val_metric:.6f} *")
        else:
            no_improvement += 1
            print(f"Epoch {epoch + 1:3d} | Loss: {train_loss:.6f} | Val {metric_fn.__name__}: {val_metric:.6f}")

        # Early stopping
        if no_improvement >= early_stop:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
            for batch in test_loader:
                embeddings = batch['embedding'].to(device)
                labels = batch['label'].view(-1).cpu().numpy()  # Use view(-1) to avoid 0-d array issue

                logits = model(embeddings).view(-1).detach().cpu().numpy()  # Use view(-1) + detach()
                
                # sigmoid for classification, logits for regression
                if task_type == "binclass":
                    pred_probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
                    test_preds.extend(pred_probs.tolist())  # Use tolist() for safety
                else:  # regression
                    test_preds.extend(logits.tolist())  # Use tolist() for safety

                test_targets.extend(labels.tolist())  # Use tolist() for safety

    test_metric = metric_fn(test_targets, test_preds)
    # Convert to Python native float for JSON serialization
    test_metric = float(test_metric)

    # Save results
    # Convert all NumPy types to Python native types for JSON serialization
    results = {
        'best_val_metric': float(best_val_metric),
        'test_metric': float(test_metric),
        'task_type': task_type,
        'embedding_dim': int(embedding_dim),
        'num_epochs_trained': int(epoch + 1),
        'train_losses': [float(x) for x in train_losses],
        'val_metrics': [float(x) for x in val_metrics],
    }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions
    np.save(output_dir / "test_predictions.npy", np.array(test_preds))
    np.save(output_dir / "test_targets.npy", np.array(test_targets))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Val {metric_fn.__name__}: {best_val_metric:.6f}")
    print(f"Test {metric_fn.__name__}: {test_metric:.6f}")

    return results


def main():
    """Main function to train prediction head from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train prediction head on TP-BERTa embeddings"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing train.csv, val.csv, test.csv (each with columns: embedding, target)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--target_col_txt",
        type=str,
        required=True,
        help="Path to target_col.txt"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (default: 256)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Learning rate (default: 0.005)"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=200,
        help="Maximum number of epochs (default: 200)"
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)"
    )
    parser.add_argument(
        "--max_round_epoch",
        type=int,
        default=20,
        help="Maximum number of batches per epoch (default: 20)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    try:
        results = train_prediction_head(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            target_col_txt_path=args.target_col_txt,
            dropout=args.dropout,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.max_epochs,
            early_stop=args.early_stop,
            max_round_epoch=args.max_round_epoch,
            device=args.device,
            seed=args.seed,
        )
        print(f"\n✅ Training completed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
