#!/usr/bin/env python3
"""
Baselining SolubNet (Graph Neural Network) for Aqueous Solubility Prediction

Trains SolubNet model using 10-fold cross-validation with multiple random seeds
on aqueous solubility datasets. Results are aggregated and saved to benchmark_results/.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import scipy.stats
from tqdm.auto import tqdm
import copy
import time

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
    print("Running in Google Colab")
except:
    IN_COLAB = False
    print("Running locally")

# Note: For Colab, install packages manually before running:
# !pip install -q dgl rdkit torch scikit-learn tqdm

# Import after installation
import torch as th
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dgl

# Setup base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'SolubNetD'))

# Import SolubNet modules
try:
    from mtMolDes import model, Utility
    print("✓ SolubNet modules imported successfully\n")
except ImportError as e:
    print(f"Error importing SolubNet modules: {e}")
    print("Please ensure SolubNetD folder is in the same directory as this script")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS = ["aqsoldb", "esol", "sc2"]
ALL_DATASETS_DIR = os.path.join(BASE_DIR, "all_datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "benchmark_results")
SEEDS = [42, 101, 123, 456, 789]  # 5 seeds with 10-fold CV each for fair comparison

# SolubNet hyperparameters
NUM_FEATURES = 4
NUM_LABELS = 1
FEATURE_STR = 'h'
LEARNING_RATE = 0.001
BATCH_SIZE = 2048
MAX_EPOCHS = 500
N_FOLDS = 10

# ============================================================================
# DEVICE SETUP
# ============================================================================

def setup_device():
    """Setup computation device (GPU if available)."""
    if th.cuda.is_available():
        device = th.device("cuda:0")
        print(f"✓ Using GPU: {th.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {th.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    # elif th.backends.mps.is_available():
    #     device = th.device("mps")
    #     print("✓ Using Apple Silicon GPU (MPS)\n")
    else:
        device = th.device("cpu")
        print("✓ Using CPU\n")
    return device

# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(dataset_name):
    """Load train and test data for a specific dataset."""
    train_file = os.path.join(ALL_DATASETS_DIR, dataset_name, "train.csv")
    test_file = os.path.join(ALL_DATASETS_DIR, dataset_name, "test.csv")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    return df_train, df_test


def load_graph_data(df, device):
    """Convert SMILES to graph representations."""
    print(f"  Converting {len(df)} molecules to graphs...")
    data = []
    failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Creating graphs",
                         disable=False):  # Always show progress bars
        try:
            graph = Utility.ParseSMILES(
                row['SMILES'],
                NUM_FEATURES,
                FEATURE_STR,
                device
            )
            # Ensure graph is on the correct device
            graph = graph.to(device)
            prop = float(row['LogS'])
            data.append([row['SMILES'], graph, prop])
        except Exception as e:
            failed += 1
            continue

    if failed > 0:
        print(f"  ⚠ Warning: Failed to process {failed}/{len(df)} molecules")

    return data


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def setup_seed(seed):
    """Setup random seeds for reproducibility."""
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    th.backends.cudnn.deterministic = True


def get_predictions(graphs, labels, net):
    """Get predictions for a batch of graphs using efficient DGL batching."""
    # Batch all graphs together for efficient GPU processing
    batched_graph = dgl.batch(graphs)

    # Forward pass on the entire batch at once
    predictions = net(batched_graph)

    # Efficient per-graph summation using scatter_add (avoids Python loops)
    batch_num_nodes = batched_graph.batch_num_nodes()

    # Create segment IDs for each node (which graph it belongs to)
    segment_ids = th.repeat_interleave(
        th.arange(len(graphs), device=predictions.device),
        batch_num_nodes
    )

    # Efficient segmented reduction using scatter_add
    graph_predictions = th.zeros(len(graphs), predictions.shape[1],
                                 device=predictions.device, dtype=predictions.dtype)
    graph_predictions.scatter_add_(0, segment_ids.unsqueeze(1).expand_as(predictions), predictions)

    # Squeeze to get final predictions per graph
    graph_predictions = graph_predictions.squeeze(1)

    return graph_predictions, labels


def create_mini_batches(num_samples, batch_size):
    """Create mini-batch indices."""
    if batch_size >= num_samples:
        return [[0, num_samples]]
    
    batch_idx = [[i * batch_size, (i + 1) * batch_size] 
                 for i in range(num_samples // batch_size)]
    
    if batch_idx[-1][1] != num_samples:
        batch_idx.append([batch_idx[-1][1], num_samples])
    
    return batch_idx


def criterion_r2(output, target):
    """Calculate R² coefficient."""
    target_mean = th.mean(target)
    ss_tot = th.sum((target - target_mean) ** 2)
    ss_res = th.sum((target - output) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


# ============================================================================
# TRAINING FUNCTION (10-FOLD CV)
# ============================================================================

def train_one_fold(fold, train_data, val_data, seed, device):
    """Train SolubNet for one fold."""
    fold_start_time = time.time()
    
    # Initialize model
    solubnet = model.GCNNet(NUM_FEATURES, NUM_LABELS, FEATURE_STR)
    solubnet.to(device)
    
    # Setup optimizer and scheduler
    optimizer = th.optim.Adam(
        solubnet.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=1e-4
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.9, 
        patience=3, 
        threshold=0.0001,
        min_lr=0.000001
    )
    
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    best_model = None
    best_epoch = 0
    patience_counter = 0
    max_patience = 40
    
    batch_idx = create_mini_batches(len(train_data), BATCH_SIZE)

    # Progress bar for epochs
    pbar = tqdm(range(MAX_EPOCHS), desc="      Epochs", leave=False, disable=False)

    for epoch in pbar:
        solubnet.train()
        epoch_loss = 0

        for idx in batch_idx:
            idx0, idx1 = idx[0], idx[1]
            
            # Get batch data
            batch_graphs = [train_data[i][1] for i in range(idx0, idx1)]
            batch_labels = th.tensor([train_data[i][2] for i in range(idx0, idx1)], 
                                    dtype=th.float32, device=device)
            
            # Forward pass
            y_pred, y_true = get_predictions(batch_graphs, batch_labels, solubnet)
            
            # Calculate loss (RMSE - 0.1 * R²)
            loss = th.sqrt(criterion(y_pred, y_true)) - 0.1 * criterion_r2(y_pred, y_true)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        solubnet.eval()
        with th.no_grad():
            val_graphs = [d[1] for d in val_data]
            val_labels = th.tensor([d[2] for d in val_data], 
                                   dtype=th.float32, device=device)
            
            val_pred, val_true = get_predictions(val_graphs, val_labels, solubnet)
            val_loss = th.sqrt(criterion(val_pred, val_true))

            scheduler.step(val_loss)

            # Update progress bar with metrics
            pbar.set_postfix({
                'train_loss': f'{epoch_loss/len(batch_idx):.4f}',
                'val_loss': f'{val_loss.item():.4f}',
                'best': f'{best_val_loss:.4f}',
                'patience': f'{patience_counter}/{max_patience}'
            })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(solubnet)
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                pbar.close()
                break

    pbar.close()
    
    fold_time = time.time() - fold_start_time
    
    return best_model, best_epoch, fold_time


def train_kfold_cv(dataset_name, train_data, test_data, seed, device):
    """Train SolubNet using 10-fold cross-validation for one seed."""
    print(f"\n  {'='*55}")
    print(f"  Seed {seed}: 10-Fold Cross-Validation")
    print(f"  {'='*55}")
    
    cv_start_time = time.time()
    
    # Setup seed
    setup_seed(seed)
    
    # Shuffle training data
    indices = np.random.permutation(len(train_data))
    shuffled_data = [train_data[i] for i in indices]
    
    # K-Fold split
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    
    fold_results = []
    trained_models = []
    total_fold_time = 0
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(shuffled_data)):
        print(f"    Fold {fold+1}/{N_FOLDS}...", end=' ')
        
        # Split data
        fold_train = [shuffled_data[i] for i in train_idx]
        fold_val = [shuffled_data[i] for i in val_idx]
        
        # Train one fold
        fold_model, best_epoch, fold_time = train_one_fold(
            fold, fold_train, fold_val, seed, device
        )
        
        total_fold_time += fold_time
        
        # Evaluate on validation set
        fold_model.eval()
        with th.no_grad():
            val_graphs = [d[1] for d in fold_val]
            val_labels = th.tensor([d[2] for d in fold_val], 
                                   dtype=th.float32, device=device)
            
            val_pred, val_true = get_predictions(val_graphs, val_labels, fold_model)
            
            val_mae = th.mean(th.abs(val_pred - val_true)).item()
            val_rmse = th.sqrt(th.mean((val_pred - val_true) ** 2)).item()
            val_r2 = r2_score(val_true.cpu().numpy(), val_pred.cpu().numpy())
            
            fold_results.append({
                'fold': fold + 1,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'best_epoch': best_epoch,
                'train_time': fold_time
            })
            
            print(f"MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, R²={val_r2:.4f} (epoch {best_epoch}, {fold_time/60:.1f}min)")
        
        trained_models.append(fold_model)
    
    # Calculate fold-level statistics
    fold_maes = [f['val_mae'] for f in fold_results]
    fold_rmses = [f['val_rmse'] for f in fold_results]
    fold_r2s = [f['val_r2'] for f in fold_results]
    
    print(f"\n    Fold Statistics (Validation):")
    print(f"      MAE:  {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}")
    print(f"      RMSE: {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}")
    print(f"      R²:   {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")
    print(f"      Total Training Time: {total_fold_time/60:.1f} minutes")
    
    # Evaluate all models on test set
    print(f"\n    Evaluating on test set with {len(trained_models)} models...")
    
    test_start_time = time.time()
    test_graphs = [d[1] for d in test_data]
    test_labels = th.tensor([d[2] for d in test_data], 
                            dtype=th.float32, device=device)
    
    # Ensemble prediction (average of all folds)
    all_test_preds = []
    
    for fold_model in trained_models:
        fold_model.eval()
        with th.no_grad():
            test_pred, _ = get_predictions(test_graphs, test_labels, fold_model)
            all_test_preds.append(test_pred.cpu().numpy())
    
    # Average predictions
    ensemble_pred = np.mean(all_test_preds, axis=0)
    test_labels_np = test_labels.cpu().numpy()
    
    # Calculate test metrics
    test_rmse = np.sqrt(mean_squared_error(test_labels_np, ensemble_pred))
    test_r2 = r2_score(test_labels_np, ensemble_pred)
    test_mae = np.mean(np.abs(test_labels_np - ensemble_pred))
    
    test_time = time.time() - test_start_time
    total_time = time.time() - cv_start_time
    
    print(f"    Test Results (Ensemble): MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R²={test_r2:.4f}")
    print(f"    Inference Time: {test_time:.2f} seconds")
    print(f"    Total Time (Train+Test): {total_time/60:.1f} minutes")
    
    # Save models
    output_dir = os.path.join(OUTPUT_DIR, dataset_name, "baselines", "SolubNet")
    os.makedirs(output_dir, exist_ok=True)
    
    for fold, fold_model in enumerate(trained_models):
        model_path = os.path.join(output_dir, f"model_seed{seed}_fold{fold+1}.pt")
        th.save(fold_model.state_dict(), model_path)
    
    return {
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'fold_results': fold_results,
        'fold_mae_mean': np.mean(fold_maes),
        'fold_mae_std': np.std(fold_maes),
        'fold_rmse_mean': np.mean(fold_rmses),
        'fold_rmse_std': np.std(fold_rmses),
        'fold_r2_mean': np.mean(fold_r2s),
        'fold_r2_std': np.std(fold_r2s),
        'train_time': total_fold_time,
        'test_time': test_time,
        'total_time': total_time
    }


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_all_seeds(dataset_name, train_data, test_data, device):
    """Train SolubNet on all seeds for a dataset."""
    print(f"\n{'='*60}")
    print(f"TRAINING SOLUBNET: {dataset_name}")
    print(f"{'='*60}")
    print(f"  Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    results = {
        'rmse': [], 'r2': [], 'mae': [], 
        'all_fold_results': [],
        'fold_mae_means': [], 'fold_mae_stds': [],
        'fold_rmse_means': [], 'fold_rmse_stds': [],
        'fold_r2_means': [], 'fold_r2_stds': [],
        'train_times': [], 'test_times': [], 'total_times': []
    }
    
    for seed in SEEDS:
        seed_results = train_kfold_cv(
            dataset_name, 
            train_data, 
            test_data, 
            seed, 
            device
        )
        
        results['rmse'].append(seed_results['test_rmse'])
        results['r2'].append(seed_results['test_r2'])
        results['mae'].append(seed_results['test_mae'])
        results['all_fold_results'].append(seed_results['fold_results'])
        
        results['fold_mae_means'].append(seed_results['fold_mae_mean'])
        results['fold_mae_stds'].append(seed_results['fold_mae_std'])
        results['fold_rmse_means'].append(seed_results['fold_rmse_mean'])
        results['fold_rmse_stds'].append(seed_results['fold_rmse_std'])
        results['fold_r2_means'].append(seed_results['fold_r2_mean'])
        results['fold_r2_stds'].append(seed_results['fold_r2_std'])
        
        results['train_times'].append(seed_results['train_time'])
        results['test_times'].append(seed_results['test_time'])
        results['total_times'].append(seed_results['total_time'])
    
    return results


# ============================================================================
# RESULTS REPORTING
# ============================================================================

def save_results(all_results):
    """Save aggregated results to JSON and CSV."""
    summary_data = []
    
    for dataset_name, metrics in all_results.items():
        # Test set metrics (from ensemble)
        rmse_mean = np.mean(metrics['rmse'])
        rmse_std = np.std(metrics['rmse'])
        r2_mean = np.mean(metrics['r2'])
        r2_std = np.std(metrics['r2'])
        mae_mean = np.mean(metrics['mae'])
        mae_std = np.std(metrics['mae'])
        
        # Fold-level statistics (variance across folds within each seed)
        fold_rmse_mean = np.mean(metrics['fold_rmse_means'])
        fold_rmse_std_avg = np.mean(metrics['fold_rmse_stds'])
        fold_r2_mean = np.mean(metrics['fold_r2_means'])
        fold_r2_std_avg = np.mean(metrics['fold_r2_stds'])
        fold_mae_mean = np.mean(metrics['fold_mae_means'])
        fold_mae_std_avg = np.mean(metrics['fold_mae_stds'])
        
        # Time statistics
        train_time_mean = np.mean(metrics['train_times'])
        train_time_std = np.std(metrics['train_times'])
        test_time_mean = np.mean(metrics['test_times'])
        total_time_mean = np.mean(metrics['total_times'])
        
        summary_data.append({
            'Dataset': dataset_name,
            'Model': 'SolubNet-10Fold',
            
            # Test metrics (ensemble of 10 folds)
            'Test_MAE_mean': mae_mean,
            'Test_MAE_std': mae_std,
            'Test_RMSE_mean': rmse_mean,
            'Test_RMSE_std': rmse_std,
            'Test_R2_mean': r2_mean,
            'Test_R2_std': r2_std,
            
            # Fold-level validation metrics (mean across seeds)
            'Fold_MAE_mean': fold_mae_mean,
            'Fold_MAE_std': fold_mae_std_avg,
            'Fold_RMSE_mean': fold_rmse_mean,
            'Fold_RMSE_std': fold_rmse_std_avg,
            'Fold_R2_mean': fold_r2_mean,
            'Fold_R2_std': fold_r2_std_avg,
            
            # Training information
            'Seeds': len(metrics['rmse']),
            'Folds_per_seed': N_FOLDS,
            'Total_models': len(metrics['rmse']) * N_FOLDS,
            
            # Time statistics
            'Train_time_minutes': train_time_mean / 60,
            'Train_time_std_minutes': train_time_std / 60,
            'Test_time_seconds': test_time_mean,
            'Total_time_minutes': total_time_mean / 60
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, "solubnet_10fold_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved summary to: {csv_path}")
    
    # Save to JSON (with all detailed results)
    json_path = os.path.join(OUTPUT_DIR, "solubnet_10fold_results.json")
    # Convert numpy types to native Python types for JSON serialization
    json_results = {}
    for dataset, metrics in all_results.items():
        json_results[dataset] = {
            'test_rmse': [float(x) for x in metrics['rmse']],
            'test_r2': [float(x) for x in metrics['r2']],
            'test_mae': [float(x) for x in metrics['mae']],
            'fold_results': metrics['all_fold_results'],
            'train_times_seconds': [float(x) for x in metrics['train_times']],
            'test_times_seconds': [float(x) for x in metrics['test_times']],
            'total_times_seconds': [float(x) for x in metrics['total_times']]
        }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ Saved full results to: {json_path}")
    
    # Print summary table
    print("\n" + "="*100)
    print("SOLUBNET 10-FOLD CV BENCHMARK SUMMARY")
    print("="*100)
    print(f"Reporting: Mean ± Std across {summary_df.iloc[0]['Seeds']} seeds")
    print(f"Each seed: 10-fold CV ensemble prediction on test set")
    print("="*100)

    for _, row in summary_df.iterrows():
        print(f"\n{row['Dataset']}:")
        print("-"*100)
        print(f"  Model: {row['Model']}")
        print(f"\n  TEST SET PERFORMANCE (Mean ± Std across {row['Seeds']} seeds, each with {row['Folds_per_seed']}-fold ensemble):")
        print(f"    MAE:  {row['Test_MAE_mean']:.4f} ± {row['Test_MAE_std']:.4f}")
        print(f"    RMSE: {row['Test_RMSE_mean']:.4f} ± {row['Test_RMSE_std']:.4f}")
        print(f"    R²:   {row['Test_R2_mean']:.4f} ± {row['Test_R2_std']:.4f}")
        print(f"\n  FOLD VALIDATION PERFORMANCE (Mean across {row['Folds_per_seed']} folds within each seed):")
        print(f"    MAE:  {row['Fold_MAE_mean']:.4f} ± {row['Fold_MAE_std']:.4f}")
        print(f"    RMSE: {row['Fold_RMSE_mean']:.4f} ± {row['Fold_RMSE_std']:.4f}")
        print(f"    R²:   {row['Fold_R2_mean']:.4f} ± {row['Fold_R2_std']:.4f}")
        print(f"\n  TRAINING INFO:")
        print(f"    Seeds: {row['Seeds']}")
        print(f"    Folds per seed: {row['Folds_per_seed']}")
        print(f"    Total models trained: {row['Total_models']}")
        print(f"    Training time per seed: {row['Train_time_minutes']:.1f} ± {row['Train_time_std_minutes']:.1f} minutes")
        print(f"    Test inference time per seed: {row['Test_time_seconds']:.2f} seconds")
        print(f"    Total time per seed: {row['Total_time_minutes']:.1f} minutes")
    
    # Create a comparison-ready summary (matches format of other baseline methods)
    comparison_df = pd.DataFrame([{
        'Dataset': row['Dataset'],
        'Model': 'SolubNet',
        'RMSE_mean': row['Test_RMSE_mean'],
        'RMSE_std': row['Test_RMSE_std'],
        'R2_mean': row['Test_R2_mean'],
        'R2_std': row['Test_R2_std'],
        'Seeds': row['Seeds'],
        'Training_Strategy': f"{row['Folds_per_seed']}-fold CV ensemble"
    } for _, row in summary_df.iterrows()])

    comparison_csv = os.path.join(OUTPUT_DIR, "solubnet_comparison_summary.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\n✓ Saved comparison summary to: {comparison_csv}")

    print("\n" + "="*100)
    print("COMPARISON FORMAT (for easy comparison with other baselines):")
    print("="*100)
    print(comparison_df.to_string(index=False))
    
    return summary_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("SOLUBNET 10-FOLD CROSS-VALIDATION BASELINE BENCHMARK")
    print("="*80)
    print(f"Datasets: {len(DATASETS)} {DATASETS}")
    print(f"Seeds per dataset: {len(SEEDS)} {SEEDS}")
    print(f"Folds per seed: {N_FOLDS}")
    print(f"Total models to train: {len(DATASETS) * len(SEEDS) * N_FOLDS} "
          f"({len(DATASETS)} datasets × {len(SEEDS)} seeds × {N_FOLDS} folds)")
    print(f"Strategy: Train 10-fold CV on each seed, ensemble predictions, report mean ± std across {len(SEEDS)} seeds")
    print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup device
    device = setup_device()
    
    all_results = {}
    
    for dataset_name in DATASETS:
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING DATASET: {dataset_name}")
        print(f"{'#'*80}")
        
        # Load data
        print(f"\nLoading {dataset_name} data...")
        df_train, df_test = load_dataset(dataset_name)
        print(f"  ✓ Train: {len(df_train)} samples")
        print(f"  ✓ Test: {len(df_test)} samples")
        
        # Convert to graph representations
        train_data = load_graph_data(df_train, device)
        test_data = load_graph_data(df_test, device)
        print(f"  ✓ Successfully created {len(train_data)} train graphs and {len(test_data)} test graphs")
        
        # Train on all seeds with 10-fold CV
        results = train_all_seeds(dataset_name, train_data, test_data, device)
        all_results[dataset_name] = results
    
    # Save and display results
    save_results(all_results)
    
    print("\n" + "="*80)
    print("✓ BENCHMARK COMPLETE")
    print("="*80)
    print(f"All models saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
