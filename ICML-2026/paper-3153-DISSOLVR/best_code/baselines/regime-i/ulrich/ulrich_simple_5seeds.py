# ulrich_simple_5seeds.py
"""
Ulrich et al. (2024) Baseline - Simplified 5 Seed Evaluation

Train 5 independent models (one per seed) with tautomer augmentation.
Reports: RMSE ± std and R² across 5 seeds

Simplified from the original 5×5 design for practical compute constraints.
"""

# ================= SILENCE LOGGING =================
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_WARNINGS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_USE_LEGACY_KERAS'] = 'True'  # For Keras 3 compatibility

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# ================= IMPORTS =================
import numpy as np
import deepchem as dc
from deepchem.models import GraphConvModel
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import random
import time
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

# ================= CONFIG =================
SEEDS = [42, 101, 123, 456, 789]

BATCH_SIZE = 50
NUM_EPOCHS = 130
NEURONS_LAYER1 = 64
NEURONS_LAYER2 = 128
DROPOUT = 0.1
LEARNING_RATE = 3e-4
VAL_FRAC = 0.23  # 23% validation (paper: 77/23 split)

DATASETS = {
    "AqSolDB": ("../all_datasets/aqsoldb/train.csv", "../all_datasets/aqsoldb/test.csv"),
    "ESOL": ("../all_datasets/esol/train.csv", "../all_datasets/esol/test.csv"),
    "SC2": ("../all_datasets/sc2/train.csv", "../all_datasets/sc2/test.csv"),
}


# ================= TAUTOMER AUGMENTATION =================
def augment_training_data(dataset_df, max_tautomers=50):
    """Paper-faithful augmentation with SMILES variants then tautomers"""
    enumerator = rdMolStandardize.TautomerEnumerator()
    augmented_smiles = []
    augmented_values = []

    for idx, row in tqdm(dataset_df.iterrows(), total=len(dataset_df), 
                         desc="      Augmenting", leave=False):
        smi = row["SMILES"]
        val = row["LogS"]

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # Original
        augmented_smiles.append(smi)
        augmented_values.append(val)

        # SMILES variants (non-canonical)
        try:
            for _ in range(min(5, max_tautomers // 10)):
                rand_smi = Chem.MolToSmiles(mol, doRandom=True)
                if rand_smi != smi:
                    augmented_smiles.append(rand_smi)
                    augmented_values.append(val)
        except:
            pass

        # Tautomers
        try:
            tautomers = list(enumerator.Enumerate(mol))[:max_tautomers]
            for taut in tautomers:
                taut_smi = Chem.MolToSmiles(taut)
                if taut_smi != smi:
                    augmented_smiles.append(taut_smi)
                    augmented_values.append(val)
        except:
            pass

    return pd.DataFrame({"SMILES": augmented_smiles, "LogS": augmented_values})


def set_all_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_one_model(seed, train_df, test_df, featurizer):
    """Train one model with given seed and return RMSE, R²"""
    set_all_seeds(seed)
    
    print(f"\n  SEED {seed}")
    print(f"  {'='*50}")
    
    # Split train_df into train/val (on DataFrame level, before featurization)
    n_train = len(train_df)
    n_val = int(n_train * VAL_FRAC)
    
    # Shuffle indices
    indices = np.random.permutation(n_train)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    train_df_split = train_df.iloc[train_indices].reset_index(drop=True)
    val_df_split = train_df.iloc[val_indices].reset_index(drop=True)
    
    print(f"    Train: {len(train_df_split)}, Val: {len(val_df_split)}")
    
    # Augment training data (on DataFrame, before featurization)
    aug_train_df = augment_training_data(train_df_split, max_tautomers=50)
    print(f"    Augmented train: {len(aug_train_df)} samples")
    
    # Featurize augmented training data
    aug_train_dataset = dc.data.NumpyDataset(
        X=featurizer.featurize(aug_train_df["SMILES"].tolist()),
        y=aug_train_df["LogS"].values.reshape(-1, 1)
    )
    
    # Featurize validation data (no augmentation)
    val_dataset = dc.data.NumpyDataset(
        X=featurizer.featurize(val_df_split["SMILES"].tolist()),
        y=val_df_split["LogS"].values.reshape(-1, 1)
    )
    
    # Featurize test data
    test_dataset = dc.data.NumpyDataset(
        X=featurizer.featurize(test_df["SMILES"].tolist()),
        y=test_df["LogS"].values.reshape(-1, 1)
    )
    
    # Create model
    model = GraphConvModel(
        n_tasks=1,
        mode='regression',
        graph_conv_layers=[NEURONS_LAYER1, NEURONS_LAYER2],
        dense_layer_size=256,
        dropout=DROPOUT,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        model_dir=f"./_tmp_models/seed_{seed}"
    )
    
    # Train
    print(f"    Training for {NUM_EPOCHS} epochs...")
    pbar = tqdm(range(NUM_EPOCHS), desc="    Training", leave=False)
    for epoch in pbar:
        model.fit(aug_train_dataset, nb_epoch=1)
        
        if (epoch + 1) % 20 == 0:
            train_pred = model.predict(aug_train_dataset).flatten()[:len(aug_train_dataset.y)]
            train_rmse = np.sqrt(mean_squared_error(aug_train_dataset.y.flatten(), train_pred))
            
            val_pred = model.predict(val_dataset).flatten()[:len(val_dataset.y)]
            val_rmse = np.sqrt(mean_squared_error(val_dataset.y.flatten(), val_pred))
            
            pbar.set_postfix({'train': f'{train_rmse:.3f}', 'val': f'{val_rmse:.3f}'})
    
    # Evaluate on test
    test_pred = model.predict(test_dataset).flatten()[:len(test_dataset.y)]
    true_vals = test_dataset.y.flatten()
    
    rmse = np.sqrt(mean_squared_error(true_vals, test_pred))
    r2 = r2_score(true_vals, test_pred)
    
    print(f"    Test RMSE: {rmse:.4f} | R²: {r2:.4f}")
    
    # Cleanup
    import shutil
    if os.path.exists(f"./_tmp_models/seed_{seed}"):
        shutil.rmtree(f"./_tmp_models/seed_{seed}")
    
    return rmse, r2


def evaluate_dataset(name, train_path, test_path):
    """Evaluate on a dataset across 5 seeds"""
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Filter invalid SMILES
    train_df = train_df[train_df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
    test_df = test_df[test_df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
    
    print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
    
    # Featurizer
    featurizer = dc.feat.ConvMolFeaturizer()
    
    results = []
    times = []
    
    for seed in SEEDS:
        start_time = time.time()
        rmse, r2 = train_one_model(seed, train_df, test_df, featurizer)
        elapsed = time.time() - start_time
        results.append((rmse, r2))
        times.append(elapsed)
        print(f"  Seed {seed} completed in {elapsed:.1f}s")
    
    rmses = [r[0] for r in results]
    r2s = [r[1] for r in results]
    
    return {
        "name": name,
        "rmse_mean": np.mean(rmses),
        "rmse_std": np.std(rmses),
        "r2_mean": np.mean(r2s),
        "r2_std": np.std(r2s),
        "avg_time": np.mean(times)
    }


def main():
    print("=" * 60)
    print("Ulrich et al. (2024) - Simplified 5 Seed Evaluation")
    print("5 independent models with tautomer augmentation")
    print("=" * 60)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {gpus[0]}")
    else:
        print("No GPU detected, using CPU")
    
    all_results = []
    
    for name, (train_path, test_path) in DATASETS.items():
        result = evaluate_dataset(name, train_path, test_path)
        all_results.append(result)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS - Ulrich et al. Baseline (Simplified)")
    print("=" * 60)
    
    for r in all_results:
        print(f"\n{r['name']}:")
        print(f"  RMSE: {r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}")
        print(f"  R²:   {r['r2_mean']:.4f} ± {r['r2_std']:.4f}")
        print(f"  Avg Time/Seed: {r['avg_time']:.1f}s")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
