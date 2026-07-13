# ulrich_5seeds.py
"""
Ulrich et al. (2024) Baseline - 5 Seed Evaluation with Consensus

For each of 5 outer seeds:
  - Train 5 inner models (with different train/val splits)
  - Average predictions to get consensus RMSE
  
Total: 25 models per dataset
Reports: RMSE ± std and R² across 5 outer seeds
"""

# ================= SILENCE LOGGING =================
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_WARNINGS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# Configure TensorFlow for GPU usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to prevent TF from allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Enable mixed precision for 2x faster training
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        # Enable XLA (Accelerated Linear Algebra) for additional speedup
        tf.config.optimizer.set_jit(True)
        print(f"GPU(s) configured: {len(gpus)} device(s)")
        print(f"Mixed precision enabled (float16)")
        print(f"XLA compilation enabled")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# ================= IMPORTS =================
import numpy as np
import deepchem as dc
from deepchem.feat.mol_graphs import ConvMol
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import random
import time
from tqdm import tqdm
from mygraphconvmodel import MyGraphConvModel

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

# ================= CONFIG =================
OUTER_SEEDS = [42, 101, 123, 456, 789]
N_INNER_MODELS = 5  # 5 models per outer seed (consensus ensemble)
PRIMES = [2, 3, 5, 7, 11]  # For generating inner seeds

BATCH_SIZE = 1024  # Large batch to reduce CPU preprocessing overhead
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

        smiles_variants = set()
        smiles_variants.add(Chem.MolToSmiles(mol, canonical=True))
        
        mol_h = Chem.AddHs(mol)
        smiles_variants.add(Chem.MolToSmiles(mol_h, canonical=True))
        
        mol_no_h = Chem.RemoveHs(mol)
        smiles_variants.add(Chem.MolToSmiles(mol_no_h, canonical=True))
        
        try:
            mol_kek = Chem.Mol(mol)
            Chem.Kekulize(mol_kek, clearAromaticFlags=True)
            smiles_variants.add(Chem.MolToSmiles(mol_kek, kekuleSmiles=True))
        except:
            pass
        
        for _ in range(4):
            try:
                smiles_variants.add(Chem.MolToSmiles(mol, doRandom=True))
            except:
                pass
        
        try:
            inchi = Chem.MolToInchi(mol)
            mol_from_inchi = Chem.MolFromInchi(inchi)
            if mol_from_inchi:
                smiles_variants.add(Chem.MolToSmiles(mol_from_inchi))
        except:
            pass

        all_structures = set()
        for variant_smi in smiles_variants:
            all_structures.add(variant_smi)
            variant_mol = Chem.MolFromSmiles(variant_smi)
            if variant_mol is None:
                continue
            
            try:
                tautomers = enumerator.Enumerate(variant_mol)
                for tauto_mol in tautomers:
                    all_structures.add(Chem.MolToSmiles(tauto_mol))
                    if len(all_structures) >= max_tautomers:
                        break
            except:
                continue
            
            if len(all_structures) >= max_tautomers:
                break
        
        for structure_smi in all_structures:
            augmented_smiles.append(structure_smi)
            augmented_values.append(val)

    aug_df = pd.DataFrame({"SMILES": augmented_smiles, "LogS": augmented_values})
    aug_df = aug_df.drop_duplicates(subset=["SMILES"])
    
    return aug_df


def dataset_to_df(dc_dataset):
    """Convert DeepChem dataset to DataFrame"""
    return pd.DataFrame({
        "SMILES": dc_dataset.ids,
        "LogS": dc_dataset.y.flatten()
    })


def df_to_dataset(df, featurizer, tag):
    """Convert DataFrame to DeepChem dataset"""
    loader = dc.data.CSVLoader(
        tasks=["LogS"],
        feature_field="SMILES",
        featurizer=featurizer
    )
    tmp_csv = f"_tmp_aug_{tag}.csv"
    df.to_csv(tmp_csv, index=False)
    dataset = loader.featurize(tmp_csv)
    os.remove(tmp_csv)
    return dataset


def data_generator(dataset, batch_size, epochs=1):
    """Data generator for model training/prediction"""
    for _ in range(epochs):
        for X_b, y_b, w_b, ids_b in dataset.iterbatches(
            batch_size=batch_size,
            deterministic=True,
            pad_batches=True
        ):
            multiConvMol = ConvMol.agglomerate_mols(X_b)
            inputs = [
                multiConvMol.get_atom_features(),
                multiConvMol.deg_slice,
                np.array(multiConvMol.membership)
            ]
            for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
                inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
            yield (inputs, [y_b], [w_b])


def evaluate_model(model, dataset, batch_size):
    """Evaluate model and return RMSE, R2, predictions"""
    predictions = []
    device = '/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'

    for X_b, y_b, w_b, ids_b in dataset.iterbatches(
        batch_size=batch_size,
        deterministic=True,
        pad_batches=True
    ):
        multiConvMol = ConvMol.agglomerate_mols(X_b)

        with tf.device(device):
            # Convert inputs to tensors and move to GPU
            inputs = [
                tf.convert_to_tensor(multiConvMol.get_atom_features(), dtype=tf.float32),
                tf.convert_to_tensor(multiConvMol.deg_slice, dtype=tf.int32),
                tf.convert_to_tensor(np.array(multiConvMol.membership), dtype=tf.int32)
            ]
            for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
                inputs.append(tf.convert_to_tensor(multiConvMol.get_deg_adjacency_lists()[i], dtype=tf.int32))

            pred_batch = model(inputs, training=False)
            predictions.append(pred_batch.numpy())

    pred = np.concatenate(predictions, axis=0).flatten()[:len(dataset.y)]
    true = dataset.y.flatten()

    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)

    return rmse, r2, pred


def set_all_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_inner_model(inner_seed, train_dataset, val_dataset, featurizer, outer_seed, inner_idx):
    """Train one inner model with given seed"""
    set_all_seeds(inner_seed)

    # Convert to df, augment, convert back
    train_df = dataset_to_df(train_dataset)
    aug_train_df = augment_training_data(train_df, max_tautomers=50)
    aug_train_dataset = df_to_dataset(aug_train_df, featurizer, tag=f"o{outer_seed}_i{inner_idx}")

    # Determine device
    device = '/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'

    with tf.device(device):
        # Create custom MyGraphConvModel
        model = MyGraphConvModel(
            batch_size=BATCH_SIZE,
            neuronslayer1=NEURONS_LAYER1,
            neuronslayer2=NEURONS_LAYER2,
            dropout=DROPOUT
        )

        # Compile model with optimizer and loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

    # Compiled training step for faster execution
    @tf.function(jit_compile=True)  # XLA compilation
    def train_step(inputs_list, y_batch):
        with tf.GradientTape() as tape:
            predictions = model(inputs_list, training=True)
            loss = tf.reduce_mean(tf.square(predictions - y_batch))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # Train with validation monitoring
    best_val_rmse = float('inf')
    pbar = tqdm(range(NUM_EPOCHS), desc=f"      Training", leave=False)

    for epoch in pbar:
        # Training loop
        epoch_losses = []
        for X_b, y_b, w_b, ids_b in aug_train_dataset.iterbatches(
            batch_size=BATCH_SIZE,
            deterministic=False,
            pad_batches=True
        ):
            multiConvMol = ConvMol.agglomerate_mols(X_b)

            with tf.device(device):
                # Convert inputs to tensors and move to GPU
                inputs = [
                    tf.convert_to_tensor(multiConvMol.get_atom_features(), dtype=tf.float32),
                    tf.convert_to_tensor(multiConvMol.deg_slice, dtype=tf.int32),
                    tf.convert_to_tensor(np.array(multiConvMol.membership), dtype=tf.int32)
                ]
                for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
                    inputs.append(tf.convert_to_tensor(multiConvMol.get_deg_adjacency_lists()[i], dtype=tf.int32))

                y_b_tensor = tf.convert_to_tensor(y_b, dtype=tf.float32)

                # Use compiled training step
                loss = train_step(inputs, y_b_tensor)
                epoch_losses.append(loss.numpy())

        # Monitor validation every 20 epochs
        if (epoch + 1) % 20 == 0:
            train_rmse, _, _ = evaluate_model(model, aug_train_dataset, BATCH_SIZE)
            val_rmse, _, _ = evaluate_model(model, val_dataset, BATCH_SIZE)
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
            pbar.set_postfix({'train': f'{train_rmse:.3f}', 'val': f'{val_rmse:.3f}'})

    return model


def train_one_outer_seed(outer_seed, full_train_dataset, test_dataset, featurizer):
    """Train 5 inner models and return consensus predictions"""
    print(f"\n  OUTER SEED {outer_seed}")
    print(f"  {'='*50}")
    
    splitter = dc.splits.RandomSplitter()
    test_predictions_list = []
    
    for i in range(N_INNER_MODELS):
        inner_seed = outer_seed * PRIMES[i]
        print(f"\n    Inner Model {i+1}/{N_INNER_MODELS} (seed={inner_seed})")
        
        # Split train into train/val using inner_seed
        train_split, val_split = splitter.train_test_split(
            full_train_dataset, 
            frac_train=1.0 - VAL_FRAC,
            seed=inner_seed
        )
        
        print(f"    Train: {len(train_split)}, Val: {len(val_split)}")
        
        # Train model
        model = train_inner_model(inner_seed, train_split, val_split, featurizer, outer_seed, i)
        
        # Get test predictions
        _, _, test_pred = evaluate_model(model, test_dataset, BATCH_SIZE)
        test_predictions_list.append(test_pred)

        # Cleanup model from memory
        del model
        tf.keras.backend.clear_session()
    
    # Consensus: average predictions from 5 models
    consensus_pred = np.mean(test_predictions_list, axis=0)
    true_vals = test_dataset.y.flatten()
    
    consensus_rmse = np.sqrt(mean_squared_error(true_vals, consensus_pred))
    consensus_r2 = r2_score(true_vals, consensus_pred)
    
    print(f"\n  Outer Seed {outer_seed} Consensus | RMSE: {consensus_rmse:.4f} | R²: {consensus_r2:.4f}")
    
    return consensus_rmse, consensus_r2


def evaluate_dataset(name, train_path, test_path):
    """Evaluate Ulrich model on a dataset across 5 outer seeds"""
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Filter out invalid SMILES that RDKit can't parse
    def is_valid_smiles(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            return mol is not None
        except:
            return False
    
    train_valid = train_df['SMILES'].apply(is_valid_smiles)
    test_valid = test_df['SMILES'].apply(is_valid_smiles)
    
    n_invalid_train = (~train_valid).sum()
    n_invalid_test = (~test_valid).sum()
    
    if n_invalid_train > 0:
        print(f"Filtered {n_invalid_train} invalid SMILES from train set")
        train_df = train_df[train_valid].reset_index(drop=True)
    
    if n_invalid_test > 0:
        print(f"Filtered {n_invalid_test} invalid SMILES from test set")
        test_df = test_df[test_valid].reset_index(drop=True)
    
    featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
    loader = dc.data.CSVLoader(tasks=["LogS"], feature_field="SMILES", featurizer=featurizer)
    
    # Create datasets
    train_df.to_csv("_tmp_train.csv", index=False)
    full_train_dataset = loader.featurize("_tmp_train.csv")
    os.remove("_tmp_train.csv")
    
    test_df.to_csv("_tmp_test.csv", index=False)
    test_dataset = loader.featurize("_tmp_test.csv")
    os.remove("_tmp_test.csv")
    
    test_var = np.var(test_df["LogS"])
    print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
    print(f"Test Variance: {test_var:.4f}")
    print(f"Training 25 models (5 outer seeds × 5 inner models)")
    
    results = []
    times = []
    
    for outer_seed in OUTER_SEEDS:
        start_time = time.time()
        rmse, r2 = train_one_outer_seed(outer_seed, full_train_dataset, test_dataset, featurizer)
        elapsed = time.time() - start_time
        
        results.append((rmse, r2))
        times.append(elapsed)
    
    rmses = [r[0] for r in results]
    r2s = [r[1] for r in results]
    
    return {
        "dataset": name,
        "rmse_mean": np.mean(rmses),
        "rmse_std": np.std(rmses),
        "r2_mean": np.mean(r2s),
        "r2_std": np.std(r2s),
        "time_per_outer_seed": np.mean(times)
    }


def main():
    print("=" * 60)
    print("Ulrich et al. (2024) - 5 Seed Consensus Evaluation")
    print("5 outer seeds × 5 inner models = 25 models per dataset")
    print("=" * 60)

    # Check GPU availability with detailed info
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✓ GPU AVAILABLE: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu.name}")
        print(f"  TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"  TensorFlow GPU support: {tf.test.is_gpu_available()}")
    else:
        print("\n✗ NO GPU DETECTED - Running on CPU")
        print("  This will be very slow. Check your CUDA installation.")
    
    all_results = []
    
    for name, (train_path, test_path) in DATASETS.items():
        result = evaluate_dataset(name, train_path, test_path)
        all_results.append(result)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Dataset':12s} | {'RMSE':16s} | {'R²':16s} | Time/Seed")
    print("-" * 60)
    
    for r in all_results:
        print(f"{r['dataset']:12s} | {r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}  | "
              f"{r['r2_mean']:.4f} ± {r['r2_std']:.4f}  | {r['time_per_outer_seed']:.0f}s")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
