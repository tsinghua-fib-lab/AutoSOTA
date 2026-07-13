# ulrich_resume_456_789.py
"""
Ulrich et al. (2024) Baseline - RESUME Script

AqSolDB: Resume from seed 456, 789 (42, 101, 123 already completed)
ESOL: Run all 5 seeds (42, 101, 123, 456, 789)
SC2: Run all 5 seeds (42, 101, 123, 456, 789)

IDENTICAL to ulrich_5seeds_local.py except with resume capability for AqSolDB
"""

# ================= SILENCE LOGGING =================
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_WARNINGS"] = "0"

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# ================= IMPORTS =================
import numpy as np
import deepchem as dc
from deepchem.feat.mol_graphs import ConvMol
from deepchem.models import KerasModel
from sklearn.metrics import mean_squared_error, r2_score
from mygraphconvmodel import MyGraphConvModel

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd
import random
import time
from tqdm import tqdm


# ================= CONFIG =================
ALL_SEEDS = [42, 101, 123, 456, 789]

# Previously completed results for AqSolDB ONLY
AQSOLDB_PREVIOUS_RESULTS = {
    42: {"rmse": 1.1181, "r2": 0.7415, "time": 2188.5},
    101: {"rmse": 1.2371, "r2": 0.6836, "time": 2334.0},
    123: {"rmse": 1.2343, "r2": 0.6850, "time": 3447.1},
}
AQSOLDB_SEEDS_TO_RUN = [456, 789]  # Only remaining seeds

BATCH_SIZE = 50
NUM_EPOCHS = 130
NEURONS_LAYER1 = 64
NEURONS_LAYER2 = 128
DROPOUT = 0.1
LEARNING_RATE = 3e-4

TASKS = ["LogS"]

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

    for idx, row in tqdm(dataset_df.iterrows(), total=len(dataset_df), desc="    Augmenting"):
        smi = row["SMILES"]
        val = row["LogS"]

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # STEP 1: Generate SMILES variants
        smiles_variants = set()
        
        # Canonical
        smiles_variants.add(Chem.MolToSmiles(mol, canonical=True))
        
        # With explicit H
        mol_h = Chem.AddHs(mol)
        smiles_variants.add(Chem.MolToSmiles(mol_h, canonical=True))
        
        # Without explicit H
        mol_no_h = Chem.RemoveHs(mol)
        smiles_variants.add(Chem.MolToSmiles(mol_no_h, canonical=True))
        
        # Kekulized
        try:
            mol_kek = Chem.Mol(mol)
            Chem.Kekulize(mol_kek, clearAromaticFlags=True)
            smiles_variants.add(Chem.MolToSmiles(mol_kek, kekuleSmiles=True))
        except:
            pass
        
        # Random SMILES
        for _ in range(4):
            try:
                smiles_variants.add(Chem.MolToSmiles(mol, doRandom=True))
            except:
                pass
        
        # InChI-based
        try:
            inchi = Chem.MolToInchi(mol)
            mol_from_inchi = Chem.MolFromInchi(inchi)
            if mol_from_inchi:
                smiles_variants.add(Chem.MolToSmiles(mol_from_inchi))
        except:
            pass

        # STEP 2: For each SMILES variant, generate tautomers
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
        
        # Add all structures for this molecule
        for structure_smi in all_structures:
            augmented_smiles.append(structure_smi)
            augmented_values.append(val)

    aug_df = pd.DataFrame({"SMILES": augmented_smiles, "LogS": augmented_values})
    aug_df = aug_df.drop_duplicates(subset=["SMILES"])
    
    return aug_df


def dataset_to_df(dc_dataset):
    return pd.DataFrame({
        "SMILES": dc_dataset.ids,
        "LogS": dc_dataset.y.flatten()
    })


def df_to_dataset(df, featurizer, tag):
    loader = dc.data.CSVLoader(
        tasks=["LogS"],
        feature_field="SMILES",
        featurizer=featurizer
    )
    tmp_csv = f"_tmp_augmented_{tag}.csv"
    df.to_csv(tmp_csv, index=False)
    dataset = loader.featurize(tmp_csv)
    os.remove(tmp_csv)  # Clean up
    return dataset


# ================= DATA GENERATOR =================
def data_generator(dataset, batch_size, epochs=1):
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
    pred = model.predict_on_generator(data_generator(dataset, batch_size))
    pred = np.vstack(pred)[:len(dataset.y)].flatten()
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


def train_one_seed(seed, train_val_set, test_set, featurizer):
    """Train one model with 5-fold consensus and return test RMSE"""
    set_all_seeds(seed)
    
    print(f"\n  SEED {seed}")
    print(f"  {'='*50}")
    
    splitter = dc.splits.RandomSplitter()
    
    # Split into train/val (77/23)
    train_split, val_split = splitter.train_test_split(
        train_val_set, frac_train=0.77, seed=seed
    )
    
    print(f"    Train: {len(train_split)}, Val: {len(val_split)}")
    
    # Augment training data
    train_df = dataset_to_df(train_split)
    aug_train_df = augment_training_data(train_df, max_tautomers=50)
    aug_train_dataset = df_to_dataset(aug_train_df, featurizer, tag=f"seed_{seed}")
    
    print(f"    Augmented train: {len(aug_train_dataset)} samples")
    
    # Create model
    model = KerasModel(
        MyGraphConvModel(
            batch_size=BATCH_SIZE,
            neuronslayer1=NEURONS_LAYER1,
            neuronslayer2=NEURONS_LAYER2,
            dropout=DROPOUT
        ),
        loss=dc.models.losses.L1Loss(),
        learning_rate=LEARNING_RATE,
        model_dir=f"./_tmp_models/seed_{seed}"
    )
    
    # Train
    print(f"    Training for {NUM_EPOCHS} epochs...")
    for epoch in tqdm(range(NUM_EPOCHS), desc="    Epochs"):
        model.fit_generator(
            data_generator(aug_train_dataset, BATCH_SIZE, epochs=1)
        )
        
        if (epoch + 1) % 20 == 0:
            train_rmse, _, _ = evaluate_model(model, aug_train_dataset, BATCH_SIZE)
            val_rmse, _, _ = evaluate_model(model, val_split, BATCH_SIZE)
            print(f"    Epoch {epoch+1:3d}/{NUM_EPOCHS} | Train: {train_rmse:.3f}, Val: {val_rmse:.3f}")
    
    # Evaluate on test
    test_rmse, test_r2, _ = evaluate_model(model, test_set, BATCH_SIZE)
    
    print(f"    Test RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")
    
    # Cleanup
    import shutil
    if os.path.exists(f"./_tmp_models/seed_{seed}"):
        shutil.rmtree(f"./_tmp_models/seed_{seed}")
    
    return test_rmse, test_r2


def evaluate_dataset(name, train_path, test_path, seeds_to_run, previous_results=None):
    """Evaluate on a dataset with optional resume capability"""
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")
    
    # Load data
    featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
    loader = dc.data.CSVLoader(
        tasks=TASKS,
        feature_field="SMILES",
        featurizer=featurizer
    )
    
    train_val_set = loader.featurize(train_path)
    test_set = loader.featurize(test_path)
    
    print(f"Train: {len(train_val_set)} samples, Test: {len(test_set)} samples")
    
    # Print previous results if resuming
    if previous_results:
        print("\nPreviously completed seeds:")
        for seed, res in previous_results.items():
            print(f"  Seed {seed}: RMSE={res['rmse']:.4f}, R²={res['r2']:.4f}")
    
    # Run seeds
    new_results = {}
    
    for seed in seeds_to_run:
        start_time = time.time()
        rmse, r2 = train_one_seed(seed, train_val_set, test_set, featurizer)
        elapsed = time.time() - start_time
        new_results[seed] = {"rmse": rmse, "r2": r2, "time": elapsed}
        print(f"  Seed {seed} completed in {elapsed:.1f}s")
    
    # Combine with previous results
    if previous_results:
        all_results = {**previous_results, **new_results}
    else:
        all_results = new_results
    
    # Calculate statistics
    all_rmses = [res["rmse"] for res in all_results.values()]
    all_r2s = [res["r2"] for res in all_results.values()]
    all_times = [res["time"] for res in all_results.values()]
    
    return {
        "name": name,
        "rmse_mean": np.mean(all_rmses),
        "rmse_std": np.std(all_rmses),
        "r2_mean": np.mean(all_r2s),
        "r2_std": np.std(all_r2s),
        "avg_time": np.mean(all_times),
        "all_results": all_results
    }


def main():
    print("=" * 60)
    print("Ulrich et al. (2024) - RESUME Script")
    print("Using MyGraphConvModel (BatchNorm + LeakyReLU)")
    print("=" * 60)
    print("\nPlan:")
    print("  AqSolDB: Resume from seeds 456, 789 (42, 101, 123 done)")
    print("  ESOL: Run all 5 seeds")
    print("  SC2: Run all 5 seeds")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nGPU available: {gpus[0]}")
    else:
        print("\nNo GPU detected, using CPU")
    
    all_results = []
    
    # 1. AqSolDB - RESUME from seeds 456, 789
    train_path, test_path = DATASETS["AqSolDB"]
    result = evaluate_dataset(
        "AqSolDB", 
        train_path, 
        test_path, 
        seeds_to_run=AQSOLDB_SEEDS_TO_RUN,
        previous_results=AQSOLDB_PREVIOUS_RESULTS
    )
    all_results.append(result)
    
    # 2. ESOL - Run all 5 seeds
    train_path, test_path = DATASETS["ESOL"]
    result = evaluate_dataset(
        "ESOL", 
        train_path, 
        test_path, 
        seeds_to_run=ALL_SEEDS,
        previous_results=None
    )
    all_results.append(result)
    
    # 3. SC2 - Run all 5 seeds
    train_path, test_path = DATASETS["SC2"]
    result = evaluate_dataset(
        "SC2", 
        train_path, 
        test_path, 
        seeds_to_run=ALL_SEEDS,
        previous_results=None
    )
    all_results.append(result)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS - Ulrich et al. Baseline")
    print("=" * 60)
    
    for r in all_results:
        print(f"\n{r['name']}:")
        print(f"  RMSE: {r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}")
        print(f"  R²:   {r['r2_mean']:.4f} ± {r['r2_std']:.4f}")
        print(f"  Avg Time/Seed: {r['avg_time']:.1f}s")
        
        # Print per-seed results
        print("  Per-seed results:")
        for seed in sorted(r['all_results'].keys()):
            res = r['all_results'][seed]
            print(f"    Seed {seed}: RMSE={res['rmse']:.4f}, R²={res['r2']:.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
