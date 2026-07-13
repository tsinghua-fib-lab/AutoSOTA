# reproduce_ulrich.py
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


# ================= TAUTOMER AUGMENTATION =================
def augment_training_data(dataset_df, max_tautomers=50):
    """Paper-faithful augmentation with SMILES variants then tautomers"""
    enumerator = rdMolStandardize.TautomerEnumerator()
    augmented_smiles = []
    augmented_values = []

    print(f"Augmenting {len(dataset_df)} molecules (paper-faithful method)...")

    for idx, row in dataset_df.iterrows():
        if idx % 500 == 0:
            print(f"  Processed {idx}/{len(dataset_df)} molecules...")
            
        smi = row["SMILES"]
        val = row["S"]

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
        
        # Random SMILES (approximates "universal" SMILES from OpenBabel)
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

    aug_df = pd.DataFrame({"SMILES": augmented_smiles, "S": augmented_values})
    
    print(f"Before deduplication: {len(aug_df)}")
    aug_df = aug_df.drop_duplicates(subset=["SMILES"])
    print(f"After deduplication: {len(aug_df)}")
    print(f"Augmentation factor: {len(aug_df)/len(dataset_df):.2f}x (paper: 6.6-6.8x)")
    
    return aug_df


def dataset_to_df(dc_dataset):
    return pd.DataFrame({
        "SMILES": dc_dataset.ids,
        "S": dc_dataset.y.flatten()
    })


def df_to_dataset(df, featurizer, tag):
    loader = dc.data.CSVLoader(
        tasks=["S"],
        feature_field="SMILES",
        featurizer=featurizer
    )
    tmp_csv = f"_tmp_augmented_{tag}.csv"
    df.to_csv(tmp_csv, index=False)
    return loader.featurize(tmp_csv)


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


def reshape_y_pred(y_true, y_pred):
    return np.vstack(y_pred)[:len(y_true)]


def evaluate_model(model, dataset, batch_size):
    """Evaluate model and return RMSE, R2, predictions"""
    pred = model.predict_on_generator(data_generator(dataset, batch_size))
    pred = reshape_y_pred(dataset.y, pred).flatten()
    true = dataset.y.flatten()
    
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    
    return rmse, r2, pred


# ================= HYPERPARAMETERS (FROM PAPER) =================
DATASET_CSV = "../data/AqSolDB_curated.csv"
TASKS = ["S"]

batch_size = 50
num_epochs = 130
neuronslayer1 = 64
neuronslayer2 = 128
dropout = 0.1
learning_rate = 3e-4

N_SPLITS = 5

# ================= LOAD DATA =================
# 1. Define your file paths
TRAIN_CSV = "../data/train.csv"
TEST_CSV = "../data/test.csv"

# 2. Setup the featurizer (Same as before)
featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
loader = dc.data.CSVLoader(
    tasks=TASKS,           # Make sure your CSVs have the column "S"
    feature_field="SMILES",
    featurizer=featurizer
)

print(f"Loading training data from {TRAIN_CSV}...")
train_val_set = loader.featurize(TRAIN_CSV)

print(f"Loading test data from {TEST_CSV}...")
test_set = loader.featurize(TEST_CSV)

# Note: We removed the outer splitter.train_test_split because 
# you provided explicit files.

print(f"Total Train Pool: {len(train_val_set)}")
print(f"Fixed Test Set:   {len(test_set)}")

# ================= TRAIN 5 MODELS WITH VALIDATION =================
splitter = dc.splits.RandomSplitter()
test_predictions_list = []
val_results = []

for i in range(N_SPLITS):
    print(f"\n{'='*70}")
    print(f"MODEL {i+1}/{N_SPLITS}")
    print(f"{'='*70}")

    # Split into train (70% of total) and validation (20% of total)
    train_split, val_split = splitter.train_test_split(
        train_val_set, frac_train=0.777, seed=i * 100
    )
    
    # FIX: Use train_val_set as the reference for total size
    total_pool_size = len(train_val_set)
    print(f"Train set: {len(train_split)} ({len(train_split)/total_pool_size*100:.1f}% of train pool)")
    print(f"Val set:   {len(val_split)} ({len(val_split)/total_pool_size*100:.1f}% of train pool)")
    print(f"Test set:  {len(test_set)} (Fixed external set)")
    
    # Augment training data
    train_df = dataset_to_df(train_split)
    aug_train_df = augment_training_data(train_df, max_tautomers=50)
    aug_train_dataset = df_to_dataset(
        aug_train_df, featurizer, tag=f"split_{i+1}"
    )

    print(f"Training size after augmentation: {len(aug_train_dataset)}")

    # Create model
    model = KerasModel(
        MyGraphConvModel(
            batch_size=batch_size,
            neuronslayer1=neuronslayer1,
            neuronslayer2=neuronslayer2,
            dropout=dropout
        ),
        loss=dc.models.losses.L1Loss(),
        learning_rate=learning_rate,
        model_dir=f"./models/consensus_split_{i+1}"
    )

    # Training with validation monitoring
    best_val_rmse = float('inf')
    patience_counter = 0
    patience = 20  # Early stopping patience
    
    for epoch in range(num_epochs):
        # Train one epoch
        model.fit_generator(
            data_generator(
                aug_train_dataset,
                batch_size=batch_size,
                epochs=1
            )
        )
        
        # Monitor validation every 10 epochs
        if (epoch + 1):
            train_rmse, train_r2, _ = evaluate_model(model, aug_train_dataset, batch_size)
            val_rmse, val_r2, _ = evaluate_model(model, val_split, batch_size)
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, Val R²: {val_r2:.4f}")
            
            # Track best validation
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping (optional - paper trained full 130 epochs)
            # if patience_counter >= patience:
            #     print(f"Early stopping at epoch {epoch+1}")
            #     break

    # Final evaluation
    print(f"\n--- Model {i+1} Final Evaluation ---")
    
    val_rmse, val_r2, val_pred = evaluate_model(model, val_split, batch_size)
    print(f"Validation: RMSE={val_rmse:.4f}, R²={val_r2:.4f}")
    
    test_rmse, test_r2, test_pred = evaluate_model(model, test_set, batch_size)
    print(f"Test:       RMSE={test_rmse:.4f}, R²={test_r2:.4f}")

    test_predictions_list.append(test_pred)
    val_results.append({
        'rmse': val_rmse,
        'r2': val_r2,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    })

# ================= CONSENSUS RESULTS =================
print(f"\n{'='*70}")
print("CONSENSUS RESULTS")
print(f"{'='*70}")

consensus_pred = np.mean(test_predictions_list, axis=0)
true_vals = test_set.y.flatten()

consensus_rmse = np.sqrt(mean_squared_error(true_vals, consensus_pred))
consensus_r2 = r2_score(true_vals, consensus_pred)

# Calculate q² (predictive R²)
ss_res = np.sum((true_vals - consensus_pred) ** 2)
ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
q2 = 1 - (ss_res / ss_tot)

print(f"\nConsensus Test Set:")
print(f"  RMSE: {consensus_rmse:.4f} (paper: 0.657)")
print(f"  R²:   {consensus_r2:.4f} (paper: 0.901)")
print(f"  q²:   {q2:.4f} (paper: 0.896)")

print(f"\nIndividual Models Summary:")
val_rmses = [r['rmse'] for r in val_results]
test_rmses = [r['test_rmse'] for r in val_results]

print(f"  Val RMSE:  {np.mean(val_rmses):.4f} ± {np.std(val_rmses):.4f} (paper: 0.807 ± 0.040)")
print(f"  Test RMSE: {np.mean(test_rmses):.4f} ± {np.std(test_rmses):.4f} (paper: 0.722 ± 0.015)")

print(f"\nIndividual Model Results:")
for i, res in enumerate(val_results):
    print(f"  Model {i+1}: Val={res['rmse']:.4f}, Test={res['test_rmse']:.4f}")
