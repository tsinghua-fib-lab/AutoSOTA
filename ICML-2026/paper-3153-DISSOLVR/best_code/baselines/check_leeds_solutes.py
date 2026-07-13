"""
Quick analysis of Leeds dataset solutes (train vs test)
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import warnings
warnings.filterwarnings('ignore')


def precompute_fingerprints(smiles_list):
    """Pre-compute fingerprints for a list of SMILES"""
    fps = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(fp)
                valid_indices.append(i)
        except:
            pass
    return fps, valid_indices


def bulk_tanimoto_similarity(test_fp, train_fps):
    """Calculate Tanimoto similarity between one test fingerprint and multiple train fingerprints"""
    similarities = []
    for train_fp in train_fps:
        sim = DataStructs.TanimotoSimilarity(test_fp, train_fp)
        similarities.append(sim)
    return similarities


# Load Leeds data
train_df = pd.read_csv('baselines/regime-ii/all_datasets/leeds/train.csv')
test_df = pd.read_csv('baselines/regime-ii/all_datasets/leeds/test.csv')

print("LEEDS DATASET ANALYSIS")
print("=" * 80)

# Solvent analysis
train_solvents = train_df['Solvent'].unique()
test_solvents = test_df['Solvent'].unique()

print(f"\nSolvents:")
print(f"  Train: {len(train_solvents)} unique solvents")
print(f"  Test: {len(test_solvents)} unique solvents")
print(f"  Train solvents: {sorted(train_solvents)}")
print(f"  Test solvents: {sorted(test_solvents)}")
print(f"  Overlap: {set(train_solvents).intersection(set(test_solvents))}")

# Solute analysis
train_solutes = train_df['Solute'].unique()
test_solutes = test_df['Solute'].unique()

print(f"\nSolutes:")
print(f"  Train: {len(train_solutes)} unique solutes")
print(f"  Test: {len(test_solutes)} unique solutes")

solute_overlap = set(train_solutes).intersection(set(test_solutes))
print(f"  Overlapping solutes: {len(solute_overlap)}")
print(f"  Novel test solutes: {len(set(test_solutes) - set(train_solutes))}")

# Now calculate solute similarities
print("\n" + "=" * 80)
print("SOLUTE SIMILARITY ANALYSIS (Train vs Test)")
print("=" * 80)

print("Pre-computing train solute fingerprints...")
train_fps, train_valid_idx = precompute_fingerprints(train_solutes.tolist())
print(f"Valid train fingerprints: {len(train_fps)}")

print("Pre-computing test solute fingerprints...")
test_fps, test_valid_idx = precompute_fingerprints(test_solutes.tolist())
print(f"Valid test fingerprints: {len(test_fps)}")

max_similarities = []
mean_similarities = []

for i, test_fp in enumerate(test_fps):
    similarities = bulk_tanimoto_similarity(test_fp, train_fps)
    if len(similarities) > 0:
        max_similarities.append(max(similarities))
        mean_similarities.append(np.mean(similarities))

valid_max_sims = [s for s in max_similarities if not np.isnan(s)]
valid_mean_sims = [s for s in mean_similarities if not np.isnan(s)]

print(f"\nResults:")
print(f"  Test solutes analyzed: {len(valid_max_sims)}")
print(f"  Mean of max similarities: {np.mean(valid_max_sims):.4f}")
print(f"  Median of max similarities: {np.median(valid_max_sims):.4f}")
print(f"  Std of max similarities: {np.std(valid_max_sims):.4f}")
print(f"  Min of max similarities: {np.min(valid_max_sims):.4f}")
print(f"  Max of max similarities: {np.max(valid_max_sims):.4f}")

# Distribution
exact_match = sum(1 for s in valid_max_sims if s >= 0.999)
very_similar = sum(1 for s in valid_max_sims if 0.8 <= s < 0.999)
similar = sum(1 for s in valid_max_sims if 0.6 <= s < 0.8)
moderate = sum(1 for s in valid_max_sims if 0.4 <= s < 0.6)
different = sum(1 for s in valid_max_sims if s < 0.4)

print(f"\nDistribution of test solutes by max similarity to train:")
print(f"  Exact match (>= 0.999): {exact_match} ({100*exact_match/len(valid_max_sims):.1f}%)")
print(f"  Very similar (0.8-0.999): {very_similar} ({100*very_similar/len(valid_max_sims):.1f}%)")
print(f"  Similar (0.6-0.8): {similar} ({100*similar/len(valid_max_sims):.1f}%)")
print(f"  Moderate (0.4-0.6): {moderate} ({100*moderate/len(valid_max_sims):.1f}%)")
print(f"  Different (< 0.4): {different} ({100*different/len(valid_max_sims):.1f}%)")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("Leeds dataset is testing SOLUTE extrapolation (novel solutes, same solvents)")
