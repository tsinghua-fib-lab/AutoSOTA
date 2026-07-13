"""
Tanimoto Similarity Analysis for Train-Test Generalization
Checks similarity between train and test sets to assess generalization difficulty
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def calculate_tanimoto_similarity(smiles1, smiles2):
    """
    Calculate Tanimoto similarity between two SMILES strings using Morgan fingerprints
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return np.nan

        # Generate Morgan fingerprints (radius=2, 2048 bits)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

        # Calculate Tanimoto similarity
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return similarity
    except:
        return np.nan


def precompute_fingerprints(smiles_list):
    """
    Pre-compute fingerprints for a list of SMILES
    """
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
    """
    Calculate Tanimoto similarity between one test fingerprint and multiple train fingerprints
    """
    similarities = []
    for train_fp in train_fps:
        sim = DataStructs.TanimotoSimilarity(test_fp, train_fp)
        similarities.append(sim)
    return similarities


def analyze_regime_i_generalization(train_path, test_path, dataset_name):
    """
    Analyze regime-I: Compare solutes in test vs solutes in train
    For each test solute, find max similarity to any training solute
    """
    print(f"\nAnalyzing Regime-I Generalization: {dataset_name}")
    print("=" * 80)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_solutes = train_df['SMILES'].tolist()
    test_solutes = test_df['SMILES'].tolist()

    print(f"Train solutes: {len(train_solutes)}")
    print(f"Test solutes: {len(test_solutes)}")

    # Pre-compute fingerprints
    print("Pre-computing train fingerprints...")
    train_fps, train_valid_idx = precompute_fingerprints(train_solutes)
    print(f"Valid train fingerprints: {len(train_fps)}")

    print("Pre-computing test fingerprints...")
    test_fps, test_valid_idx = precompute_fingerprints(test_solutes)
    print(f"Valid test fingerprints: {len(test_fps)}")

    # For each test solute, find maximum similarity to any training solute
    max_similarities = []
    mean_similarities = []

    for i, test_fp in enumerate(test_fps):
        if i % 50 == 0:
            print(f"Processing test molecule {i+1}/{len(test_fps)}...")

        similarities = bulk_tanimoto_similarity(test_fp, train_fps)

        if len(similarities) > 0:
            max_similarities.append(max(similarities))
            mean_similarities.append(np.mean(similarities))
        else:
            max_similarities.append(np.nan)
            mean_similarities.append(np.nan)

    # Remove NaN values
    valid_max_sims = [s for s in max_similarities if not np.isnan(s)]
    valid_mean_sims = [s for s in mean_similarities if not np.isnan(s)]

    if len(valid_max_sims) > 0:
        print(f"\n{'='*80}")
        print(f"Results for {dataset_name}:")
        print(f"{'='*80}")
        print(f"Test molecules analyzed: {len(valid_max_sims)}")
        print(f"\nMax Similarity Statistics (closest train neighbor):")
        print(f"  Mean of max similarities: {np.mean(valid_max_sims):.4f}")
        print(f"  Median of max similarities: {np.median(valid_max_sims):.4f}")
        print(f"  Std of max similarities: {np.std(valid_max_sims):.4f}")
        print(f"  Min of max similarities: {np.min(valid_max_sims):.4f}")
        print(f"  Max of max similarities: {np.max(valid_max_sims):.4f}")

        print(f"\nMean Similarity Statistics (average to all train):")
        print(f"  Mean of mean similarities: {np.mean(valid_mean_sims):.4f}")
        print(f"  Median of mean similarities: {np.median(valid_mean_sims):.4f}")

        # Count how many test molecules have high similarity to training
        very_similar = sum(1 for s in valid_max_sims if s >= 0.8)
        similar = sum(1 for s in valid_max_sims if 0.6 <= s < 0.8)
        moderate = sum(1 for s in valid_max_sims if 0.4 <= s < 0.6)
        different = sum(1 for s in valid_max_sims if s < 0.4)

        print(f"\nDistribution of test molecules by max similarity to train:")
        print(f"  Very similar (>= 0.8): {very_similar} ({100*very_similar/len(valid_max_sims):.1f}%)")
        print(f"  Similar (0.6-0.8): {similar} ({100*similar/len(valid_max_sims):.1f}%)")
        print(f"  Moderate (0.4-0.6): {moderate} ({100*moderate/len(valid_max_sims):.1f}%)")
        print(f"  Different (< 0.4): {different} ({100*different/len(valid_max_sims):.1f}%)")

        return {
            'regime': 'I',
            'dataset': dataset_name,
            'train_size': len(train_solutes),
            'test_size': len(test_solutes),
            'mean_max_similarity': np.mean(valid_max_sims),
            'median_max_similarity': np.median(valid_max_sims),
            'std_max_similarity': np.std(valid_max_sims),
            'min_max_similarity': np.min(valid_max_sims),
            'max_max_similarity': np.max(valid_max_sims),
            'mean_mean_similarity': np.mean(valid_mean_sims),
            'very_similar_pct': 100*very_similar/len(valid_max_sims),
            'similar_pct': 100*similar/len(valid_max_sims),
            'moderate_pct': 100*moderate/len(valid_max_sims),
            'different_pct': 100*different/len(valid_max_sims),
        }
    else:
        print(f"No valid similarities calculated for {dataset_name}")
        return None


def analyze_regime_ii_generalization(train_path, test_path, dataset_name):
    """
    Analyze regime-II: Compare solvents in test vs solvents in train
    For each test solvent, find max similarity to any training solvent
    """
    print(f"\nAnalyzing Regime-II Generalization: {dataset_name}")
    print("=" * 80)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_solvents = train_df['Solvent'].unique().tolist()
    test_solvents = test_df['Solvent'].unique().tolist()

    print(f"Unique train solvents: {len(train_solvents)}")
    print(f"Unique test solvents: {len(test_solvents)}")

    # Pre-compute fingerprints
    print("Pre-computing train solvent fingerprints...")
    train_fps, train_valid_idx = precompute_fingerprints(train_solvents)
    valid_train_solvents = [train_solvents[i] for i in train_valid_idx]
    print(f"Valid train fingerprints: {len(train_fps)}")

    print("Pre-computing test solvent fingerprints...")
    test_fps, test_valid_idx = precompute_fingerprints(test_solvents)
    valid_test_solvents = [test_solvents[i] for i in test_valid_idx]
    print(f"Valid test fingerprints: {len(test_fps)}")

    # For each test solvent, find maximum similarity to any training solvent
    max_similarities = []
    mean_similarities = []

    for i, test_fp in enumerate(test_fps):
        if i % 10 == 0:
            print(f"Processing test solvent {i+1}/{len(test_fps)}...")

        similarities = bulk_tanimoto_similarity(test_fp, train_fps)

        if len(similarities) > 0:
            max_similarities.append(max(similarities))
            mean_similarities.append(np.mean(similarities))
        else:
            max_similarities.append(np.nan)
            mean_similarities.append(np.nan)

    # Remove NaN values
    valid_max_sims = [s for s in max_similarities if not np.isnan(s)]
    valid_mean_sims = [s for s in mean_similarities if not np.isnan(s)]

    if len(valid_max_sims) > 0:
        print(f"\n{'='*80}")
        print(f"Results for {dataset_name}:")
        print(f"{'='*80}")
        print(f"Test solvents analyzed: {len(valid_max_sims)}")
        print(f"\nMax Similarity Statistics (closest train solvent):")
        print(f"  Mean of max similarities: {np.mean(valid_max_sims):.4f}")
        print(f"  Median of max similarities: {np.median(valid_max_sims):.4f}")
        print(f"  Std of max similarities: {np.std(valid_max_sims):.4f}")
        print(f"  Min of max similarities: {np.min(valid_max_sims):.4f}")
        print(f"  Max of max similarities: {np.max(valid_max_sims):.4f}")

        print(f"\nMean Similarity Statistics (average to all train solvents):")
        print(f"  Mean of mean similarities: {np.mean(valid_mean_sims):.4f}")
        print(f"  Median of mean similarities: {np.median(valid_mean_sims):.4f}")

        # Count how many test solvents have high similarity to training
        exact_match = sum(1 for s in valid_max_sims if s >= 0.999)
        very_similar = sum(1 for s in valid_max_sims if 0.8 <= s < 0.999)
        similar = sum(1 for s in valid_max_sims if 0.6 <= s < 0.8)
        moderate = sum(1 for s in valid_max_sims if 0.4 <= s < 0.6)
        different = sum(1 for s in valid_max_sims if s < 0.4)

        print(f"\nDistribution of test solvents by max similarity to train:")
        print(f"  Exact match (>= 0.999): {exact_match} ({100*exact_match/len(valid_max_sims):.1f}%)")
        print(f"  Very similar (0.8-0.999): {very_similar} ({100*very_similar/len(valid_max_sims):.1f}%)")
        print(f"  Similar (0.6-0.8): {similar} ({100*similar/len(valid_max_sims):.1f}%)")
        print(f"  Moderate (0.4-0.6): {moderate} ({100*moderate/len(valid_max_sims):.1f}%)")
        print(f"  Different (< 0.4): {different} ({100*different/len(valid_max_sims):.1f}%)")

        # Check overlap
        train_set = set(valid_train_solvents)
        test_set = set(valid_test_solvents)
        overlap = train_set.intersection(test_set)
        print(f"\nSolvent overlap:")
        print(f"  Solvents in both train and test: {len(overlap)}")
        print(f"  Novel test solvents: {len(test_set - train_set)}")

        return {
            'regime': 'II',
            'dataset': dataset_name,
            'train_solvents': len(valid_train_solvents),
            'test_solvents': len(valid_test_solvents),
            'overlapping_solvents': len(overlap),
            'novel_test_solvents': len(test_set - train_set),
            'mean_max_similarity': np.mean(valid_max_sims),
            'median_max_similarity': np.median(valid_max_sims),
            'std_max_similarity': np.std(valid_max_sims),
            'min_max_similarity': np.min(valid_max_sims),
            'max_max_similarity': np.max(valid_max_sims),
            'mean_mean_similarity': np.mean(valid_mean_sims),
            'exact_match_pct': 100*exact_match/len(valid_max_sims),
            'very_similar_pct': 100*very_similar/len(valid_max_sims),
            'similar_pct': 100*similar/len(valid_max_sims),
            'moderate_pct': 100*moderate/len(valid_max_sims),
            'different_pct': 100*different/len(valid_max_sims),
        }
    else:
        print(f"No valid similarities calculated for {dataset_name}")
        return None


def main():
    results = []

    # Regime-I datasets: Solute similarity between train and test
    print("\n" + "=" * 80)
    print("REGIME-I GENERALIZATION ANALYSIS")
    print("Comparing solutes in test vs train to assess generalization")
    print("=" * 80)

    regime_i_datasets = {
        'aqsoldb': ('baselines/regime-i/all_datasets/aqsoldb/train.csv',
                    'baselines/regime-i/all_datasets/aqsoldb/test.csv'),
        'esol': ('baselines/regime-i/all_datasets/esol/train.csv',
                 'baselines/regime-i/all_datasets/esol/test.csv'),
        'sc2': ('baselines/regime-i/all_datasets/sc2/train.csv',
                'baselines/regime-i/all_datasets/sc2/test.csv'),
    }

    for name, (train_path, test_path) in regime_i_datasets.items():
        train_file = Path(train_path)
        test_file = Path(test_path)
        if train_file.exists() and test_file.exists():
            result = analyze_regime_i_generalization(train_file, test_file, name)
            if result:
                results.append(result)

    # Regime-II datasets: Solvent similarity between train and test
    print("\n" + "=" * 80)
    print("REGIME-II GENERALIZATION ANALYSIS")
    print("Comparing solvents in test vs train to assess generalization")
    print("=" * 80)

    regime_ii_datasets = {
        'bigsol1.0': ('baselines/regime-ii/all_datasets/bigsol1.0/train.csv',
                      'baselines/regime-ii/all_datasets/bigsol1.0/test.csv'),
        'bigsol2.0': ('baselines/regime-ii/all_datasets/bigsol2.0/train.csv',
                      'baselines/regime-ii/all_datasets/bigsol2.0/test.csv'),
        'leeds': ('baselines/regime-ii/all_datasets/leeds/train.csv',
                  'baselines/regime-ii/all_datasets/leeds/test.csv'),
    }

    for name, (train_path, test_path) in regime_ii_datasets.items():
        train_file = Path(train_path)
        test_file = Path(test_path)
        if train_file.exists() and test_file.exists():
            result = analyze_regime_ii_generalization(train_file, test_file, name)
            if result:
                results.append(result)

    # Save summary results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    output_path = 'baselines/tanimoto_generalization_summary.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Regime-wise summary
    print("\n" + "=" * 80)
    print("REGIME-WISE SUMMARY")
    print("=" * 80)

    for regime in ['I', 'II']:
        regime_data = results_df[results_df['regime'] == regime]
        if len(regime_data) > 0:
            if regime == 'I':
                print(f"\nRegime {regime} (Solute Generalization):")
            else:
                print(f"\nRegime {regime} (Solvent Generalization):")
            print(f"  Avg mean max similarity: {regime_data['mean_max_similarity'].mean():.4f}")
            print(f"  Avg median max similarity: {regime_data['median_max_similarity'].mean():.4f}")
            print(f"  Overall min max similarity: {regime_data['min_max_similarity'].min():.4f}")
            print(f"  Overall max max similarity: {regime_data['max_max_similarity'].max():.4f}")


if __name__ == "__main__":
    main()
