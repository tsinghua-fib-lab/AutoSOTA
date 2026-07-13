"""
Tanimoto Similarity Analysis for Regime-I and Regime-II Datasets
Calculates similarity between solute and solvent molecules
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


def analyze_regime_i_dataset(file_path, dataset_name):
    """
    Analyze regime-I dataset (solute in water)
    """
    print(f"\nAnalyzing Regime-I: {dataset_name}")
    print("=" * 80)

    df = pd.read_csv(file_path)
    water_smiles = "O"  # Water

    similarities = []
    for smiles in df['SMILES']:
        sim = calculate_tanimoto_similarity(smiles, water_smiles)
        similarities.append(sim)

    df['tanimoto_similarity'] = similarities

    # Remove NaN values for statistics
    valid_similarities = [s for s in similarities if not np.isnan(s)]

    if len(valid_similarities) > 0:
        print(f"Dataset: {dataset_name}")
        print(f"Total samples: {len(df)}")
        print(f"Valid similarities: {len(valid_similarities)}")
        print(f"Mean Tanimoto similarity: {np.mean(valid_similarities):.4f}")
        print(f"Median Tanimoto similarity: {np.median(valid_similarities):.4f}")
        print(f"Std Tanimoto similarity: {np.std(valid_similarities):.4f}")
        print(f"Min Tanimoto similarity: {np.min(valid_similarities):.4f}")
        print(f"Max Tanimoto similarity: {np.max(valid_similarities):.4f}")

        return {
            'regime': 'I',
            'dataset': dataset_name,
            'total_samples': len(df),
            'valid_samples': len(valid_similarities),
            'mean_similarity': np.mean(valid_similarities),
            'median_similarity': np.median(valid_similarities),
            'std_similarity': np.std(valid_similarities),
            'min_similarity': np.min(valid_similarities),
            'max_similarity': np.max(valid_similarities)
        }
    else:
        print(f"No valid similarities calculated for {dataset_name}")
        return None


def analyze_regime_ii_dataset(file_path, dataset_name):
    """
    Analyze regime-II dataset (solute in various solvents)
    """
    print(f"\nAnalyzing Regime-II: {dataset_name}")
    print("=" * 80)

    df = pd.read_csv(file_path)

    similarities = []
    for idx, row in df.iterrows():
        solute = row['Solute']
        solvent = row['Solvent']
        sim = calculate_tanimoto_similarity(solute, solvent)
        similarities.append(sim)

    df['tanimoto_similarity'] = similarities

    # Remove NaN values for statistics
    valid_similarities = [s for s in similarities if not np.isnan(s)]

    if len(valid_similarities) > 0:
        print(f"Dataset: {dataset_name}")
        print(f"Total samples: {len(df)}")
        print(f"Valid similarities: {len(valid_similarities)}")
        print(f"Mean Tanimoto similarity: {np.mean(valid_similarities):.4f}")
        print(f"Median Tanimoto similarity: {np.median(valid_similarities):.4f}")
        print(f"Std Tanimoto similarity: {np.std(valid_similarities):.4f}")
        print(f"Min Tanimoto similarity: {np.min(valid_similarities):.4f}")
        print(f"Max Tanimoto similarity: {np.max(valid_similarities):.4f}")

        # Additional analysis for unique solute-solvent pairs
        unique_pairs = df[['Solute', 'Solvent']].drop_duplicates()
        print(f"Unique solute-solvent pairs: {len(unique_pairs)}")

        return {
            'regime': 'II',
            'dataset': dataset_name,
            'total_samples': len(df),
            'valid_samples': len(valid_similarities),
            'unique_pairs': len(unique_pairs),
            'mean_similarity': np.mean(valid_similarities),
            'median_similarity': np.median(valid_similarities),
            'std_similarity': np.std(valid_similarities),
            'min_similarity': np.min(valid_similarities),
            'max_similarity': np.max(valid_similarities)
        }
    else:
        print(f"No valid similarities calculated for {dataset_name}")
        return None


def main():
    results = []

    # Regime-I datasets
    print("\n" + "=" * 80)
    print("REGIME-I ANALYSIS (Aqueous Solubility - Solute vs Water)")
    print("=" * 80)

    regime_i_datasets = {
        'aqsoldb_train': 'baselines/regime-i/all_datasets/aqsoldb/train.csv',
        'aqsoldb_test': 'baselines/regime-i/all_datasets/aqsoldb/test.csv',
        'esol_train': 'baselines/regime-i/all_datasets/esol/train.csv',
        'esol_test': 'baselines/regime-i/all_datasets/esol/test.csv',
        'sc2_train': 'baselines/regime-i/all_datasets/sc2/train.csv',
        'sc2_test': 'baselines/regime-i/all_datasets/sc2/test.csv',
    }

    for name, path in regime_i_datasets.items():
        file_path = Path(path)
        if file_path.exists():
            result = analyze_regime_i_dataset(file_path, name)
            if result:
                results.append(result)

    # Regime-II datasets
    print("\n" + "=" * 80)
    print("REGIME-II ANALYSIS (Non-Aqueous Solubility - Solute vs Solvent)")
    print("=" * 80)

    regime_ii_datasets = {
        'bigsol1.0_train': 'baselines/regime-ii/all_datasets/bigsol1.0/train.csv',
        'bigsol1.0_test': 'baselines/regime-ii/all_datasets/bigsol1.0/test.csv',
        'bigsol2.0_train': 'baselines/regime-ii/all_datasets/bigsol2.0/train.csv',
        'bigsol2.0_test': 'baselines/regime-ii/all_datasets/bigsol2.0/test.csv',
        'leeds_train': 'baselines/regime-ii/all_datasets/leeds/train.csv',
        'leeds_test': 'baselines/regime-ii/all_datasets/leeds/test.csv',
    }

    for name, path in regime_ii_datasets.items():
        file_path = Path(path)
        if file_path.exists():
            result = analyze_regime_ii_dataset(file_path, name)
            if result:
                results.append(result)

    # Save summary results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    output_path = 'baselines/tanimoto_similarity_summary.csv'
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
            print(f"\nRegime {regime}:")
            print(f"  Average mean similarity: {regime_data['mean_similarity'].mean():.4f}")
            print(f"  Average median similarity: {regime_data['median_similarity'].mean():.4f}")
            print(f"  Overall min similarity: {regime_data['min_similarity'].min():.4f}")
            print(f"  Overall max similarity: {regime_data['max_similarity'].max():.4f}")


if __name__ == "__main__":
    main()
