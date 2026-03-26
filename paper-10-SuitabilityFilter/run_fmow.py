import pickle
import random
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from scipy import stats

sys.path.insert(0, '/repo')
os.chdir('/repo')

from filter.suitability_filter import SuitabilityFilter
import filter.tests as ftests

# Set seeds for reproducibility
random.seed(32)
np.random.seed(32)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

algorithm = "ERM"
model_type = "last"

classifiers = ["logistic_regression"]
margins = [0]
normalize = True
calibrated = True
feature_subsets = [
    [0, 1, 2],         # confidence subset
    [3, 4, 5, 6],      # logit subset
    [7, 8, 9, 10, 11], # loss/energy subset
]
num_fold_arr = [10]


def stouffer_zscore(p_values):
    """Combine p-values using Stouffer's Z-score method (equal weights)."""
    p_vals = np.clip(np.array(p_values), 1e-10, 1 - 1e-10)
    z_scores = stats.norm.ppf(1 - p_vals)
    combined_z = np.sum(z_scores) / np.sqrt(len(z_scores))
    return float(1 - stats.norm.cdf(combined_z))


def calculate_roc_auc(sf_results_df):
    scores = -sf_results_df['p_value'].values
    labels = sf_results_df['ground_truth'].values.astype(int)
    if len(np.unique(labels)) < 2:
        print("WARNING: Only one class in ground truth!")
        return None
    return roc_auc_score(labels, scores)


print("=" * 60)
print("Running FMoW-WILDS - Cached OOD Folds (IDEA-040)")
print("OOD: pre-train classifiers once, reuse across all user_splits")
print("=" * 60)

all_id_results = []
all_ood_results = []

for seed in [0, 1, 2]:
    data_name = "fmow"

    feature_cache_file = f"results/features/{data_name}_{algorithm}_{model_type}_{seed}.pkl"
    with open(feature_cache_file, "rb") as f:
        full_feature_dict = pickle.load(f)

    # ============================
    # ID Experiments (unchanged - source changes per user_split)
    # ============================
    print(f"\n--- FMoW ID, seed={seed} ---")
    sf_results = []

    id_feature_dict = {}
    id_feature_dict["id_val"] = full_feature_dict["id_val"]
    id_feature_dict["id_test"] = full_feature_dict["id_test"]

    split_cache_file = f"results/split_indices/{data_name}_id.pkl"
    with open(split_cache_file, "rb") as f:
        id_split_dict = pickle.load(f)

    random.seed(32)
    np.random.seed(32)

    for user_split_name, user_filter in tqdm(id_split_dict.keys(), desc=f"FMoW ID seed {seed}"):
        user_split_indices = id_split_dict[(user_split_name, user_filter)]

        all_features, all_corr = id_feature_dict[user_split_name]
        user_features = all_features[user_split_indices]
        user_corr = all_corr[user_split_indices]
        user_acc = np.mean(user_corr)

        remaining_indices = np.setdiff1d(np.arange(len(all_corr)), user_split_indices)
        remaining_features = all_features[remaining_indices]
        remaining_corr = all_corr[remaining_indices]
        if user_split_name == "id_val":
            other_split_name = "id_test"
        elif user_split_name == "id_test":
            other_split_name = "id_val"
        else:
            raise ValueError(f"Invalid split name: {user_split_name}")
        additional_features, additional_corr = id_feature_dict[other_split_name]
        source_features = np.concatenate([remaining_features, additional_features], axis=0)
        source_corr = np.concatenate([remaining_corr, additional_corr], axis=0)

        for num_folds in num_fold_arr:
            source_fold_size = len(source_corr) // num_folds
            indices = np.arange(len(source_corr))
            np.random.shuffle(indices)
            fold_indices = [
                indices[i * source_fold_size : (i + 1) * source_fold_size]
                for i in range(num_folds)
            ]

            for j, test_indices in enumerate(fold_indices):
                test_features = source_features[test_indices]
                test_corr = source_corr[test_indices]
                test_acc = np.mean(test_corr)

                reg_indices = np.concatenate([fold_indices[k] for k in range(num_folds) if k != j])
                reg_features = source_features[reg_indices]
                reg_corr = source_corr[reg_indices]

                subset_pvalues = []
                for feature_subset in feature_subsets:
                    sf = SuitabilityFilter(
                        test_features, test_corr,
                        reg_features, reg_corr,
                        device,
                        normalize=normalize,
                        feature_subset=feature_subset,
                    )
                    sf.train_classifier(calibrated=calibrated, classifier=classifiers[0])
                    sf_test = sf.suitability_test(user_features=user_features, margin=margins[0])
                    subset_pvalues.append(sf_test["p_value"])

                combined_p = stouffer_zscore(subset_pvalues)
                ground_truth = user_acc >= test_acc - margins[0]

                sf_results.append({
                    "seed": seed, "margin": margins[0],
                    "user_split": user_split_name, "user_filter": user_filter,
                    "user_acc": user_acc, "test_acc": test_acc,
                    "p_value": combined_p, "ground_truth": ground_truth,
                })

    sf_evals_id = pd.DataFrame(sf_results)
    roc_auc_id = calculate_roc_auc(sf_evals_id)
    print(f"FMoW ID seed={seed}: ROC AUC = {roc_auc_id:.4f}")
    all_id_results.append(roc_auc_id)

    # ============================
    # OOD Experiments - CACHED FOLDS
    # ============================
    print(f"\n--- FMoW OOD, seed={seed} (cached folds) ---")
    sf_results_ood = []

    split_cache_file_ood = f"results/split_indices/{data_name}_ood.pkl"
    with open(split_cache_file_ood, "rb") as f:
        ood_split_dict = pickle.load(f)

    id_features_val, id_corr_val = full_feature_dict["id_val"]
    id_features_test, id_corr_test = full_feature_dict["id_test"]
    source_features = np.concatenate([id_features_val, id_features_test], axis=0)
    source_corr = np.concatenate([id_corr_val, id_corr_test], axis=0)

    # Pre-compute FIXED fold partition once for all OOD user_splits
    random.seed(32)
    np.random.seed(32)

    for num_folds in num_fold_arr:
        source_fold_size = len(source_corr) // num_folds
        fold_indices_arr = np.arange(len(source_corr))
        np.random.shuffle(fold_indices_arr)
        ood_fold_indices = [
            fold_indices_arr[i * source_fold_size : (i + 1) * source_fold_size]
            for i in range(num_folds)
        ]

        # Pre-train all classifiers for all (fold_j, feature_subset) combinations
        print(f"  Pre-training {num_folds}x{len(feature_subsets)} classifiers...")
        cached_classifiers = {}  # (j, subset_idx) -> trained SuitabilityFilter
        cached_test_acc = {}     # j -> test_acc
        cached_test_preds = {}   # (j, subset_idx) -> test_predictions

        for j, test_indices in enumerate(ood_fold_indices):
            test_features = source_features[test_indices]
            test_corr = source_corr[test_indices]
            cached_test_acc[j] = np.mean(test_corr)

            reg_indices = np.concatenate([ood_fold_indices[k] for k in range(num_folds) if k != j])
            reg_features = source_features[reg_indices]
            reg_corr = source_corr[reg_indices]

            for si, feature_subset in enumerate(feature_subsets):
                sf = SuitabilityFilter(
                    test_features, test_corr,
                    reg_features, reg_corr,
                    device,
                    normalize=normalize,
                    feature_subset=feature_subset,
                )
                sf.train_classifier(calibrated=calibrated, classifier=classifiers[0])
                cached_classifiers[(j, si)] = sf
        print(f"  Pre-training done.")

        # Now evaluate all user_splits using cached classifiers
        for user_split_name, user_filter in tqdm(ood_split_dict.keys(), desc=f"FMoW OOD seed {seed}"):
            user_split_indices = ood_split_dict[(user_split_name, user_filter)]

            all_features_ood, all_corr_ood = full_feature_dict[user_split_name]
            user_features = all_features_ood[user_split_indices]
            user_corr = all_corr_ood[user_split_indices]
            user_acc = np.mean(user_corr)

            for j in range(num_folds):
                test_acc = cached_test_acc[j]

                subset_pvalues = []
                for si, feature_subset in enumerate(feature_subsets):
                    sf = cached_classifiers[(j, si)]
                    sf_test = sf.suitability_test(user_features=user_features, margin=margins[0])
                    subset_pvalues.append(sf_test["p_value"])

                combined_p = stouffer_zscore(subset_pvalues)
                ground_truth = user_acc >= test_acc - margins[0]

                sf_results_ood.append({
                    "seed": seed, "margin": margins[0],
                    "user_split": user_split_name, "user_filter": user_filter,
                    "num_folds": num_folds, "fold_j": j,
                    "user_acc": user_acc, "test_acc": test_acc,
                    "p_value": combined_p, "ground_truth": ground_truth,
                })

    sf_evals_ood = pd.DataFrame(sf_results_ood)
    roc_auc_ood = calculate_roc_auc(sf_evals_ood)
    print(f"FMoW OOD seed={seed}: ROC AUC = {roc_auc_ood:.4f}")
    all_ood_results.append(roc_auc_ood)

print("\n" + "=" * 60)
print("FINAL RESULTS (3 seeds)")
print("=" * 60)
print(f"FMoW-WILDS ID ROC AUC per seed: {[f'{v:.4f}' for v in all_id_results]}")
print(f"FMoW-WILDS ID ROC AUC mean: {np.mean(all_id_results):.4f}")
print(f"FMoW-WILDS ID ROC AUC std: {np.std(all_id_results):.4f}")
print(f"FMoW-WILDS OOD ROC AUC per seed: {[f'{v:.4f}' for v in all_ood_results]}")
print(f"FMoW-WILDS OOD ROC AUC mean: {np.mean(all_ood_results):.4f}")
print(f"FMoW-WILDS OOD ROC AUC std: {np.std(all_ood_results):.4f}")
