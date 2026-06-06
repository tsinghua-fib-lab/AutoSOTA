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
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

random.seed(32)
np.random.seed(32)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

algorithm = "ERM"
model_type = "last"

# Try different C values
C_VALUES = [0.01, 0.1, 1.0, 5.0, 10.0]
margins = [0]
normalize = True
calibrated = True
feature_subsets = [
    [0, 1, 2],
    [3, 4, 5, 6],
    [7, 8, 9, 10, 11],
]
num_fold_arr = [10]


def stouffer_zscore(p_values):
    p_vals = np.clip(np.array(p_values), 1e-10, 1 - 1e-10)
    z_scores = stats.norm.ppf(1 - p_vals)
    combined_z = np.sum(z_scores) / np.sqrt(len(z_scores))
    return float(1 - stats.norm.cdf(combined_z))


def calculate_roc_auc(sf_results_df):
    scores = -sf_results_df['p_value'].values
    labels = sf_results_df['ground_truth'].values.astype(int)
    if len(np.unique(labels)) < 2:
        return None
    return roc_auc_score(labels, scores)


print("=" * 60)
print("LR C-value hyperparameter sweep (seed=0 only, OOD only)")
print("=" * 60)

# Quick test: only seed=0 for OOD to pick best C, fast
seed = 0
data_name = "fmow"

feature_cache_file = f"results/features/{data_name}_{algorithm}_{model_type}_{seed}.pkl"
with open(feature_cache_file, "rb") as f:
    full_feature_dict = pickle.load(f)

split_cache_file_ood = f"results/split_indices/{data_name}_ood.pkl"
with open(split_cache_file_ood, "rb") as f:
    ood_split_dict = pickle.load(f)

id_features_val, id_corr_val = full_feature_dict["id_val"]
id_features_test, id_corr_test = full_feature_dict["id_test"]
source_features = np.concatenate([id_features_val, id_features_test], axis=0)
source_corr = np.concatenate([id_corr_val, id_corr_test], axis=0)

for C_val in C_VALUES:
    random.seed(32)
    np.random.seed(32)

    sf_results_ood = []

    for user_split_name, user_filter in tqdm(ood_split_dict.keys(), desc=f"C={C_val}"):
        user_split_indices = ood_split_dict[(user_split_name, user_filter)]
        all_features, all_corr = full_feature_dict[user_split_name]
        user_features = all_features[user_split_indices]
        user_corr = all_corr[user_split_indices]
        user_acc = np.mean(user_corr)

        for num_folds in num_fold_arr:
            source_fold_size = len(source_corr) // num_folds
            indices = np.arange(len(source_corr))
            np.random.shuffle(indices)
            fold_indices = [indices[i*source_fold_size:(i+1)*source_fold_size] for i in range(num_folds)]

            for j, test_indices in enumerate(fold_indices):
                test_features = source_features[test_indices]
                test_corr = source_corr[test_indices]
                test_acc = np.mean(test_corr)

                reg_indices = np.concatenate([fold_indices[k] for k in range(num_folds) if k != j])
                reg_features = source_features[reg_indices]
                reg_corr = source_corr[reg_indices]

                subset_pvalues = []
                for feature_subset in feature_subsets:
                    # Custom LR with specified C
                    feat = reg_features[:, feature_subset]
                    scaler = StandardScaler()
                    feat = scaler.fit_transform(feat)

                    base_model = LogisticRegression(max_iter=1000, C=C_val)
                    clf = CalibratedClassifierCV(estimator=base_model, method="isotonic", cv=5).fit(feat, reg_corr)

                    test_feat = scaler.transform(test_features[:, feature_subset])
                    user_feat = scaler.transform(user_features[:, feature_subset])
                    test_preds = clf.predict_proba(test_feat)[:, 1]
                    user_preds = clf.predict_proba(user_feat)[:, 1]

                    from filter.tests import non_inferiority_ttest
                    p_val = non_inferiority_ttest(test_preds, user_preds, margin=margins[0])["p_value"]
                    subset_pvalues.append(p_val)

                combined_p = stouffer_zscore(subset_pvalues)
                ground_truth = user_acc >= test_acc - margins[0]

                sf_results_ood.append({
                    "seed": seed, "margin": margins[0],
                    "user_split": user_split_name, "user_filter": user_filter,
                    "num_folds": num_folds, "fold_j": j,
                    "user_acc": user_acc, "test_acc": test_acc,
                    "p_value": combined_p, "ground_truth": ground_truth,
                })

    roc_auc_ood = calculate_roc_auc(pd.DataFrame(sf_results_ood))
    print(f"C={C_val:6.3f}: Seed 0 OOD ROC AUC = {roc_auc_ood:.4f}")

print("\nDone! Compare against baseline C=1.0 -> 0.9784")
