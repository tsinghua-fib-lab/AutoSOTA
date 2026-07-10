"""
This script allows to reproduce the experiments on the UKBB dataset. The UKBB
dataset cannot be shared publicly. This script assumes that the user has access
to a dataframe `df_tot` containing the UKBB proteomic data and the target, here
p21001_i0, which corresponds to the Body Mass Index (BMI) phenotype.
"""

# %%
###################################
# Load UKBB proteomic data
###################################

from pathlib import Path

import numpy as np
import pandas as pd

data_dir = Path("/path/to/data/")
df_tot = pd.read_csv(data_dir / "ukbb_proteomic_data.csv")
target = "p21001_i0"


# %%
###################################
# Feature selection
###################################
# Univariate feature selection, to keep only the top 50 features.

from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from utlis import select_k_best_features_intersection

y = df_tot[target]
X = df_tot.drop(columns=[target])
y_stratif = pd.qcut(y, q=10, labels=False)

n_folds = 5
cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
n_features = 50


print("computing feature selection")
selected_features, selected_features_names = select_k_best_features_intersection(
    X, y, y_stratif, cv, k=n_features, method=f_regression
)

X_subset = X.loc[:, selected_features_names]
print(f"Number of selected features: {len(selected_features)}")

nan_vals = np.isnan(X_subset.values).sum()
total_values = X_subset.shape[0] * X_subset.shape[1]
print(
    f"Number of missing values: {nan_vals} / {total_values} = {nan_vals/total_values*100:.2f}%"
)


# %%
########################################
# Ensemble model training and evaluation
########################################
# Creates an ensemble of HistGradientBoostingRegressor models with different
# hyperparameters, trains them on 5-fold CV splits, and evaluates their
# performance. Also computes sub-model performances.

from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from utlis import BaggingVoting, get_sub_models

scores = []
ensemble_importance = []
sub_models_importance = []
sub_models_scores = []
models_list = []


max_iter = 20
rng = np.random.default_rng(0)
lr = rng.choice(np.linspace(0.01, 1, 100), 10)
depth = rng.choice(np.linspace(5, 30, 100, dtype=int), 10)
model_ens = BaggingVoting(
    estimators=[
        (
            "1",
            HistGradientBoostingRegressor(
                max_depth=depth[0],
                random_state=0,
                max_iter=max_iter,
                learning_rate=lr[0],
            ),
        ),
        (
            "2",
            HistGradientBoostingRegressor(
                max_depth=depth[1],
                random_state=1,
                max_iter=max_iter,
                learning_rate=lr[1],
            ),
        ),
        (
            "3",
            HistGradientBoostingRegressor(
                max_depth=depth[2],
                random_state=2,
                max_iter=max_iter,
                learning_rate=lr[2],
            ),
        ),
        (
            "4",
            HistGradientBoostingRegressor(
                max_depth=depth[3],
                random_state=3,
                max_iter=max_iter,
                learning_rate=lr[3],
            ),
        ),
        (
            "5",
            HistGradientBoostingRegressor(
                max_depth=depth[4],
                random_state=4,
                max_iter=max_iter,
                learning_rate=lr[4],
            ),
        ),
        (
            "6",
            HistGradientBoostingRegressor(
                max_depth=depth[5],
                random_state=5,
                max_iter=max_iter,
                learning_rate=lr[5],
            ),
        ),
        (
            "7",
            HistGradientBoostingRegressor(
                max_depth=depth[6],
                random_state=6,
                max_iter=max_iter,
                learning_rate=lr[6],
            ),
        ),
        (
            "8",
            HistGradientBoostingRegressor(
                max_depth=depth[7],
                random_state=7,
                max_iter=max_iter,
                learning_rate=lr[7],
            ),
        ),
        (
            "9",
            HistGradientBoostingRegressor(
                max_depth=depth[8],
                random_state=8,
                max_iter=max_iter,
                learning_rate=lr[8],
            ),
        ),
        (
            "10",
            HistGradientBoostingRegressor(
                max_depth=depth[9],
                random_state=9,
                max_iter=max_iter,
                learning_rate=lr[9],
            ),
        ),
    ],
    n_jobs=10,
)


X_arr = np.array(X_subset).astype(float)
y_arr = np.array(y).astype(float)
for train_index, test_index in tqdm(cv.split(X, y_stratif)):

    X_train, y_train = X_arr[train_index], y_arr[train_index]
    imputer = SimpleImputer()
    X_train = imputer.fit_transform(X_train)
    model_c = clone(model_ens)
    model_c.fit(X_train, y_train)

    X_test, y_test = X_arr[test_index], y_arr[test_index]
    X_test = imputer.transform(X_test)
    y_pred = model_c.predict(X_test)
    score_tmp = r2_score(y_pred=y_pred, y_true=y_test)
    print(score_tmp)
    scores.append(score_tmp)
    sub_models = get_sub_models(model_c)
    scores_tmp = []
    for sub_mod in sub_models:
        y_pred = sub_mod.predict(X_test)
        scores_tmp.append(r2_score(y_pred=y_pred, y_true=y_test))
    sub_models_scores.append(np.mean(scores_tmp))
    models_list.append(model_c)

print(f"R2 score: {np.mean(scores):.2f}, {np.std(scores):.4f}")
print(f"R2 score: {np.mean(sub_models_scores):.2f}, {np.std(sub_models_scores):.4f}")


# %%
###################################
# LOCO importance computation
###################################
# Compute LOCO importances for the ensemble model and its sub-models.


import pandas as pd
from sklearn.base import clone
from utlis import loco_one

n_jobs = 5
loco_output = Parallel(n_jobs=n_jobs)(
    delayed(loco_one)(
        X_arr, y_arr, train_index, test_index, models_list[fold_id], fold_id, n_jobs=1
    )
    for fold_id, (train_index, test_index) in tqdm(enumerate(cv.split(X, y_stratif)))
)
loco_df = pd.concat([item for sublist in loco_output for item in sublist], axis=0)

loco_df["feature_name"] = loco_df["feature"].apply(lambda x: selected_features_names[x])
loco_df.to_csv("path/to/save/loco_importances.csv")


# %%
################################
# SAGE importance computation
################################

from utlis import sage_one

for fold_id, (train_index, test_index) in tqdm(enumerate(cv.split(X, y_stratif))):
    print(f"fold: {fold_id}")
    if fold_id == 0:
        continue

    n_jobs = 5
    n_samples = 1024
    sage_df = sage_one(
        X_arr,
        y_arr,
        train_index,
        test_index,
        models_list[fold_id],
        fold_id,
        seed=fold_id,
        n_jobs=n_jobs,
        n_samples=n_samples,
    )
    sage_df_save = pd.concat(sage_df)
    sage_df_save["feature_name"] = sage_df_save["feature"].apply(
        lambda x: selected_features_names[x]
    )

    sage_df_save.to_csv(f"path/to/save/sage_fold_{fold_id}.csv")
