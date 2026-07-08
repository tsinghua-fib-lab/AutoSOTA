#!/bin/bash
# Reproduction script for TAP paper (ID 2132)
# Reproduces Accuracy and Macro-F1 on MiceProtein dataset
# Usage: bash reproduce.sh

set -e

SCRIPT_DIR=""
cd ""

# Environment setup for TabPFN (direct HF, no SOCKS proxy, no HF mirror for gated models)
unset ALL_PROXY all_proxy HF_ENDPOINT 2>/dev/null || true
export HF_TOKEN=""

echo "=== TAP Paper Reproduction (ID 2132) ==="
echo "Dataset: MiceProtein, n_real=20, nsyn=500"
echo "Backbone: TabDiff (full-dataset pre-training)"
echo ""

# Step 1: Train TabDiff on full MiceProtein dataset (skip if already trained)
MODEL_DIR="runs_repro/MiceProtein_n20/model"
if [ -f "/model.pt" ]; then
    echo "[1/3] TabDiff model already exists at , skipping training."
else
    echo "[1/3] Training TabDiff on full MiceProtein dataset (8000 steps)..."
    python3 -c "
from tabcamel.data.dataset import TabularDataset
from generators import train_tabdiff
from utils import set_seed
set_seed(42)
dataset = TabularDataset(dataset_name='MiceProtein', task_type='classification')
full_df = dataset.data_df.copy()
generator = train_tabdiff(train_data=full_df, target_col=dataset.target_col, save_path='', steps=8000, device='cuda', seed=42, task_type='classification')
print('TabDiff trained successfully.')
"
fi

# Step 2: Run TAP policy training with full-dataset TabDiff
echo "[2/3] Running TAP policy training (200 steps)..."
if [ -f "runs_repro/MiceProtein_n20/synthetic_data.csv" ]; then
    echo "Synthetic data already exists, skipping TAP training."
else
    python3 reproduce_tap.py
fi

# Step 3: Evaluate downstream classifiers
echo "[3/3] Evaluating downstream classifier performance..."
python3 -c "
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb, xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from tabcamel.data.dataset import TabularDataset

dataset = TabularDataset(dataset_name='MiceProtein', task_type='classification')
full_df = dataset.data_df.copy()
tc = dataset.target_col

syn_df = pd.read_csv('runs_repro/MiceProtein_n20/synthetic_data.csv')
print(f'Loaded {len(syn_df)} synthetic rows')

np.random.seed(42)
def sample_rows(df, tcol, n, seed):
    try:
        return df.groupby(tcol, group_keys=False).sample(frac=n/len(df), random_state=seed).sample(n=n, replace=False, random_state=seed).reset_index(drop=True)
    except: pass
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

real_df = sample_rows(full_df, tc, 20, 42)
real_idx = set(real_df.index)
held_out = full_df.drop(index=real_idx).reset_index(drop=True)

train_df = pd.concat([real_df, syn_df], ignore_index=True)

def preprocess(_df):
    X = _df.drop(columns=[tc]).copy()
    y = _df[tc].copy()
    if y.dtype == 'object':
        y = pd.Series(LabelEncoder().fit_transform(y), name=tc)
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    return X, y

trX, trY = preprocess(train_df)
teX, teY = preprocess(held_out)
common = list(set(trX.columns) & set(teX.columns))
trX, teX = trX[common], teX[common]

clfs = {
    'LR': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    'RF': RandomForestClassifier(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric='mlogloss'),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_acc, all_f1 = [], []
for fold, (_, tidx) in enumerate(skf.split(teX, teY)):
    tX, tY = teX.iloc[tidx], teY.iloc[tidx]
    fa, ff = [], []
    for name, clf in clfs.items():
        clf.fit(trX, trY)
        p = clf.predict(tX)
        fa.append(accuracy_score(tY, p))
        ff.append(f1_score(tY, p, average='macro'))
    all_acc.append(np.mean(fa))
    all_f1.append(np.mean(ff))
    print(f'Fold {fold+1}: Accuracy={all_acc[-1]:.4f}, Macro-F1={all_f1[-1]:.4f}')
print(f'RESULT: Accuracy={np.mean(all_acc):.4f}+/-{np.std(all_acc):.4f}, Macro-F1={np.mean(all_f1):.4f}+/-{np.std(all_f1):.4f}')
"
echo ""
echo "=== Reproduction Complete ==="
