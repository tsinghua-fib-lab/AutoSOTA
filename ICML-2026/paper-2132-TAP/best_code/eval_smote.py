import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

from tabcamel.data.dataset import TabularDataset

dataset = TabularDataset(dataset_name='MiceProtein', task_type='classification')
full_df = dataset.data_df.copy()
target_col = dataset.target_col
print(f'Full dataset: {full_df.shape}')

# Sample 20 real rows (same as TAP)
np.random.seed(42)
def sample_real_rows(df, tc, n_rows, seed):
    if n_rows >= len(df):
        return df.reset_index(drop=True)
    try:
        return df.groupby(tc, group_keys=False).sample(frac=n_rows/len(df), random_state=seed).sample(n=n_rows, replace=False, random_state=seed).reset_index(drop=True)
    except ValueError:
        pass
    return df.sample(n=n_rows, random_state=seed).reset_index(drop=True)

real_train = sample_real_rows(full_df, target_col, 20, 42)
real_idx = set(real_train.index)
held_out = full_df.drop(index=real_idx).reset_index(drop=True)
print(f'Real train: {len(real_train)}, Held-out: {len(held_out)}')

# Preprocess
def preprocess(df, tc):
    X = df.drop(columns=[tc]).copy()
    y = df[tc].copy()
    if y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=tc)
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    return X, y

train_X, train_y = preprocess(real_train, target_col)
test_X, test_y = preprocess(held_out, target_col)
common = list(set(train_X.columns) & set(test_X.columns))
train_X = train_X[common]
test_X = test_X[common]

# SMOTE: generate 500 samples
print(f'Original train: {train_X.shape}, class counts: {np.bincount(train_y)}')
try:
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=min(5, len(train_X)-1))
    syn_X, syn_y = smote.fit_resample(train_X, train_y)
    # If SMOTE generates more than 500, take 500
    if len(syn_X) > 500:
        idx = np.random.RandomState(42).choice(len(syn_X), 500, replace=False)
        syn_X = syn_X[idx]
        syn_y = syn_y[idx]
    print(f'SMOTE generated: {syn_X.shape[0]} samples (using {min(500, syn_X.shape[0])})')
except Exception as e:
    print(f'SMOTE failed: {e}')
    syn_X, syn_y = None, None

if syn_X is not None and len(syn_X) > 0:
    # Combine
    aug_X = np.vstack([train_X.values, syn_X[:500]])
    aug_y = np.concatenate([train_y.values, syn_y[:500]])
    print(f'Augmented train: {aug_X.shape[0]} rows')

    clfs = {
        'LR': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'LGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        'XGB': xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric='mlogloss'),
    }

    # 5-fold CV on held-out
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_acc, all_f1 = [], []
    for fold, (_, tidx) in enumerate(skf.split(test_X, test_y)):
        fa, ff = [], []
        for name, clf in clfs.items():
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(aug_X)
            X_te = scaler.transform(test_X.values[tidx])
            clf.fit(X_tr, aug_y)
            p = clf.predict(X_te)
            fa.append(accuracy_score(test_y.values[tidx], p))
            ff.append(f1_score(test_y.values[tidx], p, average='macro'))
        all_acc.append(np.mean(fa))
        all_f1.append(np.mean(ff))
        print(f'Fold {fold+1}: Acc={all_acc[-1]:.4f}, F1={all_f1[-1]:.4f}')
    print(f'SMOTE FINAL: Accuracy={np.mean(all_acc):.4f}+/-{np.std(all_acc):.4f}, Macro-F1={np.mean(all_f1):.4f}+/-{np.std(all_f1):.4f}')

