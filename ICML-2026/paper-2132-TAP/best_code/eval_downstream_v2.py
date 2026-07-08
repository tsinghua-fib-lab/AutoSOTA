import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def get_classifiers(random_state=42):
    return {
        'LR': LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=1),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=1),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=random_state),
        'RF': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1),
        'LGBM': lgb.LGBMClassifier(n_estimators=100, random_state=random_state, verbose=-1, n_jobs=1),
        'XGB': xgb.XGBClassifier(n_estimators=100, random_state=random_state, verbosity=0, n_jobs=1, eval_metric='mlogloss'),
    }

def preprocess(df, target_col):
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    if y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_col)
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    return X, y

def sample_real_rows(df, target_col, n_rows, seed):
    if n_rows >= len(df):
        return df.reset_index(drop=True)
    try:
        return (
            df.groupby(target_col, group_keys=False)
            .sample(frac=n_rows / len(df), random_state=seed)
            .sample(n=n_rows, replace=False, random_state=seed)
            .reset_index(drop=True)
        )
    except ValueError:
        pass
    return df.sample(n=n_rows, random_state=seed).reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MiceProtein')
    parser.add_argument('--synthetic_path', type=str, required=True)
    parser.add_argument('--n_real', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_splits', type=int, default=5)
    args = parser.parse_args()

    from tabcamel.data.dataset import TabularDataset
    dataset = TabularDataset(dataset_name=args.dataset, task_type='classification')
    full_df = dataset.data_df.copy()
    target_col = dataset.target_col
    print(f'Full dataset: {full_df.shape}, target={target_col}')

    syn_df = pd.read_csv(args.synthetic_path)
    print(f'Synthetic data: {syn_df.shape}')

    np.random.seed(args.seed)
    real_train_df = sample_real_rows(full_df, target_col, args.n_real, args.seed)
    print(f'Real train: {len(real_train_df)} rows')

    real_idx = set(real_train_df.index)
    held_out = full_df.drop(index=real_idx).reset_index(drop=True)
    print(f'Held-out: {held_out.shape}')

    train_df = pd.concat([real_train_df, syn_df], ignore_index=True)
    print(f'Train (real+syn): {train_df.shape}')

    train_X, train_y = preprocess(train_df, target_col)
    test_X, test_y = preprocess(held_out, target_col)

    common = list(set(train_X.columns) & set(test_X.columns))
    train_X = train_X[common]
    test_X = test_X[common]
    print(f'Features: {len(common)}')

    clfs = get_classifiers(args.seed)

    # 5-fold CV on held-out
    print(f'\n=== 5-fold CV on held-out (no scaling) ===')
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    all_acc, all_f1 = [], []
    for fold, (_, test_idx) in enumerate(skf.split(test_X, test_y)):
        tX = test_X.iloc[test_idx]
        ty = test_y.iloc[test_idx]
        fa, ff = [], []
        for name, clf in clfs.items():
            clf.fit(train_X, train_y)
            p = clf.predict(tX)
            fa.append(accuracy_score(ty, p))
            ff.append(f1_score(ty, p, average='macro'))
        all_acc.append(np.mean(fa))
        all_f1.append(np.mean(ff))
        print(f'Fold {fold+1}: Acc={all_acc[-1]:.4f}, F1={all_f1[-1]:.4f}')
    print(f'FINAL: Accuracy={np.mean(all_acc):.4f}+/-{np.std(all_acc):.4f}, Macro-F1={np.mean(all_f1):.4f}+/-{np.std(all_f1):.4f}')

    # Alternative: 5-fold CV on FULL dataset (test includes real_train rows)
    print(f'\n=== 5-fold CV on FULL dataset (1080 rows) ===')
    all_X, all_y = preprocess(full_df, target_col)
    all_X = all_X[common]
    skf2 = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    all_acc2, all_f1_2 = [], []
    for fold, (_, test_idx) in enumerate(skf2.split(all_X, all_y)):
        tX = all_X.iloc[test_idx]
        ty = all_y.iloc[test_idx]
        fa, ff = [], []
        for name, clf in clfs.items():
            clf.fit(train_X, train_y)
            p = clf.predict(tX)
            fa.append(accuracy_score(ty, p))
            ff.append(f1_score(ty, p, average='macro'))
        all_acc2.append(np.mean(fa))
        all_f1_2.append(np.mean(ff))
        print(f'Fold {fold+1}: Acc={all_acc2[-1]:.4f}, F1={all_f1_2[-1]:.4f}')
    print(f'FINAL: Accuracy={np.mean(all_acc2):.4f}+/-{np.std(all_acc2):.4f}, Macro-F1={np.mean(all_f1_2):.4f}+/-{np.std(all_f1_2):.4f}')

if __name__ == '__main__':
    main()
