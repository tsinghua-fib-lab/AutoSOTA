"""Evaluate downstream classifier performance for TAP synthetic data.

Matches the paper protocol:
- n_real=20, nsyn=500
- 5-fold stratified CV on held-out data (random_state=42)
- 6 classifiers: LR, KNN, MLP, RF, LightGBM, XGBoost
- Metrics: Accuracy, Macro-F1
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=random_state, verbose=-1, n_jobs=1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=random_state, verbosity=0, n_jobs=1, eval_metric='mlogloss'),
    }


def preprocess_data(df, target_col):
    """Preprocess: encode categoricals, standardize."""
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Encode target if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_col)

    # Encode categorical features
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.fillna(0)
    return X, y


def evaluate_fold(train_X, train_y, test_X, test_y, classifiers, scaler=None):
    """Train all classifiers and evaluate."""
    if scaler is not None:
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

    results = {}
    for name, clf in classifiers.items():
        clf.fit(train_X, train_y)
        preds = clf.predict(test_X)
        results[name] = {
            'accuracy': accuracy_score(test_y, preds),
            'macro_f1': f1_score(test_y, preds, average='macro'),
        }
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MiceProtein')
    parser.add_argument('--synthetic_path', type=str, required=True)
    parser.add_argument('--n_real', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='runs')
    return parser.parse_args()


def sample_real_rows(df, target_col, n_rows, seed):
    """Sample n_rows stratified by target, matching run_tap.py logic."""
    if n_rows >= len(df):
        return df.reset_index(drop=True)

    try:
        sampled = (
            df.groupby(target_col, group_keys=False)
            .sample(frac=n_rows / len(df), random_state=seed)
            .sample(n=n_rows, replace=False, random_state=seed)
            .reset_index(drop=True)
        )
        return sampled
    except ValueError:
        pass

    return df.sample(n=n_rows, random_state=seed).reset_index(drop=True)


def main():
    args = parse_args()

    # Load full dataset
    from tabcamel.data.dataset import TabularDataset
    dataset = TabularDataset(dataset_name=args.dataset, task_type='classification')
    full_df = dataset.data_df.copy()
    target_col = dataset.target_col
    print(f"Full dataset: {full_df.shape}, target={target_col}")

    # Load synthetic data
    syn_df = pd.read_csv(args.synthetic_path)
    print(f"Synthetic data: {syn_df.shape}")

    # Sample n_real rows (same as TAP training used, seed=42)
    np.random.seed(args.seed)
    real_train_df = sample_real_rows(full_df, target_col, args.n_real, args.seed)
    print(f"Real training rows: {len(real_train_df)}")
    print(f"Real train class distribution:\n{real_train_df[target_col].value_counts()}")

    # The remaining rows are used for 5-fold CV testing
    # Remove the real_train_df from full_df to get the held-out set
    real_train_indices = set(real_train_df.index)
    held_out_df = full_df.drop(index=real_train_indices).reset_index(drop=True)
    print(f"Held-out for CV: {held_out_df.shape}")

    # Combine real + synthetic for training
    train_df = pd.concat([real_train_df, syn_df], ignore_index=True)
    print(f"Training set (real + syn): {train_df.shape}")

    # Preprocess
    train_X, train_y = preprocess_data(train_df, target_col)
    test_X_all, test_y_all = preprocess_data(held_out_df, target_col)

    # Ensure consistent features
    common_cols = list(set(train_X.columns) & set(test_X_all.columns))
    train_X = train_X[common_cols]
    test_X_all = test_X_all[common_cols]

    print(f"Features: {len(common_cols)}")

    # 5-fold stratified CV on held-out data
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    classifiers = get_classifiers(random_state=args.seed)

    all_fold_acc = []
    all_fold_f1 = []

    for fold_idx, (_, test_idx) in enumerate(skf.split(test_X_all, test_y_all)):
        test_X = test_X_all.iloc[test_idx]
        test_y = test_y_all.iloc[test_idx]

        scaler = StandardScaler()
        fold_results = evaluate_fold(train_X, train_y, test_X, test_y, classifiers, scaler=scaler)

        fold_acc = np.mean([r['accuracy'] for r in fold_results.values()])
        fold_f1 = np.mean([r['macro_f1'] for r in fold_results.values()])

        all_fold_acc.append(fold_acc)
        all_fold_f1.append(fold_f1)

        print(f"Fold {fold_idx+1}: Accuracy={fold_acc:.4f}, Macro-F1={fold_f1:.4f}")
        for name, r in fold_results.items():
            print(f"  {name}: Acc={r['accuracy']:.4f}, F1={r['macro_f1']:.4f}")

    mean_acc = np.mean(all_fold_acc)
    std_acc = np.std(all_fold_acc)
    mean_f1 = np.mean(all_fold_f1)
    std_f1 = np.std(all_fold_f1)

    print(f"\n=== FINAL RESULTS ===")
    print(f"Accuracy:  {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"Macro-F1:  {mean_f1:.4f} +/- {std_f1:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame({
        'fold': list(range(1, args.n_splits + 1)),
        'accuracy': all_fold_acc,
        'macro_f1': all_fold_f1,
    })
    results_df.loc['mean'] = ['mean', mean_acc, mean_f1]
    results_df.loc['std'] = ['std', std_acc, std_f1]
    results_path = output_dir / 'eval_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
