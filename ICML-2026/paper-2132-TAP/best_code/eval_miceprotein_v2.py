"""Evaluate MiceProtein downstream metrics for TAP synthetic data.
Matches paper protocol: n_real=20, nsyn=500, 5-fold CV, 6 classifiers.
"""
import argparse, numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
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

# Use exact same sampling logic as run_tap.py
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
    parser.add_argument('--synthetic_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--n_real', type=int, default=20)
    args = parser.parse_args()

    from tabcamel.data.dataset import TabularDataset
    dataset = TabularDataset(dataset_name='MiceProtein', task_type='classification')
    full_df = dataset.data_df.copy()
    tc = dataset.target_col

    syn_df = pd.read_csv(args.synthetic_path)
    if len(syn_df) > 500:
        syn_df = syn_df.sample(n=500, random_state=args.seed)

    np.random.seed(args.seed)
    real_df = sample_real_rows(full_df, tc, args.n_real, args.seed)
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
        'LR': LogisticRegression(max_iter=1000, random_state=args.seed),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=args.seed),
        'RF': RandomForestClassifier(n_estimators=100, random_state=args.seed),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=args.seed, verbose=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=args.seed, verbosity=0, eval_metric='mlogloss'),
    }

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
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

    mean_acc = np.mean(all_acc)
    std_acc = np.std(all_acc)
    mean_f1 = np.mean(all_f1)
    std_f1 = np.std(all_f1)
    print(f'Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}')
    print(f'Macro-F1: {mean_f1:.4f} +/- {std_f1:.4f}')
    # JSON output for parsing
    import json
    print('JSON:', json.dumps({'Accuracy': round(float(mean_acc), 4), 'Macro-F1': round(float(mean_f1), 4)}))

if __name__ == '__main__':
    main()
