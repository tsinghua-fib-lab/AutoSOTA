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
import warnings
warnings.filterwarnings('ignore')

from tabcamel.data.dataset import TabularDataset

dataset = TabularDataset(dataset_name='MiceProtein', task_type='classification')
df = dataset.data_df.copy()
tc = dataset.target_col

def preprocess(_df):
    X = _df.drop(columns=[tc]).copy()
    y = _df[tc].copy()
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

X, y = preprocess(df)
print(f'Data: {X.shape}, Classes: {len(np.unique(y))}')

clfs = {
    'LR': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    'RF': RandomForestClassifier(n_estimators=100, random_state=42),
    'LGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    'XGB': xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric='mlogloss'),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_acc, all_f1 = [], []
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    fa, ff = [], []
    for name, clf in clfs.items():
        clf.fit(X_tr_s, y_tr)
        p = clf.predict(X_te_s)
        fa.append(accuracy_score(y_te, p))
        ff.append(f1_score(y_te, p, average='macro'))
    all_acc.append(np.mean(fa))
    all_f1.append(np.mean(ff))
    print(f'Fold {fold+1}: Acc={all_acc[-1]:.4f}, F1={all_f1[-1]:.4f}')
print(f'UPPER BOUND: Accuracy={np.mean(all_acc):.4f}+/-{np.std(all_acc):.4f}, Macro-F1={np.mean(all_f1):.4f}+/-{np.std(all_f1):.4f}')
