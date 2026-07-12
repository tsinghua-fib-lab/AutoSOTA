import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn import metrics
import scanpy as sc
import anndata as ad

def split_data_paired(X, y, train_ratio=0.8, random_state=None, shuffle=True):
    """
    Split feature data X and label data y simultaneously, ensuring consistent indices
    between training and test sets.
    
    Args:
        X: pandas DataFrame, feature data (rows are samples, columns are features)
        y: pandas DataFrame, label data (rows are samples, columns are labels), must have same index as X
        train_ratio: float, proportion of training set, default 0.8
        random_state: int, random seed for reproducibility
        shuffle: bool, whether to shuffle data before splitting, default True
    
    Returns:
        X_train, X_test: feature data for training and test sets
        y_train, y_test: label data for training and test sets
    """
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.DataFrame):
        raise TypeError("X and y must be pandas DataFrame type")
    
    if not X.index.equals(y.index):
        raise ValueError("X and y must have the same index")
    
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    
    # Shuffle data if needed
    if shuffle:
        indices = X.index.tolist()
        np.random.seed(random_state)
        np.random.shuffle(indices)
        X = X.loc[indices]
        y = y.loc[indices]
    
    # Calculate training set size
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    
    # Split data
    train_indices = X.index[:n_train]
    test_indices = X.index[n_train:]
    
    X_train = X.loc[train_indices].copy()
    X_test = X.loc[test_indices].copy()
    y_train = y.loc[train_indices].copy()
    y_test = y.loc[test_indices].copy()
    
    return X_train, X_test, y_train, y_test

# Metric computation functions
def CMD_dist(A, B):
    """Compute Correlation Matrix Distance."""
    a = np.multiply(A, B).sum()
    b = np.linalg.norm(A, 'fro') * np.linalg.norm(B, 'fro')
    return 1 - a / (b + 1e-8)

def compute_PCC(pred_data, true_data, by='protein'):
    """
    Compute Pearson Correlation Coefficient.
    
    Args:
        by: 'protein' - compute per protein, 'cell' - compute per cell
    """
    pcc_list = []
    if by == 'protein':
        for i in range(pred_data.shape[1]):
            x = np.array(pred_data[pred_data.columns[i]])
            y = np.array(true_data[true_data.columns[i]])
            pcc = pearsonr(x, y)[0]
            pcc_list.append(pcc)
    elif by == 'cell':
        for i in range(pred_data.shape[0]):
            x = np.array((pred_data.T)[pred_data.index[i]])
            y = np.array((true_data.T)[true_data.index[i]])
            pcc = pearsonr(x, y)[0]
            pcc_list.append(pcc)
    return pcc_list

def compute_CMD(pred_data, true_data, by='cell'):
    """
    Compute Correlation Matrix Distance.
    
    Args:
        by: 'cell' - compute cell-cell CMD, 'protein' - compute protein-protein CMD
    """
    if by == 'cell':
        A = pred_data.T.corr()
        B = true_data.T.corr()
    else:  # protein
        A = pred_data.corr()
        B = true_data.corr()
    
    # Remove rows and columns that are all NaN
    A.dropna(how='all', axis=1, inplace=True)
    A.dropna(how='all', inplace=True)
    B.dropna(how='all', axis=1, inplace=True)
    B.dropna(how='all', inplace=True)
    
    # Find intersection
    if by == 'cell':
        inter = A.index.intersection(B.index)
        B = B[inter]
        B = (B.T)[inter].T
        A = A[inter]
        A = (A.T)[inter].T
    else:  # protein
        inter_cols = A.columns.intersection(B.columns)
        inter_rows = A.index.intersection(B.index)
        B = B[inter_cols]
        B = (B.T)[inter_rows].T
        A = (A.T)[inter_rows].T
        A = A[inter_cols]
    
    cmd = CMD_dist(A.values.T, B.values)
    return cmd

def compute_RMSE(pred_data, true_data, method='cTp_net'):
    """
    Compute RMSE.
    
    Args:
        method: 'cTp_net' or 'Seurat' - prediction data needs exp first, then normalize, log1p, scale
                'totalVI' or others - prediction data directly normalize, log1p, scale (negative values set to 1e-20)
    """
    # Process ground truth data
    A = ad.AnnData(X=true_data).copy()
    sc.pp.normalize_total(A)
    sc.pp.log1p(A)
    sc.pp.scale(A)
    B = pd.DataFrame(data=A.X, columns=A.var_names, index=A.obs_names)
    
    # Process prediction data
    if method in ['cTp_net', 'Seurat']:
        # For cTP-net and Seurat, prediction data needs exp first
        temp = pred_data.apply(lambda x: np.exp(x), axis=0)
        temp = ad.AnnData(temp)
        sc.pp.normalize_total(temp)
        sc.pp.log1p(temp)
        sc.pp.scale(temp)
        temp_pred = pd.DataFrame(data=temp.X, columns=temp.var_names, index=temp.obs_names)
    else:
        # For totalVI and other methods, process directly (negative values set to 1e-20)
        temp = pred_data.copy()
        temp[temp < 0] = 1e-20
        temp = ad.AnnData(temp)
        sc.pp.normalize_total(temp)
        sc.pp.log1p(temp)
        sc.pp.scale(temp)
        temp_pred = pd.DataFrame(data=temp.X, columns=temp.var_names, index=temp.obs_names)
    
    # Compute RMSE
    true_array = np.array(B, dtype=np.float32).flatten()
    pred_array = np.array(temp_pred, dtype=np.float32).flatten()
    rmse = metrics.mean_squared_error(true_array, pred_array) ** 0.5
    return rmse