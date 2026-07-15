from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from tqdm import tqdm

@dataclass
class GlobalMetricsResult:
    """container for feature evaluation results"""
    # basic information
    feature_indices: NDArray
    
    # core metrics
    precisions: NDArray
    recalls: NDArray
    f1_scores: NDArray
    activation_ratios: NDArray
    feature_diff: NDArray
    separation_score: Optional[float]

    # filtered results (store feature IDs)
    top_f1_ids: List[int]
    top_precision_ids: List[int]
    top_recall_ids: List[int]
    top_diff_ids: List[int]
    pareto_front_ids: List[int]
    
    # metadata
    stats: Dict[str, Any] = field(default_factory=dict)

# =============================== Diff / PR Metrics ===============================

def compute_diff(X: sp.csr_matrix, y: NDArray, normalize: bool = False) -> Tuple[NDArray, Optional[float]]:
    """
    compute the average activation difference of features in positive and negative samples (Vectorized)。

    if normalize:
        Diff = (Mean(Pos) - Mean(Neg)) / ((Std(Pos) + Std(Neg)) + eps)
    else:
        Diff = Mean(Pos) - Mean(Neg)
    """
    pos_mask = (y == 1)
    neg_mask = (y == 0)
    
    if not np.any(pos_mask):
        return -np.array(X.mean(axis=0)).flatten()
    if not np.any(neg_mask):
        return np.array(X.mean(axis=0)).flatten()
    
    X_pos = X[pos_mask]
    X_neg = X[neg_mask]
    
    # 2. compute the first moment (Mean, E[x])
    pos_mean = np.array(X_pos.mean(axis=0)).flatten()
    neg_mean = np.array(X_neg.mean(axis=0)).flatten()
    
    diff = pos_mean - neg_mean
    if not normalize:
        return diff, None
    
    # 3. compute the second moment (Mean of Squares, E[x^2])
    # for sparse matrix, use .multiply() to perform element-wise square, which is more robust than power(2)
    pos_sq_mean = np.sum(pos_mean**2)
    neg_sq_mean = np.sum(neg_mean**2)
    
    # 4. compute the standard deviation: Std = sqrt(E[x^2] - (E[x])^2)
    # np.clip is used to prevent the error of floating point precision from causing the square root to be negative
    pos_sq_var = np.clip(pos_sq_mean - np.average(pos_mean)**2, 0, None)
    neg_sq_var = np.clip(neg_sq_mean - np.average(neg_mean)**2, 0, None)
    pos_std = np.sqrt(pos_sq_var)
    neg_std = np.sqrt(neg_sq_var)
    
    # 5. compute the normalized difference
    eps = 1e-6
    diff = diff / (pos_std + neg_std + eps)
    
    separation_score = np.sum((pos_mean - neg_mean)**2) / (pos_sq_var + neg_sq_var + eps)

    return diff, separation_score

def _batch_pr_calc(
    X_batch: sp.csr_matrix, 
    y_pos: sp.csr_matrix, 
    y_neg: sp.csr_matrix, 
    n_pos: int
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    [internal function] batch compute Precision, Recall, Activation Count
    """
    
    X_bin = X_batch.copy()
    X_bin.data = np.ones_like(X_bin.data) # set all positions > 0 to 1
    
    # matrix multiplication to compute TP, FP
    # X_bin.T: (n_feats, n_samples) @ y: (n_samples, 1) -> (n_feats, 1)

    tp = (X_bin.T * y_pos).toarray().flatten()
    fp = (X_bin.T * y_neg).toarray().flatten()
    
    # compute metrics
    # TP + FN = n_pos -> FN = n_pos - TP
    # Recall = TP / n_pos
    # Precision = TP / (TP + FP)
    
    # avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = tp / (tp + fp)
        recall = tp / n_pos # n_pos is a constant, usually > 0
        
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    count = tp + fp
    
    return precision, recall, count

def compute_pr_stats(
    X: sp.csr_matrix, 
    y: NDArray, 
    batch_size: int = 2048
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    compute Precision, Recall and activation count for all features.
    use batch processing to control memory.
    """
    n_features = X.shape[1]
    y = y.astype(int)
    n_pos = int(np.sum(y == 1))
    
    # prebuild sparse label vector, accelerate matrix multiplication
    y_pos_sp = sp.csr_matrix(y.reshape(-1, 1))
    y_neg_sp = sp.csr_matrix((1 - y).reshape(-1, 1))
    
    p_list, r_list, c_list = [], [], []

    for start in tqdm(range(0, n_features, batch_size), desc="Computing PR"):
        end = min(start + batch_size, n_features)
        X_batch = X[:, start:end]
        p, r, c = _batch_pr_calc(X_batch, y_pos_sp, y_neg_sp, n_pos)
        p_list.append(p)
        r_list.append(r)
        c_list.append(c)
        
    return (
        np.concatenate(p_list),
        np.concatenate(r_list),
        np.concatenate(c_list)
    )


# =============================== F1 / Pareto Metrics (Base on PR) ===============================

def compute_f1(precision: NDArray, recall: NDArray) -> NDArray:
    """compute F1 based on P/R"""
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = 2 * (precision * recall) / (precision + recall)
    return np.nan_to_num(f1)

def compute_pareto_front(
    precision: NDArray, 
    recall: NDArray, 
    feature_ids: NDArray, 
    f1: Optional[NDArray] = None
) -> List[int]:
    """
    compute Pareto frontier (non-dominated solution set).
    logic: in the case of higher Recall, there is no other point with higher Precision (consider epsilon tolerance, here keep it pure).
    """
    # sort: by Precision descending, Recall descending
    # so we only need to traverse once, maintain the maximum Recall seen so far
    sort_idx = np.lexsort((-recall, -precision))
    
    sorted_ids = feature_ids[sort_idx]
    sorted_recall = recall[sort_idx]
    
    pareto_ids = []
    max_recall_so_far = -1.0
    
    for idx, rec in enumerate(sorted_recall):
        # if the current point's Recall is greater than all previous points with higher Precision
        # it means it is not dominated
        if rec > max_recall_so_far:
            pareto_ids.append(int(sorted_ids[idx]))
            max_recall_so_far = rec
            
    # if there is F1, sort by F1 and output, for easy viewing
    if f1 is not None:
        f1_map = dict(zip(feature_ids, f1))
        pareto_ids.sort(key=lambda x: f1_map.get(x, 0), reverse=True)
        
    return pareto_ids

def evaluate_features(
    X: sp.spmatrix, 
    y: NDArray, 
    top_k: int = 50,
    batch_size: int = 2048
) -> GlobalMetricsResult:
    """
    execute the complete feature evaluation process.
    
    Flow:
    1. compute Diff
    2. compute PR & Count
    3. derive F1 & Pareto
    4. summarize Top K
    """
    # 0. standardize the format
    if not sp.isspmatrix_csr(X):
        X = X.tocsr()
    n_samples, n_features = X.shape
    
    # 1. compute difference (Diff)
    print("⚡ Step 1/3: Computing Feature Differences...")
    diff_scores, separation_score = compute_diff(X, y, normalize=True)
    
    # 2. compute basic metrics (PR)
    print("⚡ Step 2/3: Computing Precision & Recall...")
    precisions, recalls, counts = compute_pr_stats(X, y, batch_size=batch_size)
    
    # 3. compute derived metrics
    print("⚡ Step 3/3: Deriving F1 & Pareto Front...")
    f1_scores = compute_f1(precisions, recalls)
    feature_ids = np.arange(n_features)
    
    pareto_ids = compute_pareto_front(precisions, recalls, feature_ids, f1=f1_scores)
    
    # 4. filter Top K
    # use argpartition to accelerate (O(N)), then sort the top k
    def get_top_k_ids(values: NDArray, k: int) -> List[int]:
        if k >= len(values):
            return np.argsort(values)[::-1].tolist()
        # get the indices of the top K (unsorted)
        unsorted_top = np.argpartition(values, -k)[-k:]
        # sort the top K
        sorted_top = unsorted_top[np.argsort(values[unsorted_top])[::-1]]
        return sorted_top.tolist()

    return GlobalMetricsResult(
        feature_indices=feature_ids,
        precisions=precisions,
        recalls=recalls,
        f1_scores=f1_scores,
        activation_ratios=counts / n_samples,
        feature_diff=diff_scores,
        top_f1_ids=get_top_k_ids(f1_scores, top_k),
        top_precision_ids=get_top_k_ids(precisions, top_k),
        top_recall_ids=get_top_k_ids(recalls, top_k),
        top_diff_ids=get_top_k_ids(diff_scores, top_k),
        pareto_front_ids=pareto_ids,
        separation_score=separation_score,
        stats={
            "n_features": n_features,
            "n_samples": n_samples,
            "n_pos": int(np.sum(y))
        }
    )