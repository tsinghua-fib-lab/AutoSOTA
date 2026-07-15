import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from tqdm import tqdm

def analyze_feature_pairs(X, y, feature_indices):
    """
    analyze the pairwise combinations of given feature list (AND Logic), compute F1 and Precision.
    
    Args:
        X: sparse matrix (N_samples, N_features)
        y: labels (N_samples,)
        feature_indices: list of feature IDs to analyze (e.g. top 50)
    
    Returns:
        DataFrame: contains 'Feat_A', 'Feat_B', 'F1', 'Precision', 'Recall'
    """
    n_feats = len(feature_indices)
    if n_feats > 200:
        print(f"Warning: number of features {n_feats} is较多，计算次数为 {n_feats*(n_feats-1)//2}，可能较慢。")

    # 1. preload data and binarize (Dense Boolean Matrix)
    # Shape: [N_samples, N_selected_feats]
    print("Extracting and binarizing feature subset...")
    # note: here assume X is CSR format, slicing is efficient
    subset_acts = X[:, feature_indices].toarray()
    subset_bool = (subset_acts > 0) 
    
    # ensure y is boolean or int
    y_true = np.array(y, dtype=int)
    
    results = []
    
    # 2. two-dimensional loop (only traverse upper triangle, avoid duplicate and self-loop)
    # use tqdm to show progress
    pbar = tqdm(total=n_feats * (n_feats - 1) // 2, desc="Pairwise Analysis")
    
    for i in range(n_feats):
        vec_a = subset_bool[:, i]
        id_a = feature_indices[i]
        
        for j in range(i + 1, n_feats):
            vec_b = subset_bool[:, j]
            id_b = feature_indices[j]
            
            # --- core logic: AND combination ---
            # only when both features are activated, predict as 1
            preds = vec_a & vec_b 
            
            # --- quick calculation of metrics ---
            # manually calculate faster than sklearn function calls (avoid function overhead)
            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            
            # only save meaningful results (e.g. F1 > 0)
            if f1 > 0:
                results.append({
                    "Feat_A": id_a,
                    "Feat_B": id_b,
                    "F1": f1,
                    "Precision": prec,
                    "Recall": rec,
                    "Support": tp + fp # number of samples activated by the combination
                })
            
            pbar.update(1)
            
    pbar.close()
    
    # 3. summarize results
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="F1", ascending=False).reset_index(drop=True)
    
    return df


def search_boosting_features(X, y, target_indices, candidate_indices=None, batch_size=2048):
    """
    asymmetric search: for each feature in target_indices, find the best combination in candidate_indices that can improve F1.
    
    Args:
        X: sparse matrix (N_samples, N_features)
        y: labels (N_samples,)
        target_indices: list of features you are interested in (List A, e.g. Top 10)
        candidate_indices: search range (List B). if None, search all features in X (65k).
        batch_size: number of candidate features to process in parallel.
        
    Returns:
        DataFrame: contains detailed metrics for each combination.
    """
    n_samples, n_total_features = X.shape
    
    # 1. determine the candidate range
    if candidate_indices is None:
        candidate_indices = np.arange(n_total_features)
    
    # 2. preprocess label mask, accelerate TP/FP calculation
    # convert y to boolean index, avoid repeated comparison in loop
    y_true = np.array(y).astype(bool)
    y_pos_indices = np.where(y_true)[0] # positive sample indices
    y_neg_indices = np.where(~y_true)[0] # negative sample indices
    
    results = []

    # 3. outer loop: traverse each target feature (your List A)
    print(f"Searching for best partners for {len(target_indices)} target features...")
    
    for t_idx in tqdm(target_indices, desc="Target Feats"):
        # get the boolean vector of the target feature (Dense)
        vec_target = (X[:, t_idx].toarray().flatten() > 0)
        
        # simple pruning: if the target feature is not activated, skip
        if vec_target.sum() == 0: 
            continue
            
        # pre-extract the activation status of the target feature in positive/negative samples
        # so we only need to do logical_and in the inner loop
        vec_target_pos = vec_target[y_pos_indices] # Shape: [N_pos]
        vec_target_neg = vec_target[y_neg_indices] # Shape: [N_neg]
        
        # 4. inner loop: traverse candidate features in batches (List B)
        # range(0, len, batch)
        for i in range(0, len(candidate_indices), batch_size):
            batch_idxs = candidate_indices[i : i + batch_size]
            
            # exclude itself (avoid self-loop)
            batch_idxs = batch_idxs[batch_idxs != t_idx]
            if len(batch_idxs) == 0: 
                continue

            # load a batch of candidate feature matrix (Dense Boolean)
            # Shape: [N_samples, Batch_Size]
            batch_acts = X[:, batch_idxs].toarray() > 0
            
            # --- vectorized calculation core ---
            
            # slice out positive/negative sample regions
            # Shape: [N_pos, Batch_Size]
            batch_pos = batch_acts[y_pos_indices]
            # Shape: [N_neg, Batch_Size]
            batch_neg = batch_acts[y_neg_indices]
            
            # calculate Intersection (AND logic)
            # use broadcasting: vec_target_pos[:, None] (N, 1) & batch_pos (N, B) -> (N, B)
            # but here vec_target_pos is already (N,), direct & can also be used, because numpy will automatically align columns
            # the sum(axis=0) will get an array of shape=(Batch_Size,)
            
            # TP: number of samples in positive samples that are both activated
            tp_batch = (batch_pos & vec_target_pos[:, None]).sum(axis=0)
            
            # FP: number of samples in negative samples that are both activated
            fp_batch = (batch_neg & vec_target_neg[:, None]).sum(axis=0)
            
            # FN: number of samples in positive samples that are (Target & Candidate) are not activated
            # FN = Total_Positives - TP
            n_total_pos = len(y_pos_indices)
            fn_batch = n_total_pos - tp_batch
            
            # --- calculate metrics (Vectorized) ---
            with np.errstate(divide='ignore', invalid='ignore'):
                precision = tp_batch / (tp_batch + fp_batch)
                recall = tp_batch / (tp_batch + fn_batch)
                f1 = 2 * (precision * recall) / (precision + recall)
                
            # clean NaN
            f1 = np.nan_to_num(f1)
            
            # 5. filter and save (e.g. only retain F1 > 0.5 or significant improvement)
            # here we simply keep the top 3 F1 in the batch, to avoid results explosion
            # or you can set a threshold
            top_k_in_batch = 3
            if np.max(f1) > 0.01: # only keep slightly effective results
                # argsort from small to large, get the last k
                best_local_indices = np.argsort(f1)[-top_k_in_batch:]
                
                for loc_idx in best_local_indices:
                    if f1[loc_idx] > 0:
                        results.append({
                            "Target_Feat": t_idx,
                            "Boost_Feat": batch_idxs[loc_idx],
                            "F1": f1[loc_idx],
                            "Precision": precision[loc_idx],
                            "Recall": recall[loc_idx]
                        })

    # 6. summarize results
    df = pd.DataFrame(results)
    if not df.empty:
        # for each target feature, keep its best booster
        df = df.sort_values(by="F1", ascending=False)
    
    return df
