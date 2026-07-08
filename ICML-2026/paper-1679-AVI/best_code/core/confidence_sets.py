
import numpy as np
import pandas as pd


def compute_rank_confidence_sets(R_matrix, m):

    better_counts = R_matrix.sum(axis=0)
    worse_counts = R_matrix.sum(axis=1)
    
    return pd.DataFrame({
        'model': np.arange(1, m + 1),
        'lower': 1 + better_counts,
        'upper': m - worse_counts
    })


def compute_topk_confidence_set(R_matrix, m, k):

    rank_confidence = compute_rank_confidence_sets(R_matrix, m)
    topk_models = sorted(set(rank_confidence.loc[rank_confidence['lower'] <= k, 'model'].values))
    
    return {
        'rank_confidence': rank_confidence,
        'topk_models': topk_models
    }

