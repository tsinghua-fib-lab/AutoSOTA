import numpy as np
from dag.dag_ops import adj_from_W, skel_from_adj


def shd_directed(A_pred: np.ndarray, A_true: np.ndarray) -> int:
    """
    Structural Hamming Distance where an orientation flip counts as 1.
    For each unordered pair {i,j}:
      - neither edge in both -> 0
      - edge in one skeleton only -> 1
      - edge in both skeletons but directions differ -> 1
      - same direction -> 0
    """
    D = A_true.shape[0]
    shd = 0
    for i in range(D):
        for j in range(i+1, D):
            tp = (A_true[i,j], A_true[j,i])
            pp = (A_pred[i,j], A_pred[j,i])
            has_true = (tp[0] + tp[1]) > 0
            has_pred = (pp[0] + pp[1]) > 0
            if not has_true and not has_pred:
                continue
            if has_true != has_pred:
                shd += 1
            else:
                # both have an edge; check orientation
                if tp != pp:
                    shd += 1
    return shd


def metrics_single_sample(A_pred: np.ndarray, A_true: np.ndarray) -> dict:
    """
    Structural metrics for ONE predicted directed adjacency vs true:
      - skeleton precision/recall/F1
      - orientation precision/recall/F1 (exact arrow match)
      - SHD (flip=1)
    """
    D = A_true.shape[0]
    # Skeletons
    S_pred = skel_from_adj(A_pred)
    S_true = skel_from_adj(A_true)

    # Skeleton counts
    tp_skel = int(np.sum((S_pred == 1) & (S_true == 1)) // 2)  # each undirected edge counted once
    fp_skel = int(np.sum((S_pred == 1) & (S_true == 0)) // 2)
    fn_skel = int(np.sum((S_pred == 0) & (S_true == 1)) // 2)

    prec_skel = tp_skel / (tp_skel + fp_skel + 1e-9)
    rec_skel  = tp_skel / (tp_skel + fn_skel + 1e-9)
    f1_skel   = 2*prec_skel*rec_skel / (prec_skel + rec_skel + 1e-9)

    # Orientation counts (exact arrow match)
    tp_or = int(np.sum((A_pred == 1) & (A_true == 1)))
    fp_or = int(np.sum((A_pred == 1) & (A_true == 0)))
    fn_or = int(np.sum((A_pred == 0) & (A_true == 1)))

    prec_or = tp_or / (tp_or + fp_or + 1e-9)
    rec_or  = tp_or / (tp_or + fn_or + 1e-9)
    f1_or   = 2*prec_or*rec_or / (prec_or + rec_or + 1e-9)

    # SHD with flip=1
    shd = shd_directed(A_pred, A_true)

    return dict(
        skel_precision=prec_skel, skel_recall=rec_skel, skel_f1=f1_skel,
        orient_precision=prec_or, orient_recall=rec_or, orient_f1=f1_or,
        shd=shd
    )


def metrics_from_weighted_samples(particles: list, weights: np.ndarray, A_true: np.ndarray) -> dict:
    """
    Weighted (Bayesian model averaged) structural metrics:
    E_q[metric(G)] ~= sum_s w_s metric(G_s)
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / (weights.sum() + 1e-12)

    agg = dict(
        skel_precision=0.0, skel_recall=0.0, skel_f1=0.0,
        orient_precision=0.0, orient_recall=0.0, orient_f1=0.0,
        shd=0.0
    )
    for W_s, w in zip(particles, weights):
        A_s = adj_from_W(W_s)
        m = metrics_single_sample(A_s, A_true)
        for k in agg:
            agg[k] += w * m[k]
    return agg

