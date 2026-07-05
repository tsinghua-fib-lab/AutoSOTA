import causaldag as cd
import numpy as np
import networkx as nx


def is_dag(B):
    """Check whether B corresponds to a DAG.

    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))


def get_cpdag(B):
    assert is_dag(B)
    return cd.DAG.from_amat(B).cpdag().to_amat()[0]



def get_dag_from_pdag(B_bin_pdag):
    # There is bug for G.to_dag().to_amat() from cd package
    # i.e., the shape of B is not preserved
    # So we need to manually preserve the shape
    B_bin_dag = np.zeros_like(B_bin_pdag)
    if np.all(B_bin_pdag == 0):
        # All entries in B_pdag are zeros
        return B_bin_dag
    else:
        G = cd.PDAG.from_amat(B_bin_pdag)
        B_bin_sub_dag, nodes = G.to_dag().to_amat()
        B_bin_dag[np.ix_(nodes, nodes)] = B_bin_sub_dag
        return B_bin_dag


def compute_shd_cpdag(B_bin_true, B_bin_est):
    assert is_dag(B_bin_true)
    assert is_dag(B_bin_est)
    cpdag_true = get_cpdag(B_bin_true)
    cpdag_est = get_cpdag(B_bin_est)
    return cd.PDAG.from_amat(cpdag_true).shd(cd.PDAG.from_amat(cpdag_est))


def count_precision_recall_f1(tp, fp, fn):
    # Precision
    if tp + fp == 0:
        precision = None
    else:
        precision = float(tp) / (tp + fp)

    # Recall
    if tp + fn == 0:
        recall = None
    else:
        recall = float(tp) / (tp + fn)

    # F1 score
    if precision is None or recall is None:
        f1 = None
    elif precision == 0 or recall == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def get_skeleton(B):
    B_bin = (B != 0).astype(int)
    return ((B_bin + B_bin.T) != 0).astype(int)


def count_skeleton_accuracy(B_bin_true, B_bin_est):
    skeleton_true = get_skeleton(B_bin_true)
    skeleton_est = get_skeleton(B_bin_est)
    d = len(skeleton_true)
    skeleton_triu_true = skeleton_true[np.triu_indices(d, k=1)]
    skeleton_triu_est = skeleton_est[np.triu_indices(d, k=1)]
    pred = np.flatnonzero(skeleton_triu_est)
    cond = np.flatnonzero(skeleton_triu_true)

    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos
    false_pos = np.setdiff1d(pred, cond, assume_unique=True)
    # false neg
    false_neg = np.setdiff1d(cond, true_pos, assume_unique=True)
    # compute ratio
    nnz = len(pred)
    cond_neg_size = len(skeleton_triu_true) - len(cond)
    fdr = float(len(false_pos)) / max(nnz, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(false_pos)) / max(cond_neg_size, 1)
    try:
        f1 = len(true_pos) / (len(true_pos) + 0.5 * (len(false_pos) + len(false_neg)))
    except:
        f1 = None

    # structural hamming distance
    extra_lower = np.setdiff1d(pred, cond, assume_unique=True)
    missing_lower = np.setdiff1d(cond, pred, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower)
    return {'f1_skeleton': f1, 'precision_skeleton': 1 - fdr, 'recall_skeleton': tpr,
            'shd_skeleton': shd, 'nnz_skeleton': nnz}


def count_arrows_accuracy(B_bin_true, B_bin_est):
    dag_est = cd.DAG.from_amat(B_bin_est)
    dag_true = cd.DAG.from_amat(B_bin_true)
    cm_cpdag = dag_est.confusion_matrix(dag_true)
    tp_arrows = len(cm_cpdag['true_positive_arcs'])
    fp_arrows = len(cm_cpdag['false_positive_arcs'])
    fn_arrows = len(cm_cpdag['false_negative_arcs'])
    precision_arrows, recall_arows, f1_arrows \
        = count_precision_recall_f1(tp_arrows, fp_arrows, fn_arrows)
    return {'f1_arrows': f1_arrows, 'precision_arrows': precision_arrows, 'recall_arows': recall_arows}


def count_dag_accuracy(B_bin_true, B_bin_est):
    d = B_bin_true.shape[0]
    # linear index of nonzeros
    pred = np.flatnonzero(B_bin_est)
    cond = np.flatnonzero(B_bin_true)
    cond_reversed = np.flatnonzero(B_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    if pred_size == 0:
        fdr = None
    else:
        fdr = float(len(reverse) + len(false_pos)) / pred_size
    if len(cond) == 0:
        tpr = None
    else:
        tpr = float(len(true_pos)) / len(cond)
    if cond_neg_size == 0:
        fpr = None
    else:
        fpr = float(len(reverse) + len(false_pos)) / cond_neg_size
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_bin_est + B_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(B_bin_true + B_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    # false neg
    false_neg = np.setdiff1d(cond, true_pos, assume_unique=True)
    precision, recall, f1 = count_precision_recall_f1(tp=len(true_pos),
                                                      fp=len(reverse) + len(false_pos),
                                                      fn=len(false_neg))
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size, 
            'precision': precision, 'recall': recall, 'f1': f1}


def count_accuracy(B_bin_true, B_bin_est):
    """Compute various accuracy metrics for B_bin_est."""
    results = {}
    try:
        # Calculate performance metrics for DAG
        results_dag = count_dag_accuracy(B_bin_true, B_bin_est)
        results.update(results_dag)
    except:    # To be safe
        pass

    try:
        # Calculate SHD-CPDAG
        shd_cpdag = compute_shd_cpdag(B_bin_true, B_bin_est)
        results['shd_cpdag'] = shd_cpdag
    except:    # To be safe
        pass

    try:
        # Calculate performance metrics for skeleton
        results_skeleton = count_skeleton_accuracy(B_bin_true, B_bin_est)
        results.update(results_skeleton)
    except:    # To be safe
        pass

    try:
        # Calculate performance metrics for arrows
        results_arrows = count_arrows_accuracy(B_bin_true, B_bin_est)
        results.update(results_arrows)
    except:    # To be safe
        pass
    return results