def f1_counts(tp, fp, fn, n_true=None, n_pred=None):
    """
    :param tp: number of true positives.
    :param fp: number of false positives.
    :param fn: number of false negatives.
    :param n_true: total number of true positives in the ground truth. If None, computed as tp + fn.
    :param n_pred: total number of predicted positives. If None, computed as tp + fp.
    :return: dictionary containing precision, recall, F1-score and confusion counts.
    """
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_true": n_true if n_true is not None else tp + fn,
        "n_pred": n_pred if n_pred is not None else tp + fp,
    }


def set_f1(true_set, pred_set):
    true_set = set(true_set)
    pred_set = set(pred_set)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    return f1_counts(tp, fp, fn, len(true_set), len(pred_set))


def canon_undirected(edge):
    """
    :param edge: temporal edge represented as ((source_name, source_time), (target_name, target_time)).
    :return: canonical undirected representation of the edge.
    """
    u, v = edge
    return tuple(sorted((u, v), key=lambda x: (x[0], x[1])))


def evaluate_all_ts(true_edges, pred_oriented, pred_undirected, shifted_nodes,):
    """
    :param true_edges: set of true directed temporal edges.
    :param pred_oriented: set of predicted directed temporal edges.
    :param pred_undirected: set of predicted undirected temporal edges.
    :param shifted_nodes: set of true shifted node names.
    :return: dictionary containing the temporal evaluation metrics.
    """

    true_edges = set(true_edges)
    pred_oriented = set(pred_oriented)
    pred_undirected = set(pred_undirected)
    shifted_nodes = set(shifted_nodes)

    # =====================================================
    # 1. Incoming shifted node F1
    # =====================================================
    pred_incoming_shifted_nodes = {target[0] for _, target in pred_oriented}

    incoming_shifted = set_f1(shifted_nodes, pred_incoming_shifted_nodes,)

    # =====================================================
    # 2. Relaxed shifted node F1
    # =====================================================
    pred_shifted_nodes_relaxed = set(pred_incoming_shifted_nodes)

    for source, target in pred_undirected:
        pred_shifted_nodes_relaxed.add(source[0])
        pred_shifted_nodes_relaxed.add(target[0])

    node_relaxed = set_f1(shifted_nodes, pred_shifted_nodes_relaxed,)

    return {"incoming_shifted": incoming_shifted, "node_relaxed": node_relaxed,}