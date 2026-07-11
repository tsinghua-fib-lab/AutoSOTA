import pandas as pd

def f1_score_edges(true_graph: pd.DataFrame, learned_graph: pd.DataFrame) -> dict:
    """
    Compute precision, recall, and F1 score between true and learned graphs.

    Reference:
        https://github.com/ban-epfl/rcd/blob/main/rcd/utilities/utils.py#L57
    """

  
    true_edges = set()
    learned_edges = set()
    nodes = true_graph.index.tolist()

    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i < j:  # consider each edge only once
                if true_graph.iloc[i, j] != 0:
                    true_edges.add((node_i, node_j))
                if learned_graph.iloc[i, j] != 0:
                    learned_edges.add((node_i, node_j))

    # Compute precision, recall, F1 score
    tp = len(true_edges.intersection(learned_edges))
    fp = len(learned_edges - true_edges)
    fn = len(true_edges - learned_edges)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'f1': f1, 'precision': precision, 'recall': recall}
