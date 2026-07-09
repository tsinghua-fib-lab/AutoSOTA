from tqdm import tqdm
import numpy as np

def compute_attribution(discoverer, probe, seqs, batch_size=8, sequential=False, freeze_attention=True, source="mlp_output"):
    global_attr = {}    
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        is_cnn = (probe.__class__.__name__ == "CNNProbe")
        attr_batch = discoverer.get_gradients(
            batch, 
            probe, 
            cnn=is_cnn,
            sequential=sequential, 
            freeze_attention=freeze_attention,
            source=source 
        )
        for l, scores in attr_batch.items():
            if l not in global_attr: 
                global_attr[l] = np.zeros_like(scores)
            global_attr[l] += scores 
     
    return global_attr

def rank_nodes(global_attr):
    """
    Flattens and ranks nodes by attribution score.
    Returns: List of (layer, node_idx, score) sorted descending.
    """
    ranking = []
    for l, scores in global_attr.items():
        for idx, s in enumerate(scores):
            ranking.append((l, idx, s))
    ranking.sort(key=lambda x: x[2], reverse=True)
    return ranking

def circuit_search(
    discoverer, 
    probe, 
    ranking, 
    val_seqs, 
    val_y, 
    target_metric, 
    metric_fn, # e.g. evaluate_circuit for F1, or evaluate_regression for Pearson
    batch_size=8,
    step_size=32,
    max_nodes=1000,
    desc="Scanning",
    **kwargs
):
    """
    Performs selection of circuit nodes.
    Returns: (best_nodes_dict, best_k, best_metric_val)
    """
    best_nodes = {}
    best_k = max_nodes
    best_metric_val = -float('inf') 
    highest_seen_metric = -float('inf')
    best_seen_config = None # Will store (nodes, k, metric)
    step_iter = range(step_size, max_nodes + 1, step_size)
    with tqdm(step_iter, desc=desc, position=1, leave=False) as pbar:
        for k in pbar:
            # 1. Select top k nodes
            top_k = ranking[:k]
            active = {}
            for l, n, _ in top_k:
                if l not in active: active[l] = set()
                active[l].add(n)

            # 2. Evaluate
            # metric_fn must accept (discoverer, probe, seqs, y, active_nodes, batch_size)
            curr_metric = metric_fn(discoverer, probe, val_seqs, val_y, active, batch_size)
            if curr_metric > highest_seen_metric:
                highest_seen_metric = curr_metric
                best_seen_config = (active, k, curr_metric)
            pbar.set_postfix({
                "nodes": k,
                "score": f"{curr_metric:.3f}",
                "target": f"{target_metric:.3f}",
                "peak": f"{highest_seen_metric:.3f}"
            })
            
            # 3. Check stopping condition
            if curr_metric >= target_metric:
                best_nodes = active
                best_k = k
                best_metric_val = curr_metric
                break
    
    # If we never reached the target, take the best config
    if not best_nodes:
        if best_seen_config is not None:
            best_nodes, best_k, best_metric_val = best_seen_config
        else:
            best_k = max_nodes
            top_k = ranking[:best_k]
            for l, n, _ in top_k:
                if l not in best_nodes: best_nodes[l] = set()
                best_nodes[l].add(n)
            best_metric_val = metric_fn(discoverer, probe, val_seqs, val_y, best_nodes, batch_size, **kwargs)
            
    return best_nodes, best_k, best_metric_val
