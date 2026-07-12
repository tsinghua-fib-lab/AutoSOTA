from typing import List, Dict

def kendall_tau_loss(true_ranking: List[int], predicted_ranking: List[int], k: int = -1) -> float:
    """
    Computes the normalized Kendall Tau distance between two rankings.
    It considers only the pairs of items within the top-k of the true_ranking.
    If k is -1, it considers all items.
    """
    if k == -1:
        k = len(true_ranking)
    
    true_top_k = true_ranking[:k]
    
    # Position map for the predicted ranking for efficient lookups.
    pred_pos: Dict[int, int] = {item: i for i, item in enumerate(predicted_ranking)}

    # If an item from true_top_k is not in predicted_ranking, we can assign it
    # the worst possible rank (len(predicted_ranking)).
    max_pos = len(predicted_ranking)

    dist = 0
    for i in range(len(true_top_k)):
        for j in range(i + 1, len(true_top_k)):
            item1 = true_top_k[i]
            item2 = true_top_k[j]
            
            pos1_pred = pred_pos.get(item1, max_pos)
            pos2_pred = pred_pos.get(item2, max_pos)

            # In true_ranking, item1 is ranked higher than item2.
            # We check if this order is preserved in predicted_ranking.
            if pos1_pred > pos2_pred:
                dist += 1
    
    num_pairs = k * (k - 1) / 2
    if num_pairs == 0:
        return 0.0

    return dist / num_pairs
