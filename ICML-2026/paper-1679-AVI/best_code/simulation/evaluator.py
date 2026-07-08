
import numpy as np
import pandas as pd
from core.confidence_sets import compute_topk_confidence_set


def evaluate_results(result, true_probs_info, m, top_k=None):

    true_probs = true_probs_info['probs']
    true_ranking = true_probs_info['true_ranking']
    
    print("\n=== Evaluation ===")
    
    rejected_pairs = np.argwhere(result['final_rejected'])
    
    all_correct_pairs = np.sum(true_probs > 0.5)
    
    if len(rejected_pairs) == 0:
        print("No pairs were rejected.")
        topk_set = [] if top_k is not None else None
        missing_topk = true_probs_info['true_ranking'][:top_k].tolist() if top_k is not None else None
        return {
            'correct_discoveries': 0,
            'false_discoveries': 0,
            'precision': 0,
            'power': 0,
            'all_correct_pairs': all_correct_pairs,
            'discovered_preferences': pd.DataFrame(),
            'all_discovered_pairs': pd.DataFrame(),
            'top_k_confidence_set': topk_set,
            'top_k_missing': missing_topk,
            'top_k': top_k
        }
    
    correct_discoveries = 0
    false_discoveries = 0
    discovered_pairs_list = []
    
    for idx in range(len(rejected_pairs)):
        j, i = rejected_pairs[idx]
        
        if true_probs[j, i] > 0.5:
            correct_discoveries += 1
            is_correct = True
        else:
            false_discoveries += 1
            is_correct = False
        
        discovered_pairs_list.append({
            'model_j': j + 1,
            'model_i': i + 1,
            'true_prob': true_probs[j, i],
            'is_correct': is_correct
        })
    
    all_discovered_pairs = pd.DataFrame(discovered_pairs_list)
    
    precision = correct_discoveries / (correct_discoveries + false_discoveries)
    power = correct_discoveries / all_correct_pairs if all_correct_pairs > 0 else 0
    
    print(f"Correct discoveries: {correct_discoveries}")
    print(f"False discoveries: {false_discoveries}")
    print(f"All correct pairs: {all_correct_pairs}")
    print(f"Precision: {precision:.3f}")
    print(f"Power: {power:.3f}")
    
    print(f"\nTrue ranking (best to worst): {' > '.join(map(str, true_ranking))}")
    
    if len(rejected_pairs) > 0:
        print("\nDiscovered preferences:")
        display_count = min(len(rejected_pairs), 20)
        for idx in range(display_count):
            j, i = rejected_pairs[idx]
            correctness = "✓" if true_probs[j, i] > 0.5 else "✗"
            print(f"{correctness} Model {j+1} ≻ Model {i+1} (true prob: {true_probs[j, i]:.3f})")
        if len(rejected_pairs) > 20:
            print(f"... and {len(rejected_pairs) - 20} more pairs")
    
    if top_k is not None:
        final_topk_info = compute_topk_confidence_set(result['final_rejected'], m, top_k)
        topk_models = final_topk_info['topk_models']
        true_topk = true_ranking[:top_k]
        missing_topk = list(set(true_topk) - set(topk_models))
        
        print(f"\nTop-{top_k} confidence set (size {len(topk_models)}): {', '.join(map(str, topk_models)) if len(topk_models) > 0 else '∅'}")
        if len(missing_topk) == 0:
            print("All true top-k models are included in the confidence set.")
        else:
            print(f"Missing true top-k models: {', '.join(map(str, missing_topk))}")
    else:
        topk_models = None
        missing_topk = None
    
    return {
        'correct_discoveries': correct_discoveries,
        'false_discoveries': false_discoveries,
        'precision': precision,
        'power': power,
        'all_correct_pairs': all_correct_pairs,
        'discovered_preferences': all_discovered_pairs,
        'all_discovered_pairs': all_discovered_pairs,
        'true_ranking': true_ranking,
        'total_pairs': len(rejected_pairs),
        'top_k_confidence_set': topk_models,
        'top_k_missing': missing_topk,
        'top_k': top_k
    }

