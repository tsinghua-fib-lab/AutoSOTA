import numpy as np
import ipywidgets as widgets
from IPython.display import display
from .heatmap import render_context_heatmap

def compare_texts_with_sae(guard, text1, text2, top_k=20, pooling="max", include_response=False):
    """
    compare the SAE activation differences between two text segments.
    guard: SAEStreamGuard instance
    """
    # 1. Get the activations
    # Note: here we assume the guard object has a predict_full_acts method
    data1 = {"prompt": [text1]}
    acts1, _ = guard.predict_full_acts(data1, include_response=include_response)
    acts1 = acts1[0].float().cpu() # [Length1, Features]
    
    data2 = {"prompt": [text2]}
    acts2, _ = guard.predict_full_acts(data2, include_response=include_response)
    acts2 = acts2[0].float().cpu() # [Length2, Features]
    
    # 2. Pooling
    if pooling == "mean":
        p1 = acts1.mean(dim=0)
        p2 = acts2.mean(dim=0)
    else: # max
        p1 = acts1.max(dim=0)[0]
        p2 = acts2.max(dim=0)[0]
        
    # 3. Diff
    diff = (p1 - p2).numpy()
    top_indices = np.argsort(diff)[-top_k:][::-1]
    
    top_feats = []
    for idx in top_indices:
        top_feats.append({
            "feature_id": int(idx),
            "diff_value": float(diff[idx]),
            "text1_activation": float(p1[idx]),
            "text2_activation": float(p2[idx])
        })
        
    return {
        "top_diff_features": top_feats,
        "all_diffs": diff,
        "text1_acts": acts1,
        "text2_acts": acts2,
        "pooling": pooling
    }

def visualize_top_diff_features(result, guard, text1, top_k=10):
    """
    Visualize the difference features.
    """
    top_features = result["top_diff_features"][:top_k]
    text1_acts = result["text1_acts"]
    text2_acts = result["text2_acts"]
    
    # Get the token strings
    str_tokens = guard.model.to_str_tokens(text1)
    
    w_list = [widgets.HTML(f"<h3>🔍 Diff Visualization (Text 1 - Text 2)</h3>")]
    
    for feat in top_features:
        fid = feat['feature_id']
        
        # Calculate the difference for each token position
        # Need to align the lengths
        min_len = min(len(text1_acts), len(text2_acts))
        seq1 = text1_acts[:, fid].numpy()
        seq2 = text2_acts[:, fid].numpy()
        
        diff_vals = seq1[:min_len] - seq2[:min_len]
        # If text1 is longer, the remaining part is displayed as is
        if len(seq1) > min_len:
            diff_vals = np.concatenate([diff_vals, seq1[min_len:]])
            
        val_dict = {i: float(v) for i, v in enumerate(diff_vals)}
        
        # Render
        header = f"<div style='background:#eee; padding:5px;'><b>Feature #{fid}</b> | Diff: {feat['diff_value']:.4f} (T1: {feat['text1_activation']:.2f}, T2: {feat['text2_activation']:.2f})</div>"
        w_list.append(widgets.HTML(header))
        
        heatmap = render_context_heatmap(
            str_tokens, val_dict,
            positive_color=(0, 0, 255), # Blue represents Text1 is stronger
            negative_color=(255, 0, 0)  # Red represents Text2 is stronger
        )
        w_list.append(widgets.HTML(heatmap))
        w_list.append(widgets.HTML("<hr style='margin:5px 0'>"))
        
    display(widgets.VBox(w_list))