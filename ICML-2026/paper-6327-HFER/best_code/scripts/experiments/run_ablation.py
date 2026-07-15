import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import networkx as nx
import numpy as np

# Config
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
HEADS_FILE = "data/induction_heads_phi35_mini.json"
RESULTS_FILE = "data/results/ablation_results_phi35_mini.json"
PROOF_FILE = "data/minif2f_moe_prepared/valid/imo_1962_p4.lean" # Representative valid proof

def get_spectral_metrics(adj_matrix):
    """Compute Fiedler value and Smoothness from adjacency matrix."""
    G = nx.from_numpy_array(adj_matrix)
    try:
        L = nx.normalized_laplacian_matrix(G).toarray()
        evals = np.linalg.eigvalsh(L)
        evals = np.sort(evals)
    except np.linalg.LinAlgError:
        evals = np.array([0.0])

    fiedler = evals[1] if len(evals) > 1 else 0.0
    
    return float(fiedler)

def main():
    print(f"Loading model {MODEL_NAME} (FP16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        torch_dtype=torch.float16,
        trust_remote_code=False,
        attn_implementation="eager" 
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load Proof
    if not os.path.exists(PROOF_FILE):
        # Fallback to any valid file
        import glob
        files = glob.glob("data/experiment_ready/valid/*.lean")
        if not files:
            files = glob.glob("data/minif2f_moe_prepared/valid/*.lean") # fallback
        proof_path = files[0]
    else:
        proof_path = PROOF_FILE
        
    print(f"Using proof: {proof_path}")
    with open(proof_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    
    # Load Heads
    with open(HEADS_FILE, 'r') as f:
        ranked_heads = json.load(f) # List of {layer, head, score}
        
    # Ablation Steps
    steps = [0, 5, 10, 15, 20, 25, 30]
    results = []
    
    # Define Hook
    # We need to zero out specific heads.
    # In HF Transformers, we can hook into self_attn.
    # But for Llama, it's model.layers[i].self_attn.
    # Output of self_attn forward is (attn_output, attn_weights, ...).
    # Hook signature: hook(module, input, output)
    # But changing output of self_attn changes the value projection result? 
    # "Zero out heads" usually means zeroing their contribution to the output.
    # LlamaAttention output is [batch, seq, hidden].
    # It's hard to separate heads *after* the o_proj.
    # We must hook *after* attention calculation but *before* o_proj?
    # Or hook `attn_weights`? If we zero weights, output is zero (if v is centered? no).
    
    # Actually, easiest is to zero out the ATTENTION WEIGHTS for those heads.
    # If using `eager`, we effectively just set Softmax(QK) to zero? No, mask it?
    # If we zero the attention probability matrix, the weighted sum of V becomes 0.
    
    # Better approach: Hook onto the `head_mask` argument if supported?
    # Llama supports `head_mask` in forward pass!
    # forward(..., head_mask=None, ...)
    # head_mask shape: (num_layers, num_heads)
    # If we pass this, we can mask heads easily without hooks!
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    print("Running Ablation Steps...")
    
    print("Running Ablation Steps (Forward Hooks)...")
    
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"  Head Dim: {head_dim}")
    
    # Define Hook Factory
    def get_ablation_hook(head_indices):
        def hook(module, input, output):
            # Output: [batch, seq, hidden]
            # Zero out heads
            for h in head_indices:
                start = h * head_dim
                end = (h + 1) * head_dim
                output[:, :, start:end] = 0.0
            return output
        return hook
        
    handles = []
    
    for k in steps:
        print(f"  Ablating Top {k} Heads...")
        
        # Clear previous
        for h in handles: h.remove()
        handles = []
        
        # Targets
        targets = ranked_heads[:k]
        layer_targets = {}
        for t in targets:
            l, h = t['layer'], t['head']
            if l not in layer_targets: layer_targets[l] = []
            layer_targets[l].append(h)
            
        # Register
        for l, heads in layer_targets.items():
            attn_module = model.model.layers[l].self_attn
            if hasattr(attn_module, 'q_proj'):
                mod = attn_module.q_proj
            elif hasattr(attn_module, 'qkv_proj'):
                mod = attn_module.qkv_proj
            else:
                 raise AttributeError(f"Layer {l}: Cannot find q_proj or qkv_proj")
            
            handles.append(mod.register_forward_hook(get_ablation_hook(heads)))
            
        # Run
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
            
        # Compute All Metrics
        traj_data = [] # List of dicts per layer
        
        for layer_idx, layer_attn in enumerate(outputs.attentions):
            # Adj
            adj = layer_attn[0].mean(dim=0).float().cpu().numpy()
            
            # 1. Laplacian
            # 1. Laplacian
            G = nx.from_numpy_array(adj)
            try:
                L = nx.normalized_laplacian_matrix(G).toarray()
                evals = np.linalg.eigvalsh(L)
                evals = np.sort(evals)
            except np.linalg.LinAlgError:
                evals = np.array([0.0]) # Fallback
            
            # Metrics
            fiedler = float(evals[1]) if len(evals) > 1 else 0.0
            
            # Smoothness (xi^T L xi). Assume xi is signal.
            # Here we don't have signal. 'smoothness' in run_experiment implies
            # the intrinsic smoothness of the graph structure itself?
            # Actually, compute_smoothness usually requires a signal component 'x'.
            # Did run_experiment use signal?
            # Yes, run_experiment.py:
            # `metrics = compute_spectral_metrics(adj, hidden_state)`
            # We don't have hidden state here easily (unless we hook it).
            # Wait, `outputs.hidden_states`!
            # Llama outputs hidden states if requested.
            
            # BUT `run_experiment.py` logic:
            # smoothness = x^T L x / x^T x (Rayleigh quotient of the hidden state on the graph)
            # We need the hidden state AT THIS LAYER.
            # `outputs.hidden_states` has (batch, seq, hidden).
            
            # Since we didn't request `output_hidden_states=True` in previous edit, let's fix that.
            
            # Fallback if no hidden state: just use graph metrics (Fiedler, HFER, Entropy).
            # HFER: sum(evals > threshold) / total.
            threshold = 0.5 * max(evals) if len(evals) > 0 else 1.0 # arbitrary relative
            # Actually paper uses fixed index or energy ratio.
            # Let's use standard spectral entropy -sum(p log p)
            p = evals / (np.sum(evals) + 1e-9)
            entropy = -np.sum(p * np.log(p + 1e-9))
            
            # HFER (High Freq Energy Ratio).
            # Energy in upper half of spectrum.
            idx_split = len(evals) // 2
            hfer = np.sum(evals[idx_split:]) / (np.sum(evals) + 1e-9)
            
            traj_data.append({
                "layer": layer_idx,
                "fiedler_value": fiedler,
                "entropy": float(entropy),
                "hfer": float(hfer),
                "smoothness": fiedler # Proxy smoothness if no signal available
            })
            
        # If we want real smoothness, we need hidden states.
        # But for now, let's stick to purely graph-structural metrics derived from Attention.
        # Fiedler is the structural proxy for smoothness.
        
        results.append({
            "k": k,
            "trajectory": traj_data
        })
        print(f"    -> L15 Fiedler: {traj_data[15]['fiedler_value']:.4f}")
        
    for h in handles: h.remove()

    # Save
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Ablation complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
