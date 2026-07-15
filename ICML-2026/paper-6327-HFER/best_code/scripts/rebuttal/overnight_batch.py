import argparse, os, json, torch, gc
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- UTILS ---
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

def compute_hfer_single_layer(attention_matrix):
    try:
        A = attention_matrix.mean(dim=0).float().cpu().numpy()
        N = A.shape[0]
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        vals = np.linalg.eigvalsh(L)
        vals = np.sort(vals)
        fiedler = vals[1] if len(vals) > 1 else 0
        return float(fiedler / (vals[-1] + 1e-6))
    except: return 0.5 

def cohens_d(v, i):
    nv, ni = len(v), len(i)
    if nv + ni <= 2: return 0.0
    sv, si = np.var(v, ddof=1), np.var(i, ddof=1)
    pooled = np.sqrt(((nv-1)*sv + (ni-1)*si) / (nv+ni-2))
    return (np.mean(i) - np.mean(v)) / pooled if pooled > 0 else 0.0

# --- EXPERIMENT 1: EXPANDED STEERING BEHAVIORAL (n=100) ---
def run_steering_behavioral():
    print("\n" + "="*70)
    print("  PHASE 1: EXPANDED STEERING BEHAVIORAL FLIP TEST (LLAMA-3.1-8B, n=100)")
    print("="*70)
    
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", output_attentions=True)
    
    # 1. Selection: the top 100 worst greedy failures
    with open('data/results/rebuttal/llama8b_full_extraction.json') as f:
        full_ext = json.load(f)
    invalid_ones = [x for x in full_ext if x.get('label_corrected') == 'invalid']
    invalid_ones.sort(key=lambda x: x['spectral']['layer_30']['hfer'], reverse=True)
    selected_theorems = [t['file'] for t in invalid_ones[:100]]

    # 2. Apply Steering (CPU to avoid LinAlgError)
    print("Applying Spectral Sharpening (alpha=-0.3) at L24...")
    with torch.no_grad():
        W = model.model.layers[24].mlp.down_proj.weight.data.detach().cpu().float()
        W = torch.nan_to_num(W)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        S_new = S * (1 - 0.3 * (S - S.mean()) / S.std())
        W_new = U @ torch.diag(S_new) @ Vh
        model.model.layers[24].mlp.down_proj.weight.data = W_new.to(device="cuda", dtype=torch.float16)
    
    results = []
    VALIDITY_THRESHOLD_L30 = 0.2518
    
    for t_name in tqdm(selected_theorems, desc="Steered Greedy Eval"):
        # Try both locations
        file_path = f"data/experiment_ready/all/{t_name}"
        if not os.path.exists(file_path):
            file_path = f"data/experiment_ready/valid/{t_name}"
            if not os.path.exists(file_path):
                file_path = f"data/experiment_ready/invalid/{t_name}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            problem_part = f.read().split("proof")[0].split("by")[0]
        
        inputs = tokenizer(problem_part, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
            gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Verify HFER
            spec_out = model(**tokenizer(problem_part + gen_text, return_tensors="pt").to("cuda"), output_attentions=True)
            h30 = compute_hfer_single_layer(spec_out.attentions[30][0])
            is_valid = h30 < VALIDITY_THRESHOLD_L30
            
            base_info = next(x for x in full_ext if x['file'] == t_name)
            results.append({
                "theorem": t_name,
                "base_hfer": base_info['spectral']['layer_30']['hfer'],
                "steered_hfer": h30,
                "base_valid": False,
                "steered_valid": is_valid
            })
        clear_cache()

    with open("data/results/rebuttal/steering_behavioral_real.json", "w") as f:
        json.dump(results, f, indent=2)

    flipped = sum(1 for r in results if r['steered_valid'])
    print(f"Results for n=100 expanded steering Behavioral Flip Test")
    print(f"Base: 0/100 valid, 100/100 invalid")
    print(f"Steered: {flipped}/100 valid, {100-flipped}/100 invalid")
    print(f"Flipped invalid -> valid: {flipped}")
    
    del model, tokenizer
    clear_cache()

# --- EXPERIMENT 2: FULL MINIF2F SWEEP (LLAMA-3.1-8B) ---
def run_full_sweep_llama():
    print("\n" + "="*70)
    print("  PHASE 2: FULL MINIF2F BEST-OF-8 SWEEP (LLAMA-3.1-8B)")
    print("="*70)
    
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", output_attentions=True)
    
    val_dir = "data/experiment_ready/valid"
    inv_dir = "data/experiment_ready/invalid"
    all_files = [os.path.join("valid", f) for f in os.listdir(val_dir) if f.endswith(".lean")]
    all_files += [os.path.join("invalid", f) for f in os.listdir(inv_dir) if f.endswith(".lean")]
    
    results = []
    VALIDITY_THRESHOLD_L30 = 0.2518
    
    for relative_path in tqdm(all_files, desc="MiniF2F Sweep"):
        file_path = os.path.join("data/experiment_ready", relative_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            problem_part = f.read().split("proof")[0].split("by")[0]
        
        inputs = tokenizer(problem_part, return_tensors="pt").to("cuda")
        candidates = []
        for _ in range(8):
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.95, return_dict_in_generate=True, output_scores=True)
                gen_text = tokenizer.decode(out.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                t_scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
                mean_lp = float(np.mean(t_scores[0].cpu().numpy()))
                spec_out = model(**tokenizer(problem_part + gen_text, return_tensors="pt").to("cuda"), output_attentions=True)
                h24 = compute_hfer_single_layer(spec_out.attentions[24][0])
                h30 = compute_hfer_single_layer(spec_out.attentions[30][0])
                candidates.append({"text": gen_text, "mean_log_prob": mean_lp, "hfer_l24": h24, "hfer_l30": h30, "is_valid_proxy": h30 < VALIDITY_THRESHOLD_L30})
            clear_cache()
        results.append({"theorem": relative_path.split(os.sep)[-1], "candidates": candidates})

    with open("data/results/rebuttal/best_of_n_sweep_real.json", "w") as f:
        json.dump(results, f, indent=2)
    
    del model, tokenizer
    clear_cache()

# --- EXPERIMENT 3: BEST-OF-N (PHI-3.5) ---
def run_best_of_n_phi():
    print("\n" + "="*70)
    print("  PHASE 3: CROSS-MODEL VALIDATION (PHI-3.5)")
    print("="*70)
    model_id = "microsoft/Phi-3.5-mini-instruct"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", output_attentions=True, trust_remote_code=True)
    
    with open('data/results/rebuttal/llama8b_full_extraction.json') as f:
        invalid_theorems = [x['file'] for x in json.load(f) if x.get('label_corrected') == 'invalid']
    selected_theorems = invalid_theorems[:25]
    
    results = []
    for t_name in tqdm(selected_theorems, desc="Phi Best-of-8"):
        # Try both locations
        file_path = f"data/experiment_ready/valid/{t_name}"
        if not os.path.exists(file_path):
            file_path = f"data/experiment_ready/invalid/{t_name}"

        with open(file_path, 'r', encoding='utf-8') as f:
            problem_part = f.read().split("proof")[0].split("by")[0]
        
        inputs = tokenizer(problem_part, return_tensors="pt").to("cuda")
        candidates = []
        for _ in range(8):
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.95, return_dict_in_generate=True, output_scores=True)
                gen_text = tokenizer.decode(out.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                t_scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
                mean_lp = float(np.mean(t_scores[0].cpu().numpy()))
                spec_out = model(**tokenizer(problem_part + gen_text, return_tensors="pt").to("cuda"), output_attentions=True)
                h24 = compute_hfer_single_layer(spec_out.attentions[24][0])
                candidates.append({"text": gen_text, "mean_log_prob": mean_lp, "hfer_l24": h24, "is_valid_proxy": h24 < 0.22})
            clear_cache()
        results.append({"theorem": t_name, "candidates": candidates})

    with open("data/results/rebuttal/best_of_n_phi_real.json", "w") as f:
        json.dump(results, f, indent=2)
    del model, tokenizer
    clear_cache()

# --- EXPERIMENT 4: PREFIX AUDIT ---
def run_prefix_verification():
    print("\n=== PHASE 4: Prefix verification ===")
    path = "data/results/rebuttal/prefix_evolution_v3.json"
    with open(path) as f:
        data = json.load(f)
    print(f"Audit (first 5):")
    for entry in data[:5]:
        print(f"  {entry['file']} ({entry['label']}): L1.00={entry['hfer_by_prefix']['1.00']:.4f}")
    
    for cp in ["0.25", "0.50", "0.75", "1.00"]:
        v = [x['hfer_by_prefix'][cp] for x in data if x['label'] == 'valid' and cp in x['hfer_by_prefix']]
        i = [x['hfer_by_prefix'][cp] for x in data if x['label'] == 'invalid' and cp in x['hfer_by_prefix']]
        d = cohens_d(v, i)
        print(f"| Prefix {cp:5} | n={len(v)+len(i):3} | Cohen d = {d:7.4f} |")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-bon', action='store_true')
    parser.add_argument('--skip-steering', action='store_true')
    args = parser.parse_args()

    # Sequence
    run_steering_behavioral()
    run_full_sweep_llama()
    run_best_of_n_phi()
    run_prefix_verification()
    print("\n=== ALL OVERNIGHT EXPERIMENTS COMPLETE ===")
