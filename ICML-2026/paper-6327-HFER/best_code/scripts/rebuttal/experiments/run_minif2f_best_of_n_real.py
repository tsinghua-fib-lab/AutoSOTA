import os
import json
import torch
import numpy as np
import random
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- MEMORY EFFICIENT HFER EXTRACTOR ---
def compute_hfer_single_layer(attention_matrix):
    A = attention_matrix.mean(dim=0).float().cpu().numpy()
    N = A.shape[0]
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    try:
        vals = np.linalg.eigvalsh(L)
        vals = np.sort(vals)
        fiedler = vals[1] if len(vals) > 1 else 0
        return float(fiedler / (vals[-1] + 1e-6))
    except: return 0.0

def get_hfer_pair(model, tokenizer, text):
    """Returns (hfer_l24, hfer_l30) for a given text."""
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        # IMPORTANT: Model MUST be loaded with output_attentions=True
        outputs = model(**inputs, output_attentions=True)
        # Check if attentions exist
        if not hasattr(outputs, 'attentions') or outputs.attentions is None:
            raise ValueError("Model is not returning attentions. Ensure output_attentions=True.")
        
        h24 = compute_hfer_single_layer(outputs.attentions[24][0])
        h30 = compute_hfer_single_layer(outputs.attentions[30][0])
        del outputs
        torch.cuda.empty_cache()
    return h24, h30

# --- EXPERIMENT ---
VALIDITY_THRESHOLD_L30 = 0.2518

def run_best_of_n_experiment(n_theorems=25, n_candidates=8):
    print("=" * 70)
    print("  REAL MINIF2F BEST-OF-N EVALUATION (Optimized)")
    print("=" * 70)

    # 1. Identify failing greedy proofs
    with open('data/results/rebuttal/llama8b_full_extraction.json') as f:
        full_data = json.load(f)
    failing_greedy = [x for x in full_data if x.get('label_corrected') == 'invalid']
    selected_theorems = random.sample(failing_greedy, min(n_theorems, len(failing_greedy)))

    # 2. Load Model (4-bit, output_attentions=True)
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading {model_id}...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto",
        output_attentions=True
    )

    results = []
    for theorem_entry in tqdm(selected_theorems, desc="Theorems"):
        theorem_name = theorem_entry['file']
        file_path = f"data/experiment_ready/all/{theorem_name}"
        if not os.path.exists(file_path):
            for d in ['valid', 'invalid']:
                if os.path.exists(f"data/experiment_ready/{d}/{theorem_name}"):
                    file_path = f"data/experiment_ready/{d}/{theorem_name}"; break
        
        with open(file_path, 'r', encoding='utf-8') as f:
            full_content = f.read()
            problem_part = full_content.split("proof")[0].split("by")[0]
        
        candidates = []
        inputs = tokenizer(problem_part, return_tensors="pt").to("cuda")
        for _ in range(n_candidates):
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.95, return_dict_in_generate=True, output_scores=True)
                gen_text = tokenizer.decode(out.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                # Log-probs: correct compute_transition_scores call
                t_scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
                mean_lp = float(np.mean(t_scores[0].cpu().numpy()))
                
                # Spectral: (hfer_l24, hfer_l30)
                h24, h30 = get_hfer_pair(model, tokenizer, problem_part + gen_text)
                
                candidates.append({
                    "text": gen_text, 
                    "mean_log_prob": mean_lp, 
                    "hfer_l24": h24, 
                    "hfer_l30": h30, 
                    "is_valid_proxy": h30 < VALIDITY_THRESHOLD_L30
                })
            torch.cuda.empty_cache()
            
        results.append({"theorem": theorem_name, "candidates": candidates})

    # Save
    os.makedirs("data/results/rebuttal", exist_ok=True)
    with open("data/results/rebuttal/best_of_n_real.json", "w") as f:
        json.dump(results, f, indent=2)

    # Simple Summary
    print("\n--- SUMMARY ---")
    for strat in ["Random", "Max Log-Prob", "Min HFER (Ours)"]:
        acc = []
        for r in results:
            cands = r['candidates']
            if strat == "Random": acc.append(np.mean([c['is_valid_proxy'] for c in cands]))
            elif strat == "Max Log-Prob": acc.append(max(cands, key=lambda x: x['mean_log_prob'])['is_valid_proxy'])
            elif strat == "Min HFER (Ours)": acc.append(min(cands, key=lambda x: x['hfer_l24'])['is_valid_proxy'])
        print(f"{strat:15} | {100*np.mean(acc):.1f}%")

if __name__ == "__main__":
    run_best_of_n_experiment()
