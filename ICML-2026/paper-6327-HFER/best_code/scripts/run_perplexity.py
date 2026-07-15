
import os
import glob
import torch
import numpy as np
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--data-dir", type=str, default="data/experiment_ready")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-in-4bit", action="store_true")
    # Add corrections file to filter invalids correctly
    parser.add_argument("--list-b", type=str, default="analysis/8B_list_b_confident_invalid.json")
    return parser.parse_args()

def get_log_probs(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        # Loss is average negative log-likelihood
        # loss = -mean(log(P(x)))
        # So mean_log_prob = -loss
        loss = outputs.loss
    return -loss.item()

def main():
    args = parse_args()
    print(f"Loading {args.model} for Perplexity Baseline...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    model_kwargs = {"device_map": args.device}
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    
    # Load List B for corrections
    reclaimed_files = set()
    if os.path.exists(args.list_b):
        with open(args.list_b, 'r') as f:
            raw = json.load(f)
            if raw and isinstance(raw[0], dict):
                reclaimed_files = set(item['file'] for item in raw)
            else:
                reclaimed_files = set(raw)
    
    results = []
    
    # Process Valid
    valid_files = glob.glob(os.path.join(args.data_dir, "valid", "*.lean"))
    print(f"Processing {len(valid_files)} Valid proofs...")
    for fpath in tqdm(valid_files):
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read()
        lp = get_log_probs(model, tokenizer, text, args.device)
        results.append({'label': 1, 'log_prob': lp, 'set': 'valid'})
        
    # Process Invalid
    invalid_files = glob.glob(os.path.join(args.data_dir, "invalid", "*.lean"))
    print(f"Processing {len(invalid_files)} Invalid proofs...")
    for fpath in tqdm(invalid_files):
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        fname = os.path.basename(fpath)
        label = 0
        if fname in reclaimed_files:
            label = 1
            
        lp = get_log_probs(model, tokenizer, text, args.device)
        results.append({'label': label, 'log_prob': lp, 'set': 'invalid'})
        
    # Analysis
    valid_lps = [r['log_prob'] for r in results if r['label'] == 1]
    invalid_lps = [r['log_prob'] for r in results if r['label'] == 0]
    
    mu_v, std_v = np.mean(valid_lps), np.std(valid_lps)
    mu_i, std_i = np.mean(invalid_lps), np.std(invalid_lps)
    
    print("\n--- PERPLEXITY (LOG-PROB) BASELINE RESULTS ---")
    print(f"Valid Mean LogProb:   {mu_v:.4f} (Perplexity: {np.exp(-mu_v):.2f})")
    print(f"Invalid Mean LogProb: {mu_i:.4f} (Perplexity: {np.exp(-mu_i):.2f})")
    
    # Effect Size
    n_v, n_i = len(valid_lps), len(invalid_lps)
    pooled_std = np.sqrt(((n_v-1)*std_v**2 + (n_i-1)*std_i**2)/(n_v+n_i-2))
    d = (mu_v - mu_i) / pooled_std
    print(f"Cohen's d: {d:.4f}")
    
    # Classification Accuracy
    best_acc = 0
    best_t = 0
    thresholds = np.linspace(min(valid_lps+invalid_lps), max(valid_lps+invalid_lps), 100)
    
    for t in thresholds:
        # Higher log_prob = More certain? 
        # Usually valid proofs are "more probable" -> Higher LogProb (less negative)
        acc_gt = sum(1 for r in results if (1 if r['log_prob'] > t else 0) == r['label']) / len(results)
        if acc_gt > best_acc:
            best_acc = acc_gt
            best_t = t
            
    print(f"Best Accuracy: {best_acc*100:.2f}% (Threshold: {best_t:.4f})")
    
    if best_acc < 0.90:
        print("RESULT: Spectral Method BEATS Perplexity Baseline.")
    else:
        print("RESULT: Perplexity is competitive.")

if __name__ == "__main__":
    main()
