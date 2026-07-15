import zlib
import os
import argparse
import sys
sys.path.insert(0, "/repo")
from utils.PTDI import PowerEnhancedStableEstimator, MinStoreyEstimator, IMSEstimator, SelectivePrediction, out_calibrated_sampling
from utils.utils import seed_everything
from utils.tools import *
from tqdm import tqdm
import torch
import numpy as np
from peft import PeftConfig, PeftModel
from datasets import Dataset
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import pandas as pd
import glob
from collections import defaultdict

def flatten_dict(result_metrics, alphas, keys=None):
    flattened = {}
    for metric, values in result_metrics.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if keys:
                if key not in keys:
                    continue
            if np.isscalar(value):
                new_key = f"{metric}_{key}"
                flattened[new_key] = float(value)
                print(f"{new_key}: {value}")
                continue
            for idx, alpha in enumerate(alphas):
                new_key = f"{metric}_{key}_alpha_{alpha}"
                flattened[new_key] = float(value[idx])
                print(f"{new_key}: {value[idx]}")
    return flattened

def to_numpy(q_hat):
    if isinstance(q_hat, torch.Tensor):
        return q_hat.detach().cpu().numpy()
    elif isinstance(q_hat, np.ndarray):
        return q_hat
    else:
        return np.array(q_hat)

def convert_huggingface_data_to_lists_by_label(dataset):
    label_0_data = []
    label_1_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        if ex["label"] == 0:
            label_0_data.append(ex)
        else:
            label_1_data.append(ex)
    return label_1_data, label_0_data

def load_model(args, fine_tuning):
    if not fine_tuning:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, return_dict=True, device_map="auto",
            torch_dtype=torch.float16
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model_path = args.model_path
    elif fine_tuning:
        config = PeftConfig.from_pretrained(args.fine_tuned_para)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, device_map="auto"
        )
        lora_model = PeftModel.from_pretrained(model, args.fine_tuned_para)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        model = lora_model
        model.eval()
        model_path = args.fine_tuned_para
    print("model path: ", model_path)
    return model, tokenizer

def compute_p_values(eval_scores, cal_scores, high_score_is_positive=True):
    n_exp, n_samples = eval_scores.shape
    n_cal = cal_scores.shape[1]
    cal_sorted = np.sort(cal_scores, axis=1)
    p_values = np.zeros((n_exp, n_samples), dtype=np.float64)
    for i in range(n_exp):
        row_cal = cal_sorted[i]
        row_eval = eval_scores[i]
        if high_score_is_positive:
            idx = np.searchsorted(row_cal, row_eval, side="left")
            p_values[i] = (n_cal - idx + 1.0) / (n_cal + 1.0)
        else:
            idx = np.searchsorted(row_cal, row_eval, side="right")
            p_values[i] = (idx + 1.0) / (n_cal + 1.0)
    return np.clip(p_values, 1e-10, 1.0 - 1e-10)

def inference(model, tokenizer, sentence, example):
    pred = {}
    p1, all_prob, p1_likelihood, mu_vocab, sigma_vocab = calculatePerplexity(
        sentence, model, tokenizer, gpu=model.device, return_vocab_stats=True
    )
    p_lower, _, p_lower_likelihood = calculatePerplexity(
        sentence.lower(), model, tokenizer, gpu=model.device
    )
    pred["ppl"] = p1
    pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
    zlib_entropy = len(zlib.compress(bytes(sentence, "utf-8")))
    pred["ppl/zlib"] = np.log(p1) / zlib_entropy
    
    # Vanilla Min-K% (keep for backward compatibility)
    ratio = 0.2
    k_length = int(len(all_prob) * ratio)
    topk_prob = np.sort(all_prob)[:k_length]
    pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()
    
    # Min-K%++: normalize each token's log-prob by vocab mean/std
    # s_i = (log_p(x_i) - mu_i) / sigma_i, then bottom-20% mean
    mu_arr = np.array(mu_vocab)
    sigma_arr = np.array(sigma_vocab) + 1e-8  # avoid div by zero
    all_prob_arr = np.array(all_prob)
    normalized_scores = (all_prob_arr - mu_arr) / sigma_arr
    k_length_pp = int(len(normalized_scores) * ratio)
    topk_norm = np.sort(normalized_scores)[:k_length_pp]
    pred[f"Min_{ratio*100}% Prob_PlusPlus"] = -np.mean(topk_norm).item()
    
    return pred

def get_dataset_local(args):
    """Load WikiMIA from local parquet files."""
    all_data = []
    text_lens = [32, 64, 128, 256]
    for tl in text_lens:
        pattern = f"/datasets/WikiMIA/WikiMIA_length{tl}-*.parquet"
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No parquet file found for WikiMIA_length{tl} at {pattern}")
        df = pd.read_parquet(files[0])
        ds = Dataset.from_pandas(df)
        all_data.extend(ds)
    
    member, non_member = convert_huggingface_data_to_lists_by_label(all_data)
    print(f"Loaded {len(member)} members, {len(non_member)} non-members")
    return member, non_member

def perform_ptdi(member_score_value, non_member_score_value, metrics, target_fdrs, seed=42):
    result_metrics = {}
    results_total = {}
    target_fdrs = to_numpy(target_fdrs)
    for key in metrics:
        certain_member_score_value = -to_numpy(member_score_value[key])
        certrain_non_member_score_value = -to_numpy(non_member_score_value[key])
        selector = SelectivePrediction()
        member_matrix, non_member_matrix, cal_matrix = out_calibrated_sampling(
            member_score=certain_member_score_value.squeeze(),
            non_member_score=certrain_non_member_score_value.squeeze(),
            seed=seed
        )
        total_score = np.concatenate([member_matrix, non_member_matrix], axis=1)
        p_values = compute_p_values(total_score, cal_matrix)
        estimator = IMSEstimator()
        estimate_pi = estimator.estimate_pi0_and_gamma(p_values)
        result_metrics[key], results_total[key] = selector.calculate_average_results_from_matrix(
            mem_p_score_matrix=-member_matrix,
            non_mem_p_score_matrix=-non_member_matrix,
            cal_p_score_matrix=-cal_matrix,
            target_fdrs=target_fdrs,
            scale_ratio=estimate_pi
        )
    return result_metrics

def prepare_score_dict(data, args=None, cache_dir="score_cache", overwrite=False):
    os.makedirs(cache_dir, exist_ok=True)
    model_name = args.model_path.replace("/", "_")
    dataset_name = args.dataset_name
    cache_file = os.path.join(cache_dir, f"scores_{dataset_name}_{model_name}.pkl")
    
    if os.path.exists(cache_file) and not overwrite:
        print(f"[Cache] Loading cached scores from {cache_file}")
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        return cache["member"], cache["non_member"]
    
    print("[Cache] No valid cache found. Running inference...")
    model, tokenizer = load_model(args, fine_tuning=False)
    
    member_score_dict = defaultdict(list)
    non_member_score_dict = defaultdict(list)
    
    member, non_member = data
    
    print("Processing member data...")
    for example in tqdm(member):
        text = example["input"]
        scores = inference(model, tokenizer, text, example)
        for metric, score in scores.items():
            member_score_dict[metric].append(score)
    
    print("\nProcessing non-member data...")
    for example in tqdm(non_member):
        text = example["input"]
        scores = inference(model, tokenizer, text, example)
        for metric, score in scores.items():
            non_member_score_dict[metric].append(score)
    
    cache = {
        "member": dict(member_score_dict),
        "non_member": dict(non_member_score_dict),
    }
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)
    
    print(f"[Cache] Scores saved to {cache_file}")
    return member_score_dict, non_member_score_dict

def calculatePerplexity(sentence, model, tokenizer, gpu, return_vocab_stats=False):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    all_mu = []
    all_sigma = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
        if return_vocab_stats:
            vocab_probs = probabilities[0, i, :]  # full vocab distribution at position i
            mu = vocab_probs.mean().item()
            sigma = vocab_probs.std().item()
            all_mu.append(mu)
            all_sigma.append(sigma)
    if return_vocab_stats:
        return torch.exp(loss).item(), all_prob, loss.item(), all_mu, all_sigma
    return torch.exp(loss).item(), all_prob, loss.item()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset_name", type=str, default="WikiMIA")
    parser.add_argument("--model_path", type=str, default="/models/pythia-6.9b")
    parser.add_argument("--fine_tuned_para", default="")
    parser.add_argument("--dataset", type=str, default="WikiMIA")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--len", type=int, default=32)
    parser.add_argument("--train_size", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    
    # Handle proxy environment variables
    for key in list(os.environ.keys()):
        if "proxy" in key.lower():
            del os.environ[key]
    
    # Set HF endpoint for model download
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HOME"] = "/autosota_cache/hf"
    
    # Load dataset from local parquet files
    data = get_dataset_local(args)
    member_score_dict, non_member_score_dict = prepare_score_dict(data, args)
    
    # --- Multi-Score Rank-Based Fusion (Idea 6153-A6) ---
    # Fuse multiple detection scores via rank normalization + averaging
    # Available scores: ppl, ppl/lowercase_ppl, ppl/zlib, Min_20.0% Prob
    fusion_metrics = ["Min_20.0% Prob", "ppl", "ppl/lowercase_ppl", "ppl/zlib"]
    
    # Gather member and non-member scores for each metric
    n_members = len(member_score_dict["Min_20.0% Prob"])
    n_nonmembers = len(non_member_score_dict["Min_20.0% Prob"])
    
    fused_member = np.zeros(n_members)
    fused_nonmember = np.zeros(n_nonmembers)
    n_valid = 0
    
    for metric in fusion_metrics:
        if metric in member_score_dict and metric in non_member_score_dict:
            mem_vals = np.array(member_score_dict[metric])
            nonmem_vals = np.array(non_member_score_dict[metric])
            
            # Combine all scores for ranking
            all_vals = np.concatenate([mem_vals, nonmem_vals])
            
            # Rank-normalize: higher rank = more member-like
            # For Min_20.0% Prob: lower score = more member-like (less surprising)
            # For ppl: lower score = more member-like
            # For ppl/zlib: lower = more member-like
            # For ppl/lowercase_ppl: higher = more member-like
            if metric == "ppl/lowercase_ppl":
                # Higher = more member-like, so rank ascending = member-like gets high rank
                ranks = np.argsort(np.argsort(all_vals)).astype(float)
            else:
                # Lower = more member-like, so rank descending
                ranks = np.argsort(np.argsort(-all_vals)).astype(float)
            
            # Normalize to [0, 1]
            ranks_norm = ranks / (len(all_vals) - 1) if len(all_vals) > 1 else ranks
            
            fused_member += ranks_norm[:n_members]
            fused_nonmember += ranks_norm[n_members:]
            n_valid += 1
    
    if n_valid > 0:
        fused_member /= n_valid
        fused_nonmember /= n_valid
        
        # Fused score: higher = more member-like (same direction as negated Min-K%)
        member_score_dict["Fused_Rank"] = fused_member.tolist()
        non_member_score_dict["Fused_Rank"] = fused_nonmember.tolist()
        print(f"[Fusion] Created Fused_Rank score from {n_valid} metrics ({n_members} members, {n_nonmembers} non-members)")
    
    target_fdrs = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = perform_ptdi(member_score_dict, non_member_score_dict, metrics=["Min_20.0% Prob", "Fused_Rank"], target_fdrs=target_fdrs, seed=args.seed)
    save_dict = flatten_dict(result, target_fdrs, keys=["FDP", "Power"])
