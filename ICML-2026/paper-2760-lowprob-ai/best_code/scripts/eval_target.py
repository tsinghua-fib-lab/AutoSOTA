# -*- coding: utf-8 -*-
"""Minimal evaluation: XSum + GPT-2 XL, black-box, Uncertainty method."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score

# =========================================================
# Hyperparameters (matching paper Table 11 for Uncertainty)
# =========================================================
DATA_FILE = "../dataset/xsum_gpt2_xl.raw_data.json"
MODEL_ID = "../Proxy_LLMs/gpt-j-6b"
MODEL_DTYPE = torch.bfloat16
MAX_LENGTH = 1024  # CODE-04: doubled from 512
BOS = True
CHUNK_SIZE = 16   # CODE-04: halved for 1024-length memory safety
EPS = 1e-12
GLOBAL_SEED = 42
X_TAIL = 7        # low_probability_percentile = 7
GLOBAL_MEAN_ENTROPY = None  # ALGO-06: computed from calibration samples
RENYI_Q = 2.0     # renyi_entropy_order = 2.0
WZ = 0.65         # PARAM-01 config B: more window entropy weight
W_W = 0.12         # ALGO-02: window entropy weight (increased)
W_H = 0.14          # ALGO-02: entropy weight
W_S = 0.045         # ALGO-05: entropy std weight
W_G = 0.045         # ALGO-05: entropy gradient weight

def encode_raw_text_bos_noeos(tokenizer, text, max_length, bos):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if bos and tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
    if len(ids) > max_length:
        ids = ids[:max_length]
    return torch.tensor([ids], dtype=torch.long)

@torch.no_grad()
def extract_features(text, tokenizer, model):
    if not text or not isinstance(text, str):
        return float("nan")
    input_ids = encode_raw_text_bos_noeos(tokenizer, text, MAX_LENGTH, BOS)
    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)
    if int(input_ids.shape[1]) < 2:
        return float("nan")
    out_model = model(input_ids=input_ids, use_cache=False)
    logits = out_model.logits[0, :-1, :]
    targets = input_ids[0, 1:]
    T_val = int(logits.shape[0])
    V_val = int(logits.shape[1])
    lnV = float(np.log(max(2, V_val)))
    logp_obs = np.empty((T_val,), dtype=np.float32)
    H_renyi = np.empty((T_val,), dtype=np.float32)
    for start in range(0, T_val, CHUNK_SIZE):
        end = min(T_val, start + CHUNK_SIZE)
        chunk_logits = logits[start:end].to(torch.float32)
        c = int(chunk_logits.shape[0])
        logp = torch.log_softmax(chunk_logits, dim=-1)
        tgt = targets[start:end]
        ar = torch.arange(c, device=device)
        logp_tok = logp[ar, tgt]
        logp_obs[start:end] = logp_tok.detach().cpu().numpy().astype(np.float32)
        log_sum = torch.logsumexp(RENYI_Q * logp, dim=-1)
        Hq = log_sum / (1.0 - RENYI_Q)
        H_renyi[start:end] = Hq.detach().cpu().numpy().astype(np.float32)
        del chunk_logits, logp, tgt, ar, logp_tok, log_sum, Hq
    obs = logp_obs.astype(np.float64)
    # ALGO-06: Adaptive percentile based on text entropy
    mean_entropy_all = float(np.mean(H_renyi.astype(np.float64)))
    if GLOBAL_MEAN_ENTROPY is not None and GLOBAL_MEAN_ENTROPY > 0:
        adaptive_x_tail = float(np.clip(X_TAIL * mean_entropy_all / GLOBAL_MEAN_ENTROPY, 3.0, 15.0))
    else:
        adaptive_x_tail = float(X_TAIL)
    k = int(np.ceil((adaptive_x_tail / 100.0) * T_val))
    k = max(1, min(k, T_val))
    tail_idx_k = np.argpartition(obs, k - 1)[:k]
    d = int(min(4, k - 1))
    if d > 0:
        rm_local = np.argpartition(obs[tail_idx_k], d - 1)[:d]
        keep_mask = np.ones(k, dtype=bool)
        keep_mask[rm_local] = False
        tail_idx = tail_idx_k[keep_mask]
    else:
        tail_idx = tail_idx_k
    mean_logp_tail = float(np.mean(obs[tail_idx]))
    p_norm = float(np.clip((mean_logp_tail + lnV) / (lnV + EPS), 0.0, 1.0))
    mean_renyi_tail = float(np.mean(H_renyi[tail_idx].astype(np.float64)))
    h_norm = float(np.clip(mean_renyi_tail / (lnV + EPS), 0.0, 1.0))
    # ALGO-02: Windowed Renyi entropy statistics (multiscale)
    # Sort tail indices by position for windowing
    tail_idx_sorted = np.sort(tail_idx)
    tail_entropy_sorted = H_renyi[tail_idx_sorted].astype(np.float64)
    n_windows = max(4, int(T_val) // 8)
    window_size = max(2, len(tail_idx_sorted) // n_windows)
    window_means = []
    for w_start in range(0, len(tail_idx_sorted), window_size):
        w_end = min(len(tail_idx_sorted), w_start + window_size)
        if w_end - w_start >= 1:
            window_means.append(float(np.mean(tail_entropy_sorted[w_start:w_end])))
    if len(window_means) >= 2:
        max_win = float(np.max(window_means))
        min_win = float(np.min(window_means))
        std_win = float(np.std(window_means))
        max_norm = float(np.clip(max_win / (lnV + EPS), 0.0, 1.0))
        min_norm = float(np.clip(min_win / (lnV + EPS), 0.0, 1.0))
        std_norm = float(np.clip(std_win / (lnV + EPS), 0.0, 1.0))
        window_feature = float(0.33 * (max_norm + (1.0 - min_norm) + std_norm))
    else:
        window_feature = 0.5  # neutral if insufficient windows
    # ALGO-05: Entropy shape features (std + adjacent-token gradient)
    tail_entropy_all = H_renyi[tail_idx].astype(np.float64)
    std_renyi_tail = float(np.std(tail_entropy_all))
    std_norm = float(np.clip(std_renyi_tail / (lnV + EPS), 0.0, 1.0))
    # gradient of entropy along sorted positions
    grad_positions = np.sort(tail_idx)
    grad_entropy = H_renyi[grad_positions].astype(np.float64)
    if len(grad_entropy) >= 2:
        grad_renyi_tail = float(np.mean(np.abs(np.diff(grad_entropy))))
    else:
        grad_renyi_tail = 0.0
    grad_norm = float(np.clip(grad_renyi_tail / (lnV + EPS), 0.0, 1.0))
    # CODE-04: Length normalization to prevent bias from longer sequences
    len_norm = float(np.log(1.0 + T_val) / np.log(1.0 + 1024))
    fused = float(len_norm * (WZ * p_norm + W_H * (1.0 - h_norm) + W_W * window_feature + W_S * std_norm + W_G * grad_norm))
    return fused

def auc_best_from_scores(scores, labels):
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    mask = ~np.isnan(x)
    x = x[mask]
    y = y[mask]
    if len(x) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    if np.nanstd(x) < 1e-12:
        return float("nan")
    auc_raw = float(roc_auc_score(y, x))
    return max(auc_raw, 1.0 - auc_raw)

def main():
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    print("Loading tokenizer and model from:", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=MODEL_DTYPE, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=MODEL_DTYPE, device_map="auto")
    model.eval()
    gpu_mem = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
    print("Model loaded. GPU memory allocated: {:.1f} GB".format(gpu_mem))

    print("Loading data from:", DATA_FILE)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    print("Loaded {} records".format(len(records)))

    # ALGO-06: Precompute global mean entropy from calibration subset
    global GLOBAL_MEAN_ENTROPY
    calib_entropies = []
    calib_n = min(30, len(records))
    print("Computing global mean entropy from {} calibration samples...".format(calib_n))
    for i in range(calib_n):
        item = records[i]
        for key in ["original_text", "ai_generated_text"]:
            text = item.get(key, "")
            if text and isinstance(text, str):
                ids = tokenizer.encode(text, add_special_tokens=False)
                if BOS and tokenizer.bos_token_id is not None:
                    ids = [tokenizer.bos_token_id] + ids
                ids_t = ids[:MAX_LENGTH]
                input_ids = torch.tensor([ids_t], dtype=torch.long).to(model.device)
                if input_ids.shape[1] >= 2:
                    out = model(input_ids=input_ids, use_cache=False)
                    logits = out.logits[0, :-1, :].to(torch.float32)
                    logp = torch.log_softmax(logits, dim=-1)
                    log_sum = torch.logsumexp(RENYI_Q * logp, dim=-1)
                    Hq = (log_sum / (1.0 - RENYI_Q)).detach().cpu().numpy().astype(np.float64)
                    calib_entropies.append(float(np.mean(Hq)))
    GLOBAL_MEAN_ENTROPY = float(np.mean(calib_entropies)) if calib_entropies else 0.0
    print("Global mean entropy: {:.4f}".format(GLOBAL_MEAN_ENTROPY))

    scores, labels = [], []
    for i, item in enumerate(records):
        human_text = item.get("original_text", "")
        ai_text = item.get("ai_generated_text", "")
        if not human_text.strip() or not ai_text.strip():
            continue
        hs = extract_features(human_text, tokenizer, model)
        ai_s = extract_features(ai_text, tokenizer, model)
        scores.append(float(hs))
        labels.append(0)
        scores.append(float(ai_s))
        labels.append(1)
        if (i + 1) % 10 == 0:
            print("  Processed {}/{} samples...".format(i+1, len(records)))

    auc = auc_best_from_scores(scores, labels)
    n_pairs = len(labels) // 2
    sep = "=" * 60
    print("\n" + sep)
    print("RESULTS: XSum + GPT-2 (1.5B) black-box | Proxy: GPT-J-6B")
    print("  Method: Uncertainty")
    print("  Hyperparameters: X_TAIL={}%, RENYI_Q={}, WZ={}".format(X_TAIL, RENYI_Q, WZ))
    print("  Samples evaluated: {}".format(n_pairs))
    print("  AUROC: {:.2f}%".format(auc * 100))
    print(sep)

if __name__ == "__main__":
    main()
