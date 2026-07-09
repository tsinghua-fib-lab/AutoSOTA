# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import csv
import json
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


# =========================================================
# Hyperparameters / paths (EDIT HERE ONLY)
# =========================================================
DATA_DIR = "../dataset"
OUTPUT_ROOT = "../uncertainty_black_experiment"

# This is the scoring/proxy LM used to compute logP and Renyi entropy.
MODEL_ID = "../Proxy_LLMs/gpt-j-6b"
MODEL_DTYPE = torch.bfloat16

MAX_LENGTH = 512
BOS = True
CHUNK_SIZE = 32
EPS = 1e-12
GLOBAL_SEED = 42

# Fixed parameters: no search
X_TAIL = 7
RENYI_Q = 2.0
WZ = 0.8

# =========================================================
# Dataset/model name mapping based on your actual filenames
# =========================================================
DATASET_KEY_TO_NAME: Dict[str, str] = {
    "xsum": "XSum",
    "writing": "WritingPrompts",
    "reddit": "Reddit",
    "squad": "SQuAD",
}

# key: model part in filename
# value: column name in output.csv
MODEL_KEY_TO_COL: Dict[str, str] = {
    "gpt2_xl": "GPT-2",
    "gptj_6b": "GPT-J-6",
    "gptneo_2.7b": "Neo-2.7",
    "opt_2.7b": "OPT-2.7",
    "llama1_13b": "Llama-13",
    "llama2_13b": "Llama2-13",
    "llama3_8b": "Llama3-8",
    "opt_13b": "OPT-13",
    "bloom_7b": "Bloom-7.1",
    "falcon_7b": "Falcon-7",
    "gemma_7b": "Gemma-7",
    "phi2": "Phi2-2.7",
    "gpt4turbo": "GPT-4-T",
    "gpt4o": "GPT-4o",
    "claude3haiku": "Claude3-Haiku",
}

# These are the only columns/rows shown in final output.csv.
# Other datasets/models are still computed, but excluded from the final table.
OUTPUT_MODEL_COLUMNS = [
    "GPT-2",
    "Neo-2.7",
    "OPT-2.7",
    "Llama-13",
    "Llama2-13",
    "Llama3-8",
    "OPT-13",
    "Bloom-7.1",
    "Falcon-7",
    "Gemma-7",
    "Phi2-2.7",
    "GPT-4-T",
]

OUTPUT_DATASET_ORDER = ["XSum", "WritingPrompts", "Reddit"]


# =========================================================
# General utils
# =========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_json_files(data_dir: str) -> List[str]:
    return sorted(
        os.path.join(data_dir, fn)
        for fn in os.listdir(data_dir)
        if fn.endswith(".json")
    )


def q_tag(q: float) -> str:
    return str(q).replace(".", "p")


def feature_name() -> str:
    return f"F_es{X_TAIL}_rq{q_tag(RENYI_Q)}_wz{q_tag(WZ)}"


def strip_known_suffix(filename: str) -> str:
    """Remove your known suffix first, then fall back to plain .json."""
    if filename.endswith(".raw_data.json"):
        return filename[: -len(".raw_data.json")]
    if filename.endswith(".json"):
        return filename[: -len(".json")]
    return filename


def parse_data_filename(json_path: str) -> Optional[Tuple[str, str]]:
    """
    Parse filename like:
      reddit_gptj_6b.raw_data.json

    Return:
      (dataset_name, model_column)
    """
    filename = os.path.basename(json_path)
    stem = strip_known_suffix(filename)

    if "_" not in stem:
        return None

    dataset_key, model_key = stem.split("_", 1)

    dataset_name = DATASET_KEY_TO_NAME.get(dataset_key)
    model_col = MODEL_KEY_TO_COL.get(model_key)

    if dataset_name is None or model_col is None:
        return None

    return dataset_name, model_col


def load_json_records(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        records = obj
    elif isinstance(obj, dict):
        records = None
        for key in ["data", "records", "samples", "items", "examples"]:
            if isinstance(obj.get(key), list):
                records = obj[key]
                break
        if records is None:
            raise ValueError(
                f"Cannot find a list of records in {json_path}. "
                "Expected a list, or a dict containing data/records/samples/items/examples."
            )
    else:
        raise ValueError(f"Unsupported JSON top-level type in {json_path}: {type(obj)}")

    return [r for r in records if isinstance(r, dict)]


def get_text_pair(item: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Return (human_text, ai_text) from fixed JSON fields."""
    human_text = item.get("original_text", "")
    ai_text = item.get("ai_generated_text", "")

    if not isinstance(human_text, str) or not human_text.strip():
        return None
    if not isinstance(ai_text, str) or not ai_text.strip():
        return None

    return human_text, ai_text


# =========================================================
# Encoding
# =========================================================
def encode_raw_text_bos_noeos(tokenizer, text: str, max_length: int, bos: bool) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if bos and tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
    if len(ids) > max_length:
        ids = ids[:max_length]
    return torch.tensor([ids], dtype=torch.long)


# =========================================================
# Core feature extraction: one fixed parameter setting only, no cache
# =========================================================
@torch.no_grad()
def extract_features_one_text_tailmean_logp_renyi(
    text: str,
    tokenizer,
    model,
) -> Dict[str, float]:
    """
    Computes one fixed fusion feature:

      P_norm = clip((mean_logp_tail + lnV) / lnV, 0, 1)
      H_norm = clip(mean_renyi_tail / lnV, 0, 1)
      F = WZ * P_norm + (1 - WZ) * (1 - H_norm)

    Tail = bottom X_TAIL% positions by observed-token logP.
    """
    fname = feature_name()

    if not text or not isinstance(text, str):
        return {"T": 0.0, fname: np.nan}

    input_ids = encode_raw_text_bos_noeos(tokenizer, text, MAX_LENGTH, BOS)
    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)

    if int(input_ids.shape[1]) < 2:
        return {"T": 0.0, fname: np.nan}

    out_model = model(input_ids=input_ids, use_cache=False)
    logits = out_model.logits[0, :-1, :]  # (T, V)
    targets = input_ids[0, 1:]            # (T,)
    T = int(logits.shape[0])
    V = int(logits.shape[1])
    lnV = float(np.log(max(2, V)))

    logp_obs = np.empty((T,), dtype=np.float32)
    H_renyi = np.empty((T,), dtype=np.float32)

    for start in range(0, T, CHUNK_SIZE):
        end = min(T, start + CHUNK_SIZE)
        chunk_logits = logits[start:end].to(torch.float32)  # (c, V)
        c = int(chunk_logits.shape[0])

        logp = torch.log_softmax(chunk_logits, dim=-1)      # (c, V)

        # Observed-token log probability.
        tgt = targets[start:end]
        ar = torch.arange(c, device=device)
        logp_tok = logp[ar, tgt]
        logp_obs[start:end] = logp_tok.detach().cpu().numpy().astype(np.float32)

        # Renyi entropy: H_q = 1/(1-q) * log sum_i p_i^q.
        log_sum = torch.logsumexp(RENYI_Q * logp, dim=-1)
        Hq = log_sum / (1.0 - RENYI_Q)
        H_renyi[start:end] = Hq.detach().cpu().numpy().astype(np.float32)

        del chunk_logits, logp, tgt, ar, logp_tok, log_sum, Hq

    obs = logp_obs.astype(np.float64)

    k = int(np.ceil((X_TAIL / 100.0) * T))
    k = max(1, min(k, T))

    # Bottom-X% positions: k smallest logP values.
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

    fused = float(WZ * p_norm + (1.0 - WZ) * (1.0 - h_norm))

    return {
        "T": float(T),
        f"P_es{X_TAIL}_logp_mean_raw": mean_logp_tail,
        f"P_es{X_TAIL}_logp_mean_norm": p_norm,
        f"H_renyi{q_tag(RENYI_Q)}_es{X_TAIL}_norm": h_norm,
        fname: fused,
    }


# =========================================================
# AUC computation
# =========================================================
def auc_best_from_scores(scores: List[float], labels: List[int]) -> float:
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


def evaluate_one_json_file(
    records: List[Dict[str, Any]],
    dataset_name: str,
    model_col: str,
    tokenizer,
    model,
) -> Tuple[float, int]:
    scores: List[float] = []
    labels: List[int] = []
    fname = feature_name()

    desc = f"[{dataset_name} | {model_col}]"
    for item in tqdm(records, desc=desc, leave=False):
        pair = get_text_pair(item)
        if pair is None:
            continue

        human_text, ai_text = pair

        human_feats = extract_features_one_text_tailmean_logp_renyi(human_text, tokenizer, model)
        ai_feats = extract_features_one_text_tailmean_logp_renyi(ai_text, tokenizer, model)

        scores.append(float(human_feats.get(fname, np.nan)))
        labels.append(0)
        scores.append(float(ai_feats.get(fname, np.nan)))
        labels.append(1)

    auc_best = auc_best_from_scores(scores, labels)
    n_pairs = len(labels) // 2
    return auc_best, n_pairs


# =========================================================
# Output generation
# =========================================================
def format_auc_percent(v: float) -> str:
    if v is None or not np.isfinite(v):
        return ""
    return f"{v * 100.0:.2f}"


def ordered_datasets(dataset_names: List[str]) -> List[str]:
    seen = set(dataset_names)
    ordered = [d for d in OUTPUT_DATASET_ORDER if d in seen]
    ordered.extend(sorted(d for d in seen if d not in DATASET_ORDER))
    return ordered


def write_output_csv(all_scores: Dict[str, Dict[str, float]], output_csv: str) -> None:
    ensure_dir(os.path.dirname(output_csv))
    header = ["Dataset", *OUTPUT_MODEL_COLUMNS, "Avg."]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for dname in OUTPUT_DATASET_ORDER:
            row: Dict[str, str] = {"Dataset": dname}
            vals: List[float] = []

            for model_col in OUTPUT_MODEL_COLUMNS:
                auc = all_scores.get(dname, {}).get(model_col, float("nan"))
                row[model_col] = format_auc_percent(auc)
                if np.isfinite(auc):
                    vals.append(float(auc))

            avg = float(np.mean(vals)) if vals else float("nan")
            row["Avg."] = format_auc_percent(avg)
            writer.writerow(row)


def print_output_table(all_scores: Dict[str, Dict[str, float]]) -> None:
    header = ["Dataset", *OUTPUT_MODEL_COLUMNS, "Avg."]
    rows: List[List[str]] = []

    for dname in OUTPUT_DATASET_ORDER:
        vals: List[float] = []
        row = [dname]

        for model_col in OUTPUT_MODEL_COLUMNS:
            auc = all_scores.get(dname, {}).get(model_col, float("nan"))
            row.append(format_auc_percent(auc))
            if np.isfinite(auc):
                vals.append(float(auc))

        avg = float(np.mean(vals)) if vals else float("nan")
        row.append(format_auc_percent(avg))
        rows.append(row)

    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    print("Final output table:")
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def process_all_json_files(tokenizer, model) -> Dict[str, Dict[str, float]]:
    json_files = list_json_files(DATA_DIR)
    if not json_files:
        raise RuntimeError(f"No .json files found under: {DATA_DIR}")

    all_scores: Dict[str, Dict[str, float]] = {}

    skipped_files: List[str] = []

    for json_path in tqdm(json_files, desc="Processing JSON files"):
        parsed = parse_data_filename(json_path)
        if parsed is None:
            skipped_files.append(os.path.basename(json_path))
            continue

        dataset_name, model_col = parsed
        records = load_json_records(json_path)

        auc_best, n_pairs = evaluate_one_json_file(
            records=records,
            dataset_name=dataset_name,
            model_col=model_col,
            tokenizer=tokenizer,
            model=model,
        )

        if n_pairs == 0:
            skipped_files.append(os.path.basename(json_path))
            continue

        all_scores.setdefault(dataset_name, {})[model_col] = auc_best

    if skipped_files:
        print("[WARNING] Skipped files:")
        for fn in skipped_files:
            print(f"  - {fn}")

    return all_scores


# =========================================================
# Main
# =========================================================
def main() -> None:
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    ensure_dir(OUTPUT_ROOT)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    # `dtype` is used by newer transformers versions; fallback keeps compatibility with older versions.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=MODEL_DTYPE,
            device_map="auto",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=MODEL_DTYPE,
            device_map="auto",
        )

    model.eval()

    all_scores = process_all_json_files(tokenizer, model)

    output_csv = os.path.join(OUTPUT_ROOT, "output.csv")
    write_output_csv(all_scores, output_csv)
    print_output_table(all_scores)

    print("[OK] All done.")
    print("Final output saved to:", output_csv)


if __name__ == "__main__":
    main()