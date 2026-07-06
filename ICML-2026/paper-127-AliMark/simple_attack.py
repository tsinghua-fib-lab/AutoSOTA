#!/usr/bin/env python3
"""Minimal Pegasus-only attack for AliMark. Bypasses all complex dependencies."""
import os, sys, json, argparse
sys.path.insert(0, "/repo")

for k in ["http_proxy","HTTP_PROXY","https_proxy","HTTPS_PROXY","all_proxy","ALL_PROXY","no_proxy","NO_PROXY"]:
    os.environ.pop(k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/autosota_cache/hf"
os.environ["HF_HUB_CACHE"] = "/autosota_cache/hf/hub"
os.environ["TRANSFORMERS_CACHE"] = "/autosota_cache/hf"

import torch
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watermark_block_size", type=int, default=8)
    parser.add_argument("--dataset_name", type=str, default="c4")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    BLOCK_SIZE = args.watermark_block_size
    DATASET = args.dataset_name
    DEVICE = args.device

    GEN_DIR = f"_result/generation/block_size_{BLOCK_SIZE}"
    ATK_DIR = f"_result/attack/block_size_{BLOCK_SIZE}"
    os.makedirs(ATK_DIR, exist_ok=True)

    GEN_FILE = os.path.join(GEN_DIR, f"{DATASET}_AliMark_facebook_opt-1.3b.json")
    ATK_FILE = os.path.join(ATK_DIR, f"{DATASET}_AliMark_facebook_opt-1.3b.json")

    df_gen = pd.read_json(GEN_FILE, orient="index")
    print(f"Loaded {len(df_gen)} generation results")

    # Load Pegasus model
    print("Loading Pegasus paraphraser...")
    model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase").to(DEVICE)
    tokenizer = PegasusTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
    print("Pegasus loaded")

    # Load or create attack results
    if os.path.exists(ATK_FILE):
        df_atk = pd.read_json(ATK_FILE, orient="index")
        print(f"Loaded existing attack results ({len(df_atk)} rows)")
    else:
        df_atk = df_gen.copy()
        print(f"Created new attack results frame ({len(df_atk)} rows)")

    attack_name = "pegasus_paraphrase_no_bigram"
    col_name = f"{attack_name}_result"

    if col_name not in df_atk.columns:
        df_atk[col_name] = None

    for idx, row in tqdm(df_atk.iterrows(), total=len(df_atk)):
        # Skip if already done
        if row.get(col_name) is not None and not pd.isna(row[col_name]):
            continue

        watermarked_text = row["watermarked_result"]["text"]
        sentences = sent_tokenize(watermarked_text)
        paras = []
        for sent in sentences:
            batch = tokenizer([sent], truncation=True, padding="longest",
                            return_tensors="pt", max_length=60).to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(**batch, max_length=60, num_beams=10,
                                        num_return_sequences=1, temperature=2.0,
                                        do_sample=True, repetition_penalty=1.03)
            para = tokenizer.decode(outputs[0], skip_special_tokens=True)
            para = para[0].upper() + para[1:] if para else para
            if para and para[-1] not in ".!?":
                para += "."
            paras.append(para)

        attacked_text = " ".join(paras)
        df_atk.at[idx, col_name] = {"text": attacked_text}
        df_atk.to_json(ATK_FILE, orient="index", indent=4)

    print(f"Pegasus attacks complete. Saved to {ATK_FILE}")

if __name__ == "__main__":
    main()
