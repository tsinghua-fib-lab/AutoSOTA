import argparse
import json
import os
import sys

import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from paraphraser.watermark_attack import WatermarkAttack
from watermark.alimark import AliMark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watermark_algorithm", type=str, default="AliMark")
    parser.add_argument("--watermark_model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--watermark_embedder", type=str, default="all-mpnet-base-v2")
    parser.add_argument('--watermark_embedding_dim', type=int, default=768)
    parser.add_argument('--watermark_block_size', type=int, default=8)
    parser.add_argument("--watermark_num_next_sentence_candidates", type=int, default=64)
    parser.add_argument("--min_new_sentences", type=int, default=12)
    parser.add_argument("--dataset_name", type=str, default="c4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    WATERMARK_BLOCK_SIZE = args.watermark_block_size
    DATASET_NAME = args.dataset_name
    
    # Only local attacks (skip DIPPER and GPT-3.5 which need downloads/API keys)
    ATTACK_ALGORITHM_LIST = [
        "pegasus_paraphrase_no_bigram",
        "parrot_paraphrase_no_bigram",
    ]
    
    # Also include probing attacks (no model needed)
    PROBING_ATTACKS = [
        "insert_10", "insert_30", "insert_50",
        "delete_10", "delete_30", "delete_50",
        "reorder_10", "reorder_30", "reorder_50",
    ]
    
    MAX_ATTACK_TRY = 3

    GENERATION_RESULT_DIR = f"_result/generation/block_size_{WATERMARK_BLOCK_SIZE}"
    ATTACK_RESULT_DIR = f"_result/attack/block_size_{WATERMARK_BLOCK_SIZE}"
    os.makedirs(ATTACK_RESULT_DIR, exist_ok=True)
    
    WATERMARK_MODEL_NAME = args.watermark_model
    WATERMARK_ALGORITHM_NAME = args.watermark_algorithm
    
    GENERATION_RESULT_FILE_NAME = os.path.join(
        GENERATION_RESULT_DIR,
        f"{DATASET_NAME}_{WATERMARK_ALGORITHM_NAME}_{WATERMARK_MODEL_NAME.replace('/', '_')}.json"
    )
    ATTACK_RESULT_FILE_NAME = os.path.join(
        ATTACK_RESULT_DIR,
        f"{DATASET_NAME}_{WATERMARK_ALGORITHM_NAME}_{WATERMARK_MODEL_NAME.replace('/', '_')}.json"
    )

    if not os.path.exists(GENERATION_RESULT_FILE_NAME):
        raise FileNotFoundError(f"Generation result file not found: {GENERATION_RESULT_FILE_NAME}")
    
    df_generation_results = pd.read_json(GENERATION_RESULT_FILE_NAME, orient="index")
    print(f"Loaded {len(df_generation_results)} generation results")

    if not os.path.exists(ATTACK_RESULT_FILE_NAME):
        df_attack_results = df_generation_results.copy()
    else:
        df_attack_results = pd.read_json(ATTACK_RESULT_FILE_NAME, orient="index")
        print(f"Resuming from {len(df_attack_results)} existing attack results")

    all_attacks = ATTACK_ALGORITHM_LIST + PROBING_ATTACKS
    for attack_algorithm in all_attacks:
        column_name = f"{attack_algorithm}_result"
        if column_name not in df_attack_results.columns:
            df_attack_results[column_name] = None

    # Init watermark (detection only, no LLM)
    watermark = AliMark(args, load_llm=False)

    # Init attack WITHOUT DIPPER (load_dipper_paraphraser=False)
    print("Initializing paraphraser (Pegasus + Parrot, no DIPPER)...")
    watermark_attack = WatermarkAttack(
        watermark_model_name=WATERMARK_MODEL_NAME, 
        load_dipper_paraphraser=False, 
        load_semstamp_paraphraser=True,
    )
    print("Paraphraser initialized.")

    # Attack the watermarked text
    for idx, row in tqdm(df_attack_results.iterrows(), total=len(df_attack_results), desc="Attacking"):
        question = row['question']
        watermarked_text = row["watermarked_result"]["text"]

        for attack_algorithm in all_attacks:
            column_name = f"{attack_algorithm}_result"

            if row[column_name] is not None and not pd.isna(row[column_name]):
                continue

            attack_try = 0
            attacked_text = None
            while attack_try < MAX_ATTACK_TRY:
                attack_try += 1
                try:
                    if attack_algorithm == "pegasus_paraphrase_no_bigram":
                        attacked_text = watermark_attack.pegasus_paraphrase_attack(watermarked_text, bigram=False)
                    elif attack_algorithm == "parrot_paraphrase_no_bigram":
                        attacked_text = watermark_attack.parrot_paraphrase_attack(watermarked_text, bigram=False)
                    elif attack_algorithm.startswith(("insert_", "delete_", "reorder_")):
                        action, rate_str = attack_algorithm.split("_")
                        rate = int(rate_str) / 100.0
                        attack_funcs = {
                            "insert": watermark_attack.probing_insert,
                            "delete": watermark_attack.probing_delete,
                            "reorder": watermark_attack.probing_reorder,
                        }
                        attacked_text = attack_funcs[action](watermarked_text, rate=rate)
                    else:
                        print(f"Skipping unknown attack: {attack_algorithm}")
                        break
                    
                    if attacked_text is not None and attacked_text != "" and len(attacked_text) > 10:
                        break
                except Exception as e:
                    print(f"Error during {attack_algorithm} (try {attack_try}): {e}")
                    import traceback
                    traceback.print_exc()

            if attacked_text is None or attacked_text == "":
                print(f"Attack {attack_algorithm} failed after {MAX_ATTACK_TRY} tries for sample {idx}")
                continue
            
            attack_result = {'text': attacked_text}
            df_attack_results.at[idx, column_name] = attack_result
            df_attack_results.to_json(ATTACK_RESULT_FILE_NAME, orient="index", indent=4)
        
        print(f"Sample {idx} done: {', '.join(a for a in all_attacks if df_attack_results.at[idx, f'{a}_result'] is not None and not pd.isna(df_attack_results.at[idx, f'{a}_result']))}")

    print(f"Attacks complete! Results saved to {ATTACK_RESULT_FILE_NAME}")
