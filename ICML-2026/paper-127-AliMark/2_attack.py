import argparse
import json
import os

import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from paraphraser.watermark_attack import WatermarkAttack
from watermark.alimark import AliMark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watermark_algorithm",                        type=str,   default="AliMark")
    parser.add_argument("--watermark_model",                            type=str,   default="facebook/opt-1.3b")
    parser.add_argument("--watermark_embedder",                         type=str,   default="all-mpnet-base-v2")
    parser.add_argument('--watermark_embedding_dim',                    type=int,   default=768)
    parser.add_argument('--watermark_block_size',                       type=int,   default=4)
    parser.add_argument("--watermark_num_next_sentence_candidates",     type=int,   default=64)
    parser.add_argument("--min_new_sentences",                          type=int,   default=12)
    parser.add_argument("--dataset_name",                               type=str,   default="booksum")
    parser.add_argument("--vllm_gpu_mem_util",                          type=float, default=0.2)
    parser.add_argument("--device",                                     type=str,   default="cuda")
    parser.add_argument('--seed',                                       type=int,   default=42)
    args = parser.parse_args()
    
    WATERMARK_ALGORITHM_NAME = args.watermark_algorithm
    WATERMARK_MODEL_NAME = args.watermark_model
    WATERMARK_EMBEDDER = args.watermark_embedder
    WATERMARK_BLOCK_SIZE = args.watermark_block_size
    WATERMARK_NUM_NEXT_SENTENCE_CANDIDATES = args.watermark_num_next_sentence_candidates
    MIN_NEW_SENTENCES = args.min_new_sentences
    DATASET_NAME = args.dataset_name
    DEVICE = args.device

    ATTACK_ALGORITHM_LIST = [
        # Paraphrasing attack (main study)
        "pegasus_paraphrase_no_bigram",
        "parrot_paraphrase_no_bigram",
        # "dipper_all_paraphrase",  # Skipped: T5-XXL model not downloaded 
        # "gpt35_turbo_paraphrase",  # Skipped: no OPENROUTER_API_KEY

        # Probing attack (optional for adversarial robustness study)
        # "insert_10", "insert_20", "insert_30", "insert_40", "insert_50",
        # "delete_10", "delete_20", "delete_30", "delete_40", "delete_50",
        # "reorder_10", "reorder_20", "reorder_30", "reorder_40", "reorder_50",
        ]
    MAX_ATTACK_TRY = 5

    GENERATION_RESULT_DIR = f"_result/generation/block_size_{WATERMARK_BLOCK_SIZE}"
    ATTACK_RESULT_DIR = f"_result/attack/block_size_{WATERMARK_BLOCK_SIZE}"
    os.makedirs(ATTACK_RESULT_DIR, exist_ok=True)
    
    GENERATION_RESULT_FILE_NAME = os.path.join(
        GENERATION_RESULT_DIR,
        f"{DATASET_NAME}_{WATERMARK_ALGORITHM_NAME}_{WATERMARK_MODEL_NAME.replace('/', '_')}.json"
    )
    ATTACK_RESULT_FILE_NAME = os.path.join(
        ATTACK_RESULT_DIR,
        f"{DATASET_NAME}_{WATERMARK_ALGORITHM_NAME}_{WATERMARK_MODEL_NAME.replace('/', '_')}.json"
    )

    # Handle result file
    if not os.path.exists(GENERATION_RESULT_FILE_NAME):
        raise FileNotFoundError(f"Generation result file not found: {GENERATION_RESULT_FILE_NAME}")
    else:
        df_generation_results = pd.read_json(GENERATION_RESULT_FILE_NAME, orient="index")

    if not os.path.exists(ATTACK_RESULT_FILE_NAME):
        df_attack_results = df_generation_results.copy()
    else:
        df_attack_results = pd.read_json(ATTACK_RESULT_FILE_NAME, orient="index")

    for attack_algorithm in ATTACK_ALGORITHM_LIST:
            column_name = f"{attack_algorithm}_result"
            if not (column_name in df_attack_results.columns):
                df_attack_results[column_name] = None

    # Init watermark and LLM
    watermark = AliMark(args, load_llm=False)

    # Init attack
    watermark_attack = WatermarkAttack(
        watermark_model_name=WATERMARK_MODEL_NAME, 
        load_dipper_paraphraser=False,  # Patched: T5-XXL not available 
        load_semstamp_paraphraser=True,
        )

    # Attack the watermarked text
    for idx, row in tqdm(df_attack_results.iterrows(), total=len(df_attack_results)):
        question = row['question']
        watermarked_text = row["watermarked_result"]["text"]

        print("*************")
        print(f"= Question: {question}")
        print('\n')

        print(f"= Watermarked Text: {watermarked_text}")
        print('\n')

        # Attack and detect
        for attack_algorithm in ATTACK_ALGORITHM_LIST:
            column_name = f"{attack_algorithm}_result"

            attack_done = False
            if row[column_name] is not None and not pd.isna(row[column_name]):
                attack_done = True
                print(f"---{idx} Attack ({attack_algorithm}) already done---")
                continue

            print(f"---{idx} Attack ({attack_algorithm})---")
            if attack_done:
                attacked_text = row[column_name]['text']
            else:
                attack_try = 0
                attacked_text = None
                while attack_try < MAX_ATTACK_TRY:
                    attack_try += 1
                    print(f"Attack Try: {attack_try} ...")
                    try:
                        # Paraphrasing attack
                        if attack_algorithm == "dipper_all_paraphrase":
                            attacked_text = watermark_attack.dipper_paraphrase_attack(watermarked_text, mode="all")
                        elif attack_algorithm == "pegasus_paraphrase_no_bigram":
                            attacked_text = watermark_attack.pegasus_paraphrase_attack(watermarked_text, bigram=False)
                        elif attack_algorithm == "parrot_paraphrase_no_bigram":
                            attacked_text = watermark_attack.parrot_paraphrase_attack(watermarked_text, bigram=False)
                        elif attack_algorithm == "gpt35_turbo_paraphrase":
                            attacked_text = watermark_attack.gpt35_turbo_paraphrase_attack(watermarked_text)
                        
                        # Probing attack
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
                            raise ValueError(f"Unknown attack algorithm: {attack_algorithm}")
                        if attacked_text is not None and attacked_text != "":
                            break
                    except Exception as e:
                        print(f"Error during {attack_algorithm}: {e}. Retrying...")

                if attacked_text is None or attacked_text == "":
                    print(f"Attack failed after {MAX_ATTACK_TRY} tries.")
                    continue
            
            attack_result = {
                'text': attacked_text,
            }

            print(f"{attack_algorithm} Attack Result: {attack_result}")
            print('\n')

            df_attack_results.at[idx, column_name] = attack_result
            df_attack_results.to_json(ATTACK_RESULT_FILE_NAME, orient="index", indent=4)
