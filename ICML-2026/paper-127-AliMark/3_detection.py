import argparse
import itertools
import os
from collections import deque

import pandas as pd
from nltk.tokenize import sent_tokenize
from rich.console import Group
from rich.live import Live
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.text import Text
from tqdm import tqdm

from watermark.alimark import AliMark


def get_df_detection_results(REGULAR_RESULT_FILE_NAME, ATTACK_RESULT_FILE_NAME, DETECTION_RESULT_FILE_NAME):
    if os.path.exists(REGULAR_RESULT_FILE_NAME):
        df_regular_results = pd.read_json(REGULAR_RESULT_FILE_NAME, orient="index")
    else:
        raise FileNotFoundError(f"{REGULAR_RESULT_FILE_NAME} not found. Please run full_study_watermarking.py first.")

    if os.path.exists(ATTACK_RESULT_FILE_NAME):
        df_attack_results = pd.read_json(ATTACK_RESULT_FILE_NAME, orient="index")
    else:
        df_attack_results = None
    
    if os.path.exists(DETECTION_RESULT_FILE_NAME):
        df_detection_results = pd.read_json(DETECTION_RESULT_FILE_NAME, orient="index")
        
        if df_attack_results is not None:
            unique_attack_cols = df_attack_results.columns.difference(df_regular_results.columns)

            for col in unique_attack_cols:
                if col not in df_detection_results.columns:
                    df_detection_results[col] = None

            attack_dict = df_attack_results[unique_attack_cols].to_dict(orient='index')
            detection_dict = df_detection_results.to_dict(orient='index')
            
            for idx, attack_row in tqdm(attack_dict.items(), total=len(attack_dict), desc="Processing rows"):
                for col, attack_result in attack_row.items():
                    if pd.isna(attack_result):
                        continue
                        
                    detection_result = detection_dict[idx].get(col)
                    
                    if pd.isna(detection_result):
                        detection_dict[idx][col] = attack_result
                    else:
                        try:
                            
                            if detection_result.get("question") != attack_result.get("question"):
                                raise ValueError(f"Row {idx}, Column '{col}': Question mismatch between detection and attack results.")
                            
                            if detection_result.get("text") != attack_result.get("text"):
                                detection_dict[idx][col] = attack_result
                        except (TypeError, AttributeError):
                            pass
            df_detection_results = pd.DataFrame.from_dict(detection_dict, orient='index')
    else:
        if df_attack_results is not None:
            unique_attack_cols = df_attack_results.columns.difference(df_regular_results.columns)
            df_detection_results = pd.merge(
                df_regular_results,
                df_attack_results[unique_attack_cols],
                left_index=True,
                right_index=True,
                how="outer"
            )
        else:
            df_detection_results = df_regular_results.copy()
    return df_detection_results

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

    GENERATION_RESULT_DIR = f"_result/generation/block_size_{WATERMARK_BLOCK_SIZE}"
    ATTACK_RESULT_DIR = f"_result/attack/block_size_{WATERMARK_BLOCK_SIZE}"
    DETECTION_RESULT_DIR = f"_result/detection/block_size_{WATERMARK_BLOCK_SIZE}"
    os.makedirs(DETECTION_RESULT_DIR, exist_ok=True)

    GENERATION_RESULT_FILE_NAME = os.path.join(
        GENERATION_RESULT_DIR,
        f"{DATASET_NAME}_{WATERMARK_ALGORITHM_NAME}_{WATERMARK_MODEL_NAME.replace('/', '_')}.json"
    )
    ATTACK_RESULT_FILE_NAME = os.path.join(
        ATTACK_RESULT_DIR,
        f"{DATASET_NAME}_{WATERMARK_ALGORITHM_NAME}_{WATERMARK_MODEL_NAME.replace('/', '_')}.json"
    )
    DETECTION_RESULT_FILE_NAME = os.path.join(
        DETECTION_RESULT_DIR,
        f"{DATASET_NAME}_{WATERMARK_ALGORITHM_NAME}_{WATERMARK_MODEL_NAME.replace('/', '_')}.json"
    )
    
    df_detection_results = get_df_detection_results(
        GENERATION_RESULT_FILE_NAME, 
        ATTACK_RESULT_FILE_NAME, 
        DETECTION_RESULT_FILE_NAME
        )

    # Detection
    watermark = AliMark(args, load_llm=False)
    IGNORE_COLS = ["question", "reference", "unwatermarked_result"]

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),   
        BarColumn(bar_width=None),                     
        MofNCompleteColumn(),                          
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),  
        TimeElapsedColumn(),                           
        TimeRemainingColumn(),                        
    )
    progress.start()
    live = Live(auto_refresh=False)
    live.start()
    buffer = deque([""] * 20, maxlen=20)
    task = progress.add_task("[bold green] Watermark Detection ...", total=len(df_detection_results))
    for idx, row in df_detection_results.iterrows():
        progress.update(task, advance=1)
        for col in df_detection_results.columns:
            if col in IGNORE_COLS:
                continue

            if pd.isna(row[col]):
                continue
            
            if not pd.isna(row[col]) and row[col].get("detect_result") is not None:
                continue

            text = row[col].get("text", "")
            if text == "":
                continue

            # AliMark Detection
            detect_result_entry_name = f"detect_result"
            if detect_result_entry_name in row[col]:
                buffer.append(f"Row {idx}, Column '{col}' already has detection result. Skipping detection.")
                live.update(Group(*[Text(line) for line in buffer]), refresh=True)
                continue

            # You can alternatively configure the detection parameters here, e.g., lower_ratio, upper_ratio, etc.
            detection_result = watermark.detect_watermark(text=text) 
            df_detection_results.loc[idx, col][detect_result_entry_name] = detection_result
            df_detection_results.to_json(DETECTION_RESULT_FILE_NAME, orient="index", indent=4)
            
            buffer.append(f"Row {idx}, Column '{col}' - Completed detection.")
            live.update(Group(*[Text(line) for line in buffer]), refresh=True)
            
        # break
    
    progress.stop()
    live.stop()
