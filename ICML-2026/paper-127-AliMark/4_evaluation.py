import argparse
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm


def calc_performance(scores_original, scores_watermarked):
    y_true = [0] * len(scores_original) + [1] * len(scores_watermarked)

    y_true = pd.Series(y_true).fillna(0)
    y_scores = pd.Series(scores_original + scores_watermarked).fillna(0)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    def safe_interp(x, xp, fp):
        if x < xp[0]:
            return fp[0]
        elif x > xp[-1]:
            return fp[-1]
        else:
            return np.interp(x, xp, fp)

    tpr_at_0001 = safe_interp(0.0001, fpr, tpr)
    tpr_at_0010 = safe_interp(0.0010, fpr, tpr)
    tpr_at_0050 = safe_interp(0.0050, fpr, tpr)
    tpr_at_0100 = safe_interp(0.0100, fpr, tpr)

    results = {
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "tpr@0.001": tpr_at_0001,
        "tpr@0.010": tpr_at_0010,
        "tpr@0.050": tpr_at_0050,
        "tpr@0.100": tpr_at_0100
    }

    return results

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
    args = parser.parse_args()

    WATERMARK_ALGORITHM_NAME = args.watermark_algorithm
    WATERMARK_MODEL_NAME = args.watermark_model
    WATERMARK_EMBEDDER = args.watermark_embedder
    WATERMARK_BLOCK_SIZE = args.watermark_block_size
    WATERMARK_NUM_NEXT_SENTENCE_CANDIDATES = args.watermark_num_next_sentence_candidates
    MIN_NEW_SENTENCES = args.min_new_sentences
    DATASET_NAME = args.dataset_name
    
    ATTACK_ALGORITHM_LIST = [
        "watermarked",

        "pegasus_paraphrase_no_bigram",
        "parrot_paraphrase_no_bigram",
        "dipper_all_paraphrase", 
        "gpt35_turbo_paraphrase",

        # "insert_10", "insert_20", "insert_30", "insert_40", "insert_50",
        # "delete_10", "delete_20", "delete_30", "delete_40", "delete_50",
        # "reorder_10", "reorder_20", "reorder_30", "reorder_40", "reorder_50",
        ]
    
    DETECTION_RESULT_DIR = f"_result/detection/block_size_{WATERMARK_BLOCK_SIZE}"
    DETECTION_RESULT_FILE_NAME = os.path.join(
        DETECTION_RESULT_DIR,
        f"{DATASET_NAME}_{WATERMARK_ALGORITHM_NAME}_{WATERMARK_MODEL_NAME.replace('/', '_')}.json"
    )

    IGNORE_COLS = ["question", "reference", "unwatermarked_result"]
    
    print(f"|| Dataset: {DATASET_NAME} || {WATERMARK_ALGORITHM_NAME} + {WATERMARK_MODEL_NAME} ||")
    print(f"{'':<50}| ROCAUC / TPR001 / TPR005 | NEG:POS |")
    
    if os.path.exists(DETECTION_RESULT_FILE_NAME):
        scores_list_map = {}
        df_detection_result = pd.read_json(DETECTION_RESULT_FILE_NAME, orient="index")

        for idx, row in df_detection_result.iterrows():
            for col in df_detection_result.columns:
                if col in IGNORE_COLS:
                    continue
                if not row[col]:
                    continue
                result = row[col]

                detect_result_entry_name = f"detect_result"
                if isinstance(result, dict) and detect_result_entry_name in result:
                    score = result[detect_result_entry_name]["score"]
                    score_name = f"{col}_{detect_result_entry_name}"
                    scores_list_map.setdefault(score_name, []).append(score)
                        
        for i, attack_algorithm in enumerate(ATTACK_ALGORITHM_LIST):
            detect_result_entry_name = f"detect_result"
            scores_original = scores_list_map[f"original_result_{detect_result_entry_name}"]
            scores_watermarked = scores_list_map[f"{attack_algorithm}_result_{detect_result_entry_name}"]

            performance = calc_performance(scores_original, scores_watermarked)
            print(f"{f'Attack Algorithm: {attack_algorithm}':<50}| {performance['roc_auc']:.4f} / {performance['tpr@0.010']:.4f} / {performance['tpr@0.050']:.4f} | {len(scores_original):<3}:{len(scores_watermarked):<3} |")
