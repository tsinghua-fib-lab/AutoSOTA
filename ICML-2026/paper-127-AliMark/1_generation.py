import argparse
import json
import os

import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

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
    # pretty print the arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("\n")

    WATERMARK_ALGORITHM_NAME = args.watermark_algorithm
    WATERMARK_MODEL_NAME = args.watermark_model
    WATERMARK_EMBEDDER = args.watermark_embedder
    WATERMARK_BLOCK_SIZE = args.watermark_block_size
    WATERMARK_NUM_NEXT_SENTENCE_CANDIDATES = args.watermark_num_next_sentence_candidates
    MIN_NEW_SENTENCES = args.min_new_sentences
    DATASET_NAME = args.dataset_name
    DEVICE = args.device

    GENERATION_RESULT_DIR = f"_result/generation/block_size_{WATERMARK_BLOCK_SIZE}"
    os.makedirs(GENERATION_RESULT_DIR, exist_ok=True)
    GENERATION_RESULT_FILE_NAME = os.path.join(
        GENERATION_RESULT_DIR,
        f"{DATASET_NAME}_{WATERMARK_ALGORITHM_NAME}_{WATERMARK_MODEL_NAME.replace('/', '_')}.json"
    )
    GENERATION_RESULT_LIST = [
        "original_result",
        "unwatermarked_result",
        "watermarked_result",
    ]

    # Handle result file
    if not os.path.exists(GENERATION_RESULT_FILE_NAME):
        df_generation_results = pd.DataFrame(data={
            'question': [],
            'reference': [],
            'original_result': [],
            'unwatermarked_result': [],
            'watermarked_result': [],
        })
    else:
        df_generation_results = pd.read_json(GENERATION_RESULT_FILE_NAME, orient="index")
        for col in GENERATION_RESULT_LIST:
            if col not in df_generation_results.columns:
                df_generation_results[col] = None

    # Load data
    with open(f'dataset/{DATASET_NAME}.json', 'r') as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]

    watermark = AliMark(args)

    # Watermark Study
    for idx, line in enumerate(tqdm(lines)):
        question = line['prompt']
        reference = line['natural_text']

        print("*************")
        print(f"==Question: {question}")
        print('\n')

        # Generation Only
        print("---Original Result---")
        generate_done = False
        original_result = None
        if idx < df_generation_results.shape[0] and df_generation_results.loc[idx, 'original_result'] is not None:
            original_result = df_generation_results.loc[idx, 'original_result']
            generate_done = True
            print(f"{idx} | Original | already processed")
        
        if not generate_done:
            original_text = reference
            original_sents = sent_tokenize(original_text)
            
            while len(original_sents) < MIN_NEW_SENTENCES:
                if len(original_sents) <= 1 and not original_text.endswith('.'):
                    original_text += "."
                original_text += " " + original_text
                original_sents = sent_tokenize(original_text)
            
            if len(original_sents) >= MIN_NEW_SENTENCES:
                original_text = ' '.join(original_sents[:MIN_NEW_SENTENCES])
            
            original_result = {
                'text': original_text,
            }
            print(f"Original Text: {original_text}")
        print("\n")


        print("---Unwatermarked Result---")
        unwatermarked_result = None
        generate_done = False
        if idx < df_generation_results.shape[0] and df_generation_results.loc[idx, 'unwatermarked_result'] is not None:
            unwatermarked_result = df_generation_results.loc[idx, 'unwatermarked_result']
            print(f"{idx} | Unwatermarked | already processed")
        else:
            unwatermarked_text = watermark.generate_unwatermarked_text(question)
            unwatermarked_result = {
                'text': unwatermarked_text,
            }
            print(f"Unwatermarked Text: {unwatermarked_text}")
        print("\n")
        
        
        print("---Watermarked Result---")
        generate_done = False
        watermarked_result = None
        if idx < df_generation_results.shape[0] and df_generation_results.loc[idx, 'watermarked_result'] is not None:
            watermarked_result = df_generation_results.loc[idx, 'watermarked_result']
            print(f"{idx} | Watermarked | already processed")
            generate_done = True
        
        if not generate_done:
            watermarked_text = watermark.generate_watermarked_text(question)
            watermarked_result = {
                'text': watermarked_text,
            }
            print(f"Watermarked Text: {watermarked_text}")
        print("\n")

        # save the results to a json file
        df_generation_results.loc[idx] = {
            'question': question,
            'reference': reference,
            'original_result': original_result,
            'unwatermarked_result': unwatermarked_result,
            'watermarked_result': watermarked_result,
        }
        df_generation_results.to_json(GENERATION_RESULT_FILE_NAME, orient="index", indent=4)

        # break # for debug, only run one example