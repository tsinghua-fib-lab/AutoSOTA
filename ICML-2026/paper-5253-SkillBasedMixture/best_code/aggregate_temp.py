import os
import time
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
from typing import List, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import read_json, write_json, get_keywords, get_alphabet_choice, remove_boxed, last_boxed_only_string
from agent import *
import logging
import pandas as pd
from ast import literal_eval
from collections import defaultdict

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--aggregator',
        type=str,
        default="QwenR1"
    )
    parser.add_argument(
        '--task',
        type=str,
        default="MMLU"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1
    )
    return parser.parse_args()

def get_model_responses(agent_name, prompts, num_gpu):
    model_id = agent_map.get(agent_name)   
    print(f"getting responses from this model: {agent_name}")
    if agent_name in ["Phi", "Mistral"]:
        llm = LLM(model = model_id, enforce_eager=True,
                  download_dir = "/models",
                  max_model_len = 16000,
                  tensor_parallel_size = num_gpu,
                  trust_remote_code = True)
    elif agent_name in ["DeepSeekMath"]:
        llm = LLM(model = model_id, enforce_eager=True,
                  download_dir = "/models",
                  max_model_len = 4096,
                  tensor_parallel_size = num_gpu,
                  trust_remote_code = True)        
    else:
        llm = LLM(model = model_id, enforce_eager=True,
                  download_dir = "/models",
                  tensor_parallel_size = num_gpu,
                  trust_remote_code = True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    results = vllm_generate(agent_name, llm, tokenizer, prompts, temperature=0.3)
    return results
    
def get_valid_responses(row, answer_columns):
    responses = {}
    for col in answer_columns:
        if pd.notna(row[col]) and pd.notnull(row[col]):
            responses[col] = row[col]
    return responses
        
if __name__ == "__main__":

    args = parse_args()
    seed_everything(args.seed)
    start_time = time.time()
    num_choice = 10 if args.task == "MMLU_Pro" else 4
    test_samples = read_json(f"./test_data/{args.task}_test.json")
    round_zero_df = pd.read_csv(f"./skills/{args.task}/round0_seed{args.seed}.csv")
    answer_columns = [col for col in round_zero_df.columns if 'answer_' in col]
    
    num_correct = 0
    for i, row in round_zero_df.iterrows():
        gt = row['gold_answer']
        valid_responses = get_valid_responses(row, answer_columns)
        preds = []
        for r in list(valid_responses.values()):
            if args.task in ["MATH", "AIME24"]:
                pred = remove_boxed(last_boxed_only_string(r))
            else:
                pred = get_alphabet_choice(r, num_choice=num_choice)
            preds.append(pred)
        maj = Counter(preds).most_common(1)[0][0]
        if is_math_equiv(maj, gt):
            num_correct += 1
    acc = round(num_correct / round_zero_df.shape[0] * 100, 2)
    print(f"Initial accuracy with the 3 experts: ", acc)
    
    agg_prompts = []
    for i, row in round_zero_df.iterrows():
        q = row['question']
        agg_prompt = (f"You have been provided with a set of responses from various open-source models to the latest user query. "
                      f"Your task is to synthesize these responses into a single, high-quality response. "
                      f"It is crucial to critically evaluate the information provided in these responses, "
                      f"recognizing that some of it may be biased or incorrect. "
                      f"Your response should not simply replicate the given answers but should offer a refined, "
                      f"accurate, and comprehensive reply to the instruction. "
                      f"Ensure your response is well-structured, coherent, and adheres" 
                      f"to the highest standards of accuracy and reliability. "
                      f"Responses from models:\n\n")
        
        valid_responses = get_valid_responses(row, answer_columns)
        valid_responses = list(valid_responses.values())
        for idx, res in enumerate(valid_responses):
            res  = res.split("</think>")[-1]
            agg_prompt += f"### Model {idx+1}'s response:\n{res}\n\n"

        if args.task in ["MATH", "AIME24"]:
            agg_prompt += (f"Question: {q}\n"
                           f"Provide your step-by-step reasoning first, and then print \"The answer is \\boxed{{X}}\", "
                           f"where X is the final answer, at the end of your response."
                        )
        else:
            agg_prompt += (f"Question: {q}\n"
                           f"Provide your step-by-step reasoning first, and then print \"The answer is (X)\", "
                           f"where X is the answer choice (one capital letter), at the end of your response."
                    )
        agg_prompts.append(agg_prompt)  
    
    round_zero_df = round_zero_df.loc[:, ['question', 'gold_answer', 'keywords', 'solvers']] # get rid of all prev answers
    result = get_model_responses(args.aggregator, agg_prompts, args.gpus)
    
    num_correct = 0
    for r, ts in zip(result, test_samples):
        gt = ts['gold_answer']
        if args.task in ["MATH", "AIME24"]:
            pred = remove_boxed(last_boxed_only_string(r))
        else:
            pred = get_alphabet_choice(r, num_choice=num_choice)
        if is_math_equiv(pred, gt):
            num_correct += 1
    acc = round(num_correct / len(test_samples) * 100, 2)
    print(f"acc: {acc} | dataset: {args.task} | aggregator: {args.aggregator} | seed: {args.seed}")
    write_json(result, f"./skills/{args.task}/fixed_{args.aggregator}_round1_seed{args.seed}_{acc}.json")