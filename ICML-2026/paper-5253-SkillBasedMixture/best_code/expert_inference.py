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
from utils import *
from agent import *

import pandas as pd
from ast import literal_eval
from collections import defaultdict

import contextlib
import gc
import os
import subprocess
import time

from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
os.environ["OMP_NUM_THREADS"] = "20"

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def get_model_responses(agent_name, task, questions, num_gpu):
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
    
    if task in ["MATH", "AIME24"]:
        prompts = [
            f"Question: {q}\n"
            f"Provide your step-by-step reasoning first, and then print \"The answer is \\boxed{{X}}\", "
            f"where X is the final answer, at the end of your response."
            for q in questions
        ]
    else:
        prompts = [
            f"Question: {q}\n"
            f"Provide your step-by-step reasoning first, and then print \"The answer is (X)\", "
            f"where X is the answer choice (one capital letter), at the end of your response."
            for q in questions
        ]

    results = vllm_generate(agent_name, llm, tokenizer, prompts, temperature=0.7)
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    if hasattr(llm, 'engine'):
        del llm.engine
    if 'llm' in locals():
        del llm
    gc.collect()
    torch.cuda.empty_cache()
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    time.sleep(10)
    subprocess.run(["nvidia-smi"], check=True)
    return results

def process_questions_and_answers(df, task, gpus):
    # Convert string representations of lists to actual lists
    df['solvers'] = df['solvers'].apply(literal_eval)
    
    # Create a mapping for each solver to their questions and tracking answer counts
    solver_questions = defaultdict(list)
    question_answer_counts = defaultdict(lambda: defaultdict(int))
    
    # Map questions to solvers and count occurrences
    for idx, row in df.iterrows():
        question = row['question']
        if task in ["MATH", "AIME24"]:
            prompt = (
                f"Question: {question}\n"
                f"Provide your step-by-step reasoning first, and then print \"The answer is \\boxed{{X}}\", "
                f"where X is the final answer, at the end of your response."
            )
        else:
            prompt = (
                f"Question: {question}\n"
                f"Provide your step-by-step reasoning first, and then print \"The answer is (X)\", "
                f"where X is the answer choice (one capital letter), at the end of your response."
            )
        for solver in row['solvers']:
            solver_questions[solver].append((idx, prompt))
            question_answer_counts[idx][solver] += 1
    
    # Process each solver's questions and update DataFrame
    for solver, questions in solver_questions.items():
        # Get question IDs and texts separately
        q_ids, q_texts = zip(*questions)

        # Get model responses
        responses = get_model_responses(solver, task, q_texts, gpus)
        
        # Add responses to DataFrame
        for q_id, response in zip(q_ids, responses):
            answer_num = question_answer_counts[q_id][solver]
            question_answer_counts[q_id][solver] -= 1
            col_name = f"{solver}_answer_{answer_num}"
            df.loc[q_id, col_name] = response
    
    return df

def get_valid_responses(row, answer_columns):
    responses = {}
    for col in answer_columns:
        if pd.notna(row[col]) and pd.notnull(row[col]):
            responses[col] = row[col]
    return responses
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
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

if __name__ == "__main__":
    
    args = parse_args()
    seed_everything(args.seed)

    df = pd.read_csv(f"./skills/{args.task}/test_samples_with_keywords_and_solvers_seed{args.seed}.csv")
    print(f"Running inference on {args.gpus} GPUS.")
    
    final_df = process_questions_and_answers(df, args.task, args.gpus)
    final_df.to_csv(f"./skills/{args.task}/round0_seed{args.seed}.csv", index=False)
    num_choice = 10 if args.task == "MMLU_Pro" else 4
    answer_columns = [col for col in final_df.columns if 'answer_' in col]

    num_correct = 0
    for i, row in final_df.iterrows():
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
        if is_math_equiv(maj, str(gt)):
            num_correct += 1
    acc = round(num_correct / final_df.shape[0] * 100, 2)
    print("Initial accuracy with the 3 experts: ", acc)