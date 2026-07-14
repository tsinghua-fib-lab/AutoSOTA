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
from utils import read_json, write_json, get_keywords, get_alphabet_choice
from agent import *

def seed_everything(seed):
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
        '--task',
        type=str,
        default="MMLU"
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1
    )
    return parser.parse_args()
    
if __name__ == "__main__":

    args = parse_args()
    agent = "Qwen" # Fixed for keyword generation 
    model_id = agent_map.get(agent)
    seed_everything(0)
    print(f"Annotating keywords on {args.task} using this model: {model_id}")
    llm = LLM(model = model_id, download_dir="/models", tensor_parallel_size=args.gpus, enforce_eager=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    evaluator = SkillEvaluator(llm, tokenizer, task=args.task, agent_name=agent)
    print(f"Annotating keywords for training samples...(k={args.k})")
    evaluator.train_samples = evaluator.annotate_skills(evaluator.train_samples, agent)
    print(f"Annotating keywords for testing samples...(k={args.k})")
    test_samples = evaluator.annotate_skills(evaluator.test_samples, agent)
        
    write_json(evaluator.train_samples, f"./skills/{args.task}/train_samples_with_keywords_Qwen.json")
    write_json(test_samples, f"./skills/{args.task}/test_samples_with_keywords_Qwen.json")