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
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--agent',
        type=str
    )
    parser.add_argument(
        '--task',
        type=str
    )
    parser.add_argument(
        '--gpus',
        type=int
    )
    return parser.parse_args()

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
if __name__ == "__main__":

    args = parse_args()
    seed_everything(0)
    model_id = agent_map.get(args.agent)   
    print(f"creating memory bank for this model: {model_id}")
    
    llm = LLM(model = model_id, enforce_eager=True,
              download_dir="/models",
              tensor_parallel_size=args.gpus,
              trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    evaluator = SkillEvaluator(llm, tokenizer, task=args.task, agent_name=args.agent)
    
    train_samples = read_json(f"./skills/{args.task}/train_samples_with_keywords_Qwen.json")
    train_samples_updated, agent_profile, train_acc = evaluator.create_profile(train_samples, args.agent)    
    write_json(agent_profile, f"./skills/{args.task}/{args.agent}_profile_{train_acc}.json")
    write_json(train_samples_updated, f"./skills/{args.task}/{args.agent}_CoT.json")