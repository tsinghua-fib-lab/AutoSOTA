import os
import time
import glob
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
    num_choice = 10 if args.task == "MMLU_Pro" else 4
    print(f"benchmarking aggregator's performance using this model: {model_id}")
    if args.agent in ["Phi", "Mistral"]:
        llm = LLM(model = model_id, enforce_eager=True,
                  download_dir = "/models",
                  max_model_len = 16000,
                  tensor_parallel_size = args.gpus,
                  trust_remote_code = True)
    elif args.agent in ["DeepSeekMath"]:
        llm = LLM(model = model_id, enforce_eager=True,
                  download_dir = "/models",
                  max_model_len = 4096,
                  tensor_parallel_size = args.gpus,
                  trust_remote_code = True)        
    else:
        llm = LLM(model = model_id, enforce_eager=True,
                  download_dir = "/models",
                  tensor_parallel_size = args.gpus,
                  trust_remote_code = True)
        
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    aggr_bench_file = f"./skills/{args.task}/aggr_bench.json"
    if not os.path.exists(aggr_bench_file):
        print("generating the sythetic task to benchmark aggregator's performance...")
        all_agent_outputs = {}
        outputs = glob.glob(f"./skills/{args.task}/*_CoT.json")
        model_names = [path.split('_CoT')[0].split(f'./skills/{args.task}/')[-1] for path in outputs]

        for pf, mn in zip(outputs, model_names):
            all_agent_outputs[mn] = read_json(pf)
        agents = list(all_agent_outputs.keys())

        aggr_samples = []

        for i in range(len(all_agent_outputs[agents[0]])):
            tmp = {}
            question = all_agent_outputs[agents[0]][i]['question']
            correct_CoT, incorrect_CoT = [], []
            
            # Collect correct and incorrect CoTs
            for agent in agents:
                CoT = all_agent_outputs[agent][i][f'reasoning_{agent}']
                if all_agent_outputs[agent][i][f'pred_{agent}'] == all_agent_outputs[agent][i]['gold_answer']:
                    correct_CoT.append(CoT)
                else:
                    incorrect_CoT.append(CoT)
            
            # Sample one correct and two incorrect CoTs
            sampled_correct = random.choice(correct_CoT) if correct_CoT else ""
            sampled_incorrect = random.sample(incorrect_CoT, k=min(2, len(incorrect_CoT))) if len(incorrect_CoT) >= 2 else [""] * 2
            
            # Combine all CoTs and shuffle them
            all_cots = [sampled_correct] + sampled_incorrect
            random.shuffle(all_cots)
            
            # Create the final string
            final_string = agg_prompt = (f"Question: {question}\n"
                                        f"Provide your step-by-step reasoning first, and then print \"The answer is (X)\", "
                                        f"where X is the answer choice (one capital letter), at the end of your response.\n\n"
                                        )
            agg_prompt += (f"You have been provided with a set of responses from various open-source models to the latest user query. "
                        f"Your task is to synthesize these responses into a single, high-quality response. "
                        f"It is crucial to critically evaluate the information provided in these responses, "
                        f"recognizing that some of it may be biased or incorrect. "
                        f"Your response should not simply replicate the given answers but should offer a refined, "
                        f"accurate, and comprehensive reply to the instruction. "
                        f"Ensure your response is well-structured, coherent, and adheres" 
                        f"to the highest standards of accuracy and reliability. "
                        f"Responses from models:\n")
            for idx, cot in enumerate(all_cots, 1):
                agg_prompt += f"\n### Model{idx}:\n{cot}\n"
            tmp['question'] = agg_prompt
            tmp['gold_answer'] = all_agent_outputs[agent][i]['gold_answer']
            aggr_samples.append(tmp)
        write_json(aggr_samples, aggr_bench_file)

    train_samples = read_json(f"./skills/{args.task}/aggr_bench.json")
    prompts = [i['question'] for i in train_samples]
    results = vllm_generate(args.agent, llm, tokenizer, prompts, temperature=0.7)

    num_correct = 0
    for sample, result in zip(train_samples, results):
        if args.task in ["MATH", "AIME24"]:
            pred = remove_boxed(last_boxed_only_string(result))
        else:
            pred = get_alphabet_choice(result, num_choice=num_choice)
        if pred == sample['gold_answer']:
            num_correct += 1
        sample['aggr_output'] = result

    acc = round(num_correct / len(train_samples) * 100, 2)
    write_json(train_samples, f"./skills/{args.task}/{args.agent}_as_aggr_{acc}.json")