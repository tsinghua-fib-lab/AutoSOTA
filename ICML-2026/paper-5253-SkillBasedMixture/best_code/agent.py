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
from utils import read_json, write_json, get_keywords, get_alphabet_choice, is_math_equiv, remove_boxed, last_boxed_only_string

# Base path for locally downloaded models
_MODEL_BASE = "/models"

agent_map = {
    "Llama": f"{_MODEL_BASE}/meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen": f"{_MODEL_BASE}/Qwen/Qwen2.5-7B-Instruct",
    "Mistral": f"{_MODEL_BASE}/mistralai/Mistral-Nemo-Instruct-2407",
    "Phi": f"{_MODEL_BASE}/microsoft/Phi-3.5-mini-instruct",
    "Gemma": f"{_MODEL_BASE}/google/gemma-2-9b-it",
    "GLM": f"{_MODEL_BASE}/THUDM/glm-4-9b-chat",
    "Exaone": f"{_MODEL_BASE}/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "Granite": f"{_MODEL_BASE}/ibm-granite/granite-3.1-8b-instruct",
    "QwenMath": f"{_MODEL_BASE}/Qwen/Qwen2.5-Math-7B",
    "QwenCode": f"{_MODEL_BASE}/Qwen/Qwen2.5-Coder-7B-Instruct",
    "DeepSeekMath": f"{_MODEL_BASE}/deepseek-ai/deepseek-math-7b-instruct",
    "QwenR1": f"{_MODEL_BASE}/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "LlamaR1": f"{_MODEL_BASE}/deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "InternLM": f"{_MODEL_BASE}/internlm/internlm3-8b-instruct",
    "Mathstral": f"{_MODEL_BASE}/mistralai/Mathstral-7B-v0.1",
    "BioLlama": f"{_MODEL_BASE}/ContactDoctor/Bio-Medical-Llama-3-8B",
    "Qwen72B": f"{_MODEL_BASE}/Qwen/Qwen2.5-72B-Instruct",
    "Llama70B": f"{_MODEL_BASE}/meta-llama/Llama-3.3-70B-Instruct"
}

class SkillEvaluator:
 
    def __init__(self, llm, tokenizer, task: str, agent_name: str):
        self.llm = llm
        self.tokenizer = tokenizer
        self.task = task
        self.agent_name = agent_name
        self.train_samples = self._load_train_data()
        self.test_samples = self._load_test_data()
        self.num_choice = 10 if self.task == "MMLU_Pro" else 4 
        assert self.task in ["MMLU", "MMLU_Pro", "MATH", "GPQA", "MedMCQA", "AIME24"]
        
    def _load_train_data(self) -> List[Dict]:
        return read_json(f"./test_data/{self.task}_train.json")
            
    def _load_test_data(self) -> List[Dict]:
        return read_json(f"./test_data/{self.task}_test.json")

    def annotate_skills(self, samples: List[Dict], agent: str, k=5) -> List[Dict]:

        keyword_prompts = [
            f"Question: {sample['question']}\n"
            f"What are the core knowledge, subjects or skills needed to solve this problem? "
            f"List 2-5 keywords separated in comma. "
            f"Example keywords: psychology, virology, behavioral theory, microbiology, "
            f"diplomacy, political science, property law, finance, business. "
            f"Give ONLY the keywords, no other words or explanation. "
            f"Follow this format: Keywords: <keyword1>, <keyword2>..."
            for sample in samples
        ]

        for _ in range(k):
            results = self.generate(keyword_prompts)
            
            for i, sample in enumerate(samples):
                if "keywords" not in sample:
                    sample["keywords"] = get_keywords(results[i])
                else:
                    sample["keywords"].extend(get_keywords(results[i]))

        for sample in samples:
            sample["keywords"] = [k for k in sample["keywords"] if len(k) <= 20]
            sample["keywords"] = [k for k, count in Counter(sample["keywords"]).items() if count > 1]
        
        return samples

    def create_profile(self, samples: List[Dict], agent: str) -> Dict[str, Dict]:

        if self.task in ["MATH", "AIME24"]:
            train_prompts = [
                f"Question: {sample['question']}\n"
                f"Provide your step-by-step reasoning first, and then print \"The answer is \\boxed{{X}}\", "
                f"where X is the final answer, at the end of your response."
                for sample in samples
            ]
        else:
            train_prompts = [
                f"Question: {sample['question']}\n"
                f"Provide your step-by-step reasoning first, and then print \"The answer is (X)\", "
                f"where X is the answer choice (one capital letter), at the end of your response."
                for sample in samples
            ]
        
        results = self.generate(train_prompts)

        num_correct = 0
        for i, sample in enumerate(samples):
            sample[f'reasoning_{agent}'] = results[i]
            if self.task in ["MATH", "AIME24"]:
                pred = remove_boxed(last_boxed_only_string(results[i]))
            else:
                pred = get_alphabet_choice(results[i], self.num_choice)
            sample[f'pred_{agent}'] = pred
            if is_math_equiv(pred, sample['gold_answer']):
                sample[f'is_correct_{agent}'] = 1  
                num_correct += 1
            else:
                sample[f'is_correct_{agent}'] = 0

        train_acc = round(num_correct / len(samples) * 100, 2)
        print(f"training acc: {train_acc}")
                
        agent_profile = {}    
        for sample in samples:
            keywords = sample[f'keywords']
            for keyword in keywords:
                if sample[f'is_correct_{agent}']:
                    agent_profile[keyword] = agent_profile.get(keyword, 0) + 1
                else:
                    agent_profile[keyword] = agent_profile.get(keyword, 0) - 1
        
        return samples, agent_profile, train_acc

    def generate(self, prompts, temperature=0.7):
        messages = []    

        for p in prompts:
            msg = [{"role": "user", "content": p}]
            msg = self.tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True
            )
            messages.append(msg)
            
        max_tokens = 32768 if "R1" in self.agent_name else 4096
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.llm.generate(messages, sampling_params)
        return [output.outputs[0].text for output in outputs]

def vllm_generate(agent_name, llm, tokenizer, prompts, temperature):
    messages = []    

    for p in prompts:
        msg = [{"role": "user", "content": p}]  
        msg = tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True
        )
        messages.append(msg)
    max_tokens = 32768 if "R1" in agent_name else 4096
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(messages, sampling_params)
    return [output.outputs[0].text for output in outputs]