# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
import torch
from transformers import set_seed
from tqdm import tqdm
from vllm import LLM, SamplingParams
from collections import Counter
from dataclasses import dataclass
import random
import argparse
import json
from datasets import Dataset

from utils.ar_func import get_python_answer, process_python_batch
from utils.extract_update import update_sample, update_lean_sample, update_python_sample, update_summary_sample, \
    process_batch_math, process_batch_prove, \
    check_answer, check_answer_maj, chunked_iterable
from prover.lean.verifier import Lean4ServerScheduler

def build_vllm(config):
    num_gpus = torch.cuda.device_count()
    vllm = LLM(
        model=config.model_id,
        tensor_parallel_size=num_gpus,
        swap_space=0,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
    )
    return vllm

def generate_batched(samples, vllm, sampling_params):
    generated = vllm.generate(samples["gen_texts"], sampling_params, use_tqdm=True)
    samples["gen_texts"] = [o.prompt + o.outputs[0].text for o in generated]
    return samples

def generate_individually(samples, vllm, sampling_params, batch_size=128, max_batch_size=128):
    new_samples = []
    buffer = []
    for sample in samples:
        batched_samples = [{"text": sample["text"], "gen_texts": sample["gen_texts"], 
                            "should_prune": False, "model_answers": "-1", "complete": False, "has_code": True}
                            for _ in range(batch_size)]
        
        buffer.extend(batched_samples)
        
        while len(buffer) >= max_batch_size:
            small_batch = buffer[:max_batch_size]
            del buffer[:max_batch_size]
            
            outputs = vllm.generate([s["gen_texts"] for s in small_batch], sampling_params, use_tqdm=False)
            for i, o in enumerate(outputs):
                small_batch[i]["gen_texts"] = o.prompt + o.outputs[0].text 
            
            new_samples.extend(small_batch)
            
    if buffer:
        outputs = vllm.generate([s["gen_texts"] for s in buffer], sampling_params, use_tqdm=False)
        for i, o in enumerate(outputs):
            buffer[i]["gen_texts"] = o.prompt + o.outputs[0].text
        new_samples.extend(buffer)
    
    return new_samples

def main_prove(config, test_list):
    set_seed(42)
    verifier = Lean4ServerScheduler(max_concurrent_requests=128, timeout=60, memory_limit=100)
    
    vllm = build_vllm(config)
    sampling_params_lean4 = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_new_tokens,
        stop=["```lean4"],
        include_stop_str_in_output=True,
    )
    
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_new_tokens,
        stop=["```"],
        include_stop_str_in_output=True,
    )
    
    complete_count = 0
    total = len(test_list)
    fw = open(config.output_path, "w")
    
    for test in tqdm(test_list, desc="Solving problems"):
        problem = {"text": "### Problem: " + test["informal_prefix"] + '\n### Solution:'}
        problem["text"] += "\n### Informal proof: " + \
                    f"We need to prove the following theorem: ```lean4\n"  + test["formal_statement"] + "```.\n"

        samples = Dataset.from_list([
            {
                "text": problem["text"],
                "gen_texts": problem["text"],
                "should_prune": False,
                "model_answers": "-1",
                "complete": False, 
                "has_code": True,
            }
            for _ in range(config.num_samples)
        ])
    
        samples = samples.map(
            generate_batched,
            batch_size=config.num_batch,
            batched=True,
            fn_kwargs={"vllm": vllm, "sampling_params": sampling_params_lean4},
            load_from_cache_file=False,
        )
        
        samples = samples.map(lambda x: update_sample(x, test))
        
        samples = samples.map(
            generate_batched,
            batch_size=config.num_batch,
            batched=True,
            fn_kwargs={"vllm": vllm, "sampling_params": sampling_params},
            load_from_cache_file=False,
        )
        
        samples = process_batch_prove(samples, verifier, test["formal_statement"], test["header"])
        
        completed_samples = samples.filter(lambda x: x["complete"] is True, load_from_cache_file=False).to_list()
            
        if len(completed_samples) > 0:
            complete_count += 1
            if len(completed_samples) > 5:
                selected_samples = random.sample(completed_samples, 5)
            else:
                selected_samples = completed_samples
        else:
            selected_samples = [random.choice(samples)]

        lean4_code_list = []
        for sample in selected_samples:
            gen_texts = sample.get("gen_texts", [])
            matches = re.findall(r'```lean4(.*?)```', gen_texts, re.DOTALL)
            if matches:
                lean4_code_list.append(matches[-1])

        first_sample = selected_samples[0]
        first_sample["lean4_code_list"] = lean4_code_list

        fw.write(json.dumps(first_sample, ensure_ascii=False) + "\n")
        fw.flush()
    
    verifier.close()
    print("Accuracy", round((complete_count / total) * 100, 2), "%")

def main_math(config, test_list, math_maj):
    set_seed(42)
    verifier = Lean4ServerScheduler(max_concurrent_requests=128, timeout=60, memory_limit=100)
    
    vllm = build_vllm(config)
    sampling_params_formal = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_new_tokens,
        top_p=0.95,
        stop=["### Formal"],
        include_stop_str_in_output=False,
    )
    
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_new_tokens,
        top_p=0.95,
        stop=["```"],
        include_stop_str_in_output=True,
    )
    
    sampling_params_numbersign = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_new_tokens,
        top_p=0.95,
        stop=["###"],
        include_stop_str_in_output=False,
    )
    
    def batch_generate(samples, stage):
        if stage == "formal":
            sampling_params_stage = sampling_params_formal
            update_fn = None
        elif stage == "lean":
            sampling_params_stage = sampling_params
            update_fn = update_lean_sample
        elif stage == "python":
            sampling_params_stage = sampling_params
            update_fn = update_python_sample
        elif stage == "summary":
            sampling_params_stage = sampling_params_numbersign
            update_fn = update_summary_sample
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        if update_fn:
            samples = samples.map(lambda x: update_fn(x), batched=False)
        
        samples = samples.map(
            lambda batch: generate_batched(
                batch, 
                vllm=vllm, 
                sampling_params=sampling_params_stage
            ),
            batched=True,
            batch_size=config.num_batch,
            load_from_cache_file=False,
        )
        
        return samples
    
    stages = ["formal", "lean", "python", "summary"]
    
    complete_count = 0
    total = len(test_list)
    fw = open(config.output_path, "w")
    
    chunk_problem_size = max(1, config.num_batch // config.num_samples)
    global_item_id = 0
    
    for test_batch in tqdm(chunked_iterable(test_list, chunk_problem_size), desc="Processing batches"):
        all_samples = []
        for test in test_batch:
            problem = {"text": "### Problem: " + test["problem"] + '\n### Solution:'}
        
            for _ in range(config.num_samples):
                sample = {
                    "text": problem["text"],
                    "gen_texts": problem["text"],
                    "complete": False, 
                    "has_code": True,
                    "sample_answer": test["answer"],
                    "python_result": "",
                    "model_answer": None,
                    "lean4_result": "",
                    "python_correct": False,
                    "lean_correct": False,
                    "final_correct": False,
                    "item_id": global_item_id
                }
                all_samples.append(sample)
            global_item_id += 1
    
        dataset = Dataset.from_list(all_samples)
        
        for stage in stages:
            dataset = batch_generate(dataset, stage)
            if stage == "lean":
                dataset = process_batch_math(dataset, verifier)
            elif stage == "python":
                dataset = dataset.map(
                    process_python_batch,
                    batched=False,
                    num_proc=config.num_batch,
                    load_from_cache_file=False,
                )
            elif stage == "summary":
                dataset = dataset.map(
                    get_python_answer,
                    batched=False,
                    num_proc=config.num_batch,
                    load_from_cache_file=False,
                )
                dataset = dataset.map(
                    check_answer,
                    batched=False,
                    num_proc=config.num_batch,
                    load_from_cache_file=False,
                )
                
        item_samples_map = {}
        for sample in dataset:
            item_id = sample["item_id"]
            if item_id not in item_samples_map:
                item_samples_map[item_id] = []
            item_samples_map[item_id].append(sample)

        for item_id, samples in item_samples_map.items():
            selected_samples = []
            item_correct = False
            
            if math_maj:
                model_answers = [s.get("model_answer") for s in samples]
                python_answers = [s.get("python_result") for s in samples]
                lean_answers = [s.get("lean_result") for s in samples]
                
                model_answers_majority = Counter(model_answers).most_common(1)[0][0]
                python_answers_majority = Counter(python_answers).most_common(1)[0][0]
                lean_answers_majority = Counter(lean_answers).most_common(1)[0][0]
                
                if check_answer_maj(model_answers_majority, python_answers_majority, lean_answers_majority, str(samples[0]["sample_answer"])):
                    item_correct = True
                
                write_sample = {
                    "text": samples[0]["text"],
                    "sample_answer": samples[0]["sample_answer"],
                    "model_answer": model_answers,
                    "python_answers": python_answers,
                    "lean_answers": lean_answers,
                    "item_id": samples[0]["item_id"]
                }
                fw.write(json.dumps(write_sample, ensure_ascii=False) + "\n")
                fw.flush()
            else:
                correct_samples = [s for s in samples if s.get("python_correct") and s.get("lean_correct")]
                
                if correct_samples:
                    selected_samples = correct_samples
                    item_correct = True
                else:
                    final_correct_samples = [s for s in samples if s.get("final_correct")]
                    if final_correct_samples:
                        selected_samples = final_correct_samples
                        item_correct = True
                    else:
                        selected_samples = [random.choice(samples)]
                
                selected_sample = random.choice(selected_samples)
            
            
                fw.write(json.dumps(selected_sample, ensure_ascii=False) + "\n")
                fw.flush()
            
            if item_correct:
                complete_count += 1

    verifier.close()
    print("Accuracy", round((complete_count / total) * 100, 2), "%")

if __name__ == "__main__":
    @dataclass
    class Config:
        model_id: str
        output_path: str

        num_samples: int
        num_batch: int

        temperature: float
        max_new_tokens: int
    
    parser = argparse.ArgumentParser(description="Run the task with specific configurations.")
    parser.add_argument("--model", type=str, required=True, help="The model name.")
    parser.add_argument("--task", type=str, required=True, choices=["gsm8k", "math", "minif2f", "amc2023", "aime2024"],
                        help="The task to run. Must be one of ['gsm8k', 'math', 'minif2f', 'amc2023', 'aime2024'].")
    parser.add_argument("--math_maj", action="store_true", help="Whether to use math maj.")
    parser.add_argument("--num_samples", type=int, default=1, help="The number of examples to use for few-shot learning.")
    parser.add_argument("--second_stage_k", type=int, default=1, help="The number of examples to use for second stage generation.")
    parser.add_argument("--num_batch", type=int, default=256, help="Number of batches to process.")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum number of new tokens to generate.")
    args = parser.parse_args()
    
    model_path = args.model
    task = args.task
    test_data_path = f"./datasets/{task}/test.jsonl"
    
    model_version = model_path.split("/")[-1]
    
    output_name = f"./output/{task}/{model_version}_@{args.num_samples}.json"
    
    output_dir = os.path.dirname(output_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if task in ["math", "gsm8k"]:
        with open(test_data_path, 'r') as f:
            infer_list = [json.loads(line) for line in f]
            
        if task == "gsm8k":
            assert len(infer_list) == 1319
            for i in range(len(infer_list)):
                infer_list[i]["problem"] = infer_list[i]["question"]
                infer_list[i]["answer"] = infer_list[i]["answer"].split("#### ")[-1]
                del infer_list[i]["question"]
        else:
            assert len(infer_list) == 5000
        
    elif task == "minif2f":
        with open(test_data_path, 'r') as f:
            infer_list = [json.loads(line) for line in f]
            assert infer_list is not []
        assert len(infer_list) == 244
            
    elif task in ["amc2023", "aime2024"]:
        with open(test_data_path, 'r') as f:
            infer_list = [json.loads(line) for line in f]
        
        if task == "amc2023":
            assert len(infer_list) == 40
            for i in range(len(infer_list)):
                infer_list[i]["problem"] = infer_list[i]["question"]
                del infer_list[i]["question"]
        else:
            assert len(infer_list) == 30
            for i in range(len(infer_list)):
                infer_list[i]["answer"] = infer_list[i]["expected_answer"]
                del infer_list[i]["expected_answer"]
    else:
        raise Exception("Bad task name")
    
    
    config = Config(
        model_id = model_path,
        output_path = output_name,
        num_samples = args.num_samples,
        num_batch = args.num_batch,
        temperature = 1.0 if args.num_samples != 1 else 0.0,
        max_new_tokens = args.max_new_tokens,
    )
    
    if task == "minif2f":
        main_prove(config, infer_list)
    else:
        main_math(config, infer_list, args.math_maj)
