import os
import json
import torch
import math
from argparse import ArgumentParser
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as np

try:
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("Please install math-verify: pip install math-verify")

def compute_score(model_output: str, ground_truth: str) -> bool:
    try:
        verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
        ret_score = 0.0
        ground_truth_boxed = "\\boxed{" + ground_truth + "}"
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
        return ret_score
    except BaseException as e:
        print(f"[Warning] math_verify failed for this sample: {repr(e)}") 
        return 0.0

def calculate_chunk_entropy_vllm(output):
    logprobs_list = output.logprobs
    
    if not logprobs_list:
        return 0.0, []

    token_entropies = []

    for step_dict in logprobs_list:
        lps = torch.tensor([lp.logprob for lp in step_dict.values()], dtype=torch.float32)
        
        if lps.numel() <= 1:
            token_entropies.append(0.0)
            continue

        log_probs_norm = lps - torch.logsumexp(lps, dim=0)
        
        probs_norm = torch.exp(log_probs_norm)
        entropy = -torch.sum(probs_norm * log_probs_norm)
        
        token_entropies.append(max(0.0, entropy.item()))
    
    return np.mean(token_entropies)

def get_finished_tasks(save_path):
    finished = set()
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    finished.add(f"{d['problem_id']}_{d['chunk_idx']}")
                except: continue
    return finished


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dir", type=str, required=True, help="dir to chunk_metadata.json")
    parser.add_argument("--batch_size", type=int, default=128, help="Processing batch size (number of chunks)")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    args = parser.parse_args()

    meta_path = os.path.join(args.dir, "chunk_metadata.json")
    encountered_dot_tokens_path = os.path.join(args.dir, "total_dot_tokens.json")
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    with open(encountered_dot_tokens_path, "r", encoding="utf-8") as f:
        encountered_dot_tokens = json.load(f)

    stop_strings = list(set(encountered_dot_tokens.values()))
    save_path = meta_path.replace(".json", "_sampled_results.jsonl")
    finished_tasks = get_finished_tasks(save_path)

    print(f"Initializing vLLM with {args.model}...")
    llm = LLM(
        model=args.model, 
        tensor_parallel_size=torch.cuda.device_count(), 
        seed=42,
    )
    
    fork_params = SamplingParams(
        temperature=1.0, 
        n=args.n_samples, 
        stop=stop_strings, 
        max_tokens=args.max_tokens // 2,
        logprobs=20,
        include_stop_str_in_output=True
    )
    completion_params = SamplingParams(
        temperature=args.temperature if args.temperature > 0 else 0.0,
        top_p=args.top_p if args.temperature > 0 else 1.0,
        top_k=args.top_k if args.temperature > 0 else -1,
        max_tokens=args.max_tokens,
        n=3
    )

    all_tasks = []
    for p_idx, item in enumerate(meta_data):
        for c_idx, chunk in enumerate(item["chunks"]):
            task_key = f"{p_idx}_{c_idx}"
            if task_key in finished_tasks:
                continue
            
            context = item["pre_text"] + "".join([c["text"] for c in item["chunks"][:c_idx]])
            all_tasks.append({
                "problem_id": p_idx,
                "chunk_idx": c_idx,
                "problem": item["problem"],
                "context": context,
                "ground_truth": item["ground_truth"],
                "avg_entropy": chunk["avg_entropy"],
                "avg_entropy_vllm": chunk["avg_entropy_vllm"],
                "original_score": item["score"],
                "ori_chunk_text": chunk["text"]
            })

    print(f"Total tasks: {len(all_tasks)}. Batches: {math.ceil(len(all_tasks)/args.batch_size)}")

    with open(save_path, "a", encoding="utf-8") as f_out:
        for i in tqdm(range(0, len(all_tasks), args.batch_size), desc="Batches"):
            batch = all_tasks[i : i + args.batch_size]
            
            batch_contexts = [t["context"] for t in batch]
            fork_outputs = llm.generate(batch_contexts, sampling_params=fork_params, use_tqdm=False)
            
            comp_prompts = []
            comp_entropies = []
            comp_task_map = []
            comp_fork_texts = [] 
            
            for task_idx, output in enumerate(fork_outputs):
                fork_out_text_sets = set()
                for fork_out in output.outputs:
                    if fork_out.text in fork_out_text_sets:
                        continue
                    fork_out_text_sets.add(fork_out.text)
                    comp_prompts.append(batch[task_idx]["context"] + fork_out.text)
                    comp_task_map.append(task_idx)
                    comp_fork_texts.append(fork_out.text)
                    comp_entropies.append(calculate_chunk_entropy_vllm(fork_out))

            if not comp_prompts: continue

            comp_outputs = llm.generate(comp_prompts, sampling_params=completion_params, use_tqdm=False)
            
            batch_fork_results = [[] for _ in range(len(batch))]
            batch_fork_texts = [[] for _ in range(len(batch))]
            batch_fork_entropies = [[] for _ in range(len(batch))]

            for j, out in enumerate(comp_outputs):
                task_idx = comp_task_map[j]
                current_fork_text = comp_fork_texts[j]
                
                sub_scores = []
                for sample in out.outputs:
                    full_resp = comp_prompts[j] + sample.text
                    is_correct = compute_score(full_resp[-500:], batch[task_idx]["ground_truth"])
                    sub_scores.append(1.0 if is_correct else 0.0)
                
                correct_count = sum(sub_scores)
                if correct_count == len(sub_scores):
                    final_chunk_score = 1.0
                elif correct_count > 0:
                    final_chunk_score = 0.5
                else:
                    final_chunk_score = 0.0
                
                batch_fork_results[task_idx].append(final_chunk_score)
                batch_fork_texts[task_idx].append(current_fork_text)
                batch_fork_entropies[task_idx].append(comp_entropies[j])

            lines = []
            for task_idx in range(len(batch)):
                res_json = {
                    "problem_id": batch[task_idx]["problem_id"],
                    "chunk_idx": batch[task_idx]["chunk_idx"],
                    "problem": batch[task_idx]["problem"],
                    "ground_truth": batch[task_idx]["ground_truth"],
                    "context": batch[task_idx]["context"],
                    "ori_chunk_text": batch[task_idx]["ori_chunk_text"],
                    "ori_chunk_entropy": batch[task_idx]["avg_entropy_vllm"],
                    "original_score": batch[task_idx]["original_score"],
                    "sampled_chunk_text": batch_fork_texts[task_idx],
                    "sampled_chunk_entropy": batch_fork_entropies[task_idx],
                    "sampled_scores": batch_fork_results[task_idx]
                }
                lines.append(json.dumps(res_json, ensure_ascii=False) + "\n")
            
            f_out.writelines(lines)
            f_out.flush()

    print(f"Sampled results saved to: {save_path}")