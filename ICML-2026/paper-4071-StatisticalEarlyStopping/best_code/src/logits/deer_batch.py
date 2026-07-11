########
# Code Snippet Adapted from 
# https://github.com/iie-ycx/DEER/blob/main/vllm-deer.py
# and
# https://github.com/chicosirius/think-or-not/blob/main/ThinkorNot/adaptive_think.py


import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings
import os
from dotenv import load_dotenv

load_dotenv()

from src.inference import Inference
import json
import torch
from vllm.outputs import CompletionOutput
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import SamplingParams

import math
import numpy as np
import random
import argparse

from src.probe.train import build_prompt_samples
from data import get_dataset_generator
from src.utils import load_data


# =========================
# General utilities
# =========================

def set_seeds(seed=42):
    # Set Python built-in random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch CPU random seed
    torch.manual_seed(seed)

    # If using GPU (especially CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)           # Set seed for current GPU
        torch.cuda.manual_seed_all(seed)       # Also effective for multi-GPU

        # For better reproducibility, enable cudnn determinism mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Optional: Set generator (for DataLoader with multi-threading)
    g = torch.Generator()
    g.manual_seed(seed)


def append_jsonl(data, file_path):
    """Append results in the list to a .jsonl file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


# =========================
# pred_prob from logprobs
# =========================

def calculate_average_max_prob_from_logprobs(
    logprobs_list: List[Dict[Any, Any]],
    policy: str = 'avg2'
) -> float:
    """
    Calculate average max token probability from logprobs list in vLLM CompletionOutput.
    Compute from the second generated token to the last token.
    policy:
        - 'min'  : minimum probability over steps
        - 'avg1' : arithmetic mean
        - 'avg2' : geometric mean (default)
    """
    num_tokens = len(logprobs_list)
    start_index = 1
    end_index = num_tokens

    if num_tokens < 1:
        print("Too few tokens to calculate valid average.")
        return 0.0

    total_prob_sum = 0.0
    log_prob_sum = 0.0  # For geometric mean
    count_for_average = 0
    min_prob = 1.0

    for i in range(start_index, end_index):
        if i < len(logprobs_list) and logprobs_list[i]:
            try:
                # Take one of the top-k candidates (the max one is at index 0 after sorting,
                # but here we just use the first in the dict as in your original code)
                logprob_obj = list(logprobs_list[i].values())[0]
                if hasattr(logprob_obj, 'logprob'):
                    prob = torch.exp(torch.tensor(logprob_obj.logprob)).item()
                    if prob < min_prob:
                        min_prob = prob
                    total_prob_sum += prob
                    log_prob_sum += math.log(max(prob, 1e-10))
                    count_for_average += 1
                else:
                    print(f"Warning: Object at logprobs_list[{i}] doesn't have '.logprob' attribute.")
            except (IndexError, KeyError, AttributeError) as e:
                print(f"Warning: Unable to process logprobs at logprobs_list[{i}]: {e}")
        else:
            print(f"Warning: logprobs_list[{i}] is empty or invalid.")

    if count_for_average == 0:
        return 0.0

    if policy == 'min':
        result = min_prob
    elif policy == 'avg1':
        result = total_prob_sum / count_for_average
    elif policy == 'avg2':
        result = math.exp(log_prob_sum / count_for_average)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    return result


def prefixes_until_each_continue(full_prompt: str, continue_str: str) -> List[str]:
    parts = full_prompt.split(continue_str)

    prefixes = []
    running = ""

    for i in range(len(parts) - 1):  # each continue_str occurrence
        running += parts[i]
        prefixes.append(running)

    return prefixes


# =========================
# Answer-level entropy via beam search over \boxed{...}
# =========================


def extract_last_boxed(text: str) -> str:
    """Extract the content of the last closed \\boxed{...} in the text."""
    # simply find the \boxed{ and extract the last one
    start_idx = text.rfind("\\boxed{")
    end_idx = text.find('}', start_idx)
    if start_idx == -1:
        return ""
    return text[start_idx + len("\\boxed{"):end_idx]


def batch_beam_search_answer_entropy(
    prompts: List[str],
    llm_engine,
    tokenizer,
    beam_size: int = 5,
    max_steps: int = 10,
    logprob_top_k: int = None,
) -> List[Tuple[float, Dict[str, float]]]:
    """
    Process multiple prompts simultaneously with beam search.
    
    This batches beam candidates across multiple prompts for maximum speedup.
    
    Args:
        prompts: List of prompts to process
        llm_engine: vLLM engine instance
        tokenizer: HF tokenizer
        beam_size: Beam width b
        max_steps: Max decoding steps T
        logprob_top_k: Number of tokens in logprobs; if None, defaults to beam_size
    
    Returns:
        List of (H_answer, answer_probs) tuples, one per input prompt
    """
    
    if logprob_top_k is None:
        logprob_top_k = beam_size
    
    # Track state for each prompt independently
    beams = [[(p, 1.0)] for p in prompts]  # List of beams
    completed_list = [{} for _ in prompts]  # List of completed dicts
    active_indices = list(range(len(prompts)))  # Which prompts still have active beams
    
    for t in range(max_steps):
        if not active_indices:
            break
        
        # Collect all active beam candidates across all prompts
        batch_prompts = []
        batch_metadata = []  # (prompt_idx, context, prefix_prob)
        
        for prompt_idx in active_indices:
            beam = beams[prompt_idx]
            for (context, prefix_prob) in beam:
                batch_prompts.append(context)
                batch_metadata.append((prompt_idx, context, prefix_prob))
        
        if not batch_prompts:
            break
        
        # Single batched generation for ALL beams across ALL prompts
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=1.0,
            logprobs=logprob_top_k,
            top_k=logprob_top_k,
        )
        
        batch_outputs = llm_engine.generate(batch_prompts, sampling_params, use_tqdm=False)
        
        # Organize new candidates by prompt
        new_beams = [[] for _ in prompts]
        
        for (prompt_idx, context, prefix_prob), output in zip(batch_metadata, batch_outputs):
            comp: CompletionOutput = output.outputs[0]
            if not comp.logprobs or not comp.logprobs[0]:
                continue
            
            step_logprobs = comp.logprobs[0]
            candidates = sorted(
                step_logprobs.items(),
                key=lambda kv: kv[1].logprob,
                reverse=True
            )[:beam_size]
            
            for tok_key, logprob_obj in candidates:
                token_str = getattr(logprob_obj, "decoded_token", None)
                if token_str is None:
                    if isinstance(tok_key, int):
                        token_str = tokenizer.decode([tok_key])
                    else:
                        token_str = str(tok_key)
                
                prob = math.exp(logprob_obj.logprob)
                new_prob = prefix_prob * prob
                new_context = context + token_str
                
                if '}' in token_str:
                    ans = extract_last_boxed(new_context)
                    if ans:
                        completed_list[prompt_idx][ans] = \
                            completed_list[prompt_idx].get(ans, 0.0) + new_prob
                else:
                    new_beams[prompt_idx].append((new_context, new_prob))
        
        # Update beams and active indices
        new_active_indices = []
        for prompt_idx in active_indices:
            if new_beams[prompt_idx]:
                new_beams[prompt_idx].sort(key=lambda cp: cp[1], reverse=True)
                beams[prompt_idx] = new_beams[prompt_idx][:beam_size]
                new_active_indices.append(prompt_idx)
        
        active_indices = new_active_indices
    
    # Compute results for each prompt
    results = []
    for completed in completed_list:
        Z = sum(completed.values())
        if Z == 0.0:
            results.append((0.0, {}))
            continue
        
        for s in completed:
            completed[s] = completed[s] / Z
        
        H_answer = -sum(p_s * math.log(p_s) for p_s in completed.values())
        results.append((H_answer, completed))
    
    return results


# =========================
# Main logic with batching
# =========================

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-8B')
    parser.add_argument('--data_name', type=str, default='gsm8k')
    parser.add_argument('--prompt_batch_size', type=int, default=8, 
                        help='Number of prompts to process simultaneously')
    return parser

def run(llm_engine, tokenizer, prompt_samples, output_path, prompt_batch_size):
    """
    Run inference with batched beam search for speedup.
    
    Args:
        llm_engine: vLLM engine
        tokenizer: HF tokenizer
        prompt_samples: List of samples with prompts and labels
        output_path: Where to save results
        prompt_batch_size: Number of prompts to batch together
    """

    # set the start index
    start_idx = 0
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            all_results = [json.loads(line) for line in f.readlines()]
            start_idx = len(all_results)

    # Process prompts in batches
    for batch_start in tqdm(range(start_idx, len(prompt_samples), prompt_batch_size), 
                            desc="Processing prompt batches"):
        batch_end = min(batch_start + prompt_batch_size, len(prompt_samples))
        batch = prompt_samples[batch_start:batch_end]
        
        # Collect all prefixes and metadata across the batch
        all_prompts_for_prob = []
        all_prompts_for_entropy = []
        batch_metadata = []  # (sample_idx, prefix_idx, sample)
        
        for sample_idx, sample in enumerate(batch):
            prompt = sample["prompt"]
            prompt_prefixes = prefixes_until_each_continue(prompt, continue_str)
            
            for prefix_idx, prefix in enumerate(prompt_prefixes):
                prompt_for_prob_check = prefix + answer_prompt_str
                all_prompts_for_prob.append(prompt_for_prob_check)
                all_prompts_for_entropy.append(prompt_for_prob_check)
                batch_metadata.append((batch_start + sample_idx, prefix_idx, sample))
        
        if not all_prompts_for_prob:
            # If no prefixes in this entire batch, still save empty results
            results_by_prompt = {}
            for sample_idx, sample in enumerate(batch):
                prompt_idx = batch_start + sample_idx
                results_by_prompt[prompt_idx] = {
                    "prompt_index": prompt_idx,
                    "pred_prob_list": [],
                    "H_bits_list": [],
                    "label": sample["label"]
                }
            for prompt_idx in sorted(results_by_prompt.keys()):
                append_jsonl(results_by_prompt[prompt_idx], output_path)
            continue
        
        # ---------------------------
        # 1) Batch generation for pred_prob
        # ---------------------------
        prob_check_max_tokens = 20
        
        sampling_params_for_batch = SamplingParams(
            max_tokens=prob_check_max_tokens,
            stop=pred_prob_stop_tokens,
            logprobs=1,
        )
        
        batch_outputs = llm_engine.generate(
            all_prompts_for_prob,
            sampling_params_for_batch,
            use_tqdm=False
        )
        
        # Calculate pred_prob for each output
        pred_probs = []
        for output in batch_outputs:
            completion_output = output.outputs[0]
            if completion_output.logprobs:
                pred_prob = calculate_average_max_prob_from_logprobs(
                    completion_output.logprobs,
                    policy='avg2'
                )
            else:
                pred_prob = 0.0
            pred_probs.append(pred_prob)
        
        # ---------------------------
        # 2) Batch beam search for entropy
        # ---------------------------
        entropy_results = batch_beam_search_answer_entropy(
            all_prompts_for_entropy,
            llm_engine=llm_engine,
            tokenizer=tokenizer,
            beam_size=5,
            max_steps=10,
        )
        
        H_bits_list = []
        for H_answer, _ in entropy_results:
            if H_answer > 0:
                H_bits = H_answer / math.log(2.0)
            else:
                H_bits = 0.0
            H_bits_list.append(H_bits)
        
        # ---------------------------
        # 3) Reorganize results by original prompt
        # ---------------------------
        # Initialize results for all samples in batch (including those with no prefixes)
        results_by_prompt = {}
        for sample_idx, sample in enumerate(batch):
            prompt_idx = batch_start + sample_idx
            results_by_prompt[prompt_idx] = {
                "prompt_index": prompt_idx,
                "pred_prob_list": [],
                "H_bits_list": [],
                "label": sample["label"]
            }
        
        # Fill in results for samples that had prefixes
        for (prompt_idx, prefix_idx, sample), pred_prob, H_bits in zip(
            batch_metadata, pred_probs, H_bits_list
        ):
            results_by_prompt[prompt_idx]["pred_prob_list"].append(pred_prob)
            results_by_prompt[prompt_idx]["H_bits_list"].append(H_bits)
        
        # Write results in order
        for prompt_idx in sorted(results_by_prompt.keys()):
            append_jsonl(results_by_prompt[prompt_idx], output_path)


if __name__ == "__main__":
    set_seeds(42)

    args = get_args_parser().parse_args()
    model_name = args.model_name
    prompt_batch_size = args.prompt_batch_size
    
    # Your vLLM inference engine
    llm_engine = Inference(model_name=model_name).llm

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Prompt pieces
    answer_prompt_str = "\n**Final Answer**\n\\boxed{"
    continue_str = "Wait"
    continue_ids = tokenizer.encode(continue_str, add_special_tokens=False)
    last_token_strs = ["</think>"]

    # Stop tokens for thinking phase generation
    generation_stop_tokens = [continue_str] + last_token_strs + [tokenizer.eos_token]
    # Stop tokens for answer generation (after closing the box)
    pred_prob_stop_tokens = [
        ' }', '}\n', '}\n\n', '}.', '}.\n', '}\\', '}}', ')}', ')}.', ')}\n'
    ]

    # Load data and build prompts
    data_name = args.data_name
    data_generator = get_dataset_generator(data_name)

    if data_name == "gsm8k":
        for split in ['train', 'test']:
            data_ = load_data(model_name, "gsm8k", split = split)
            if split == "test":
                output_path = f"logits/{model_name}/gsm8k_cal_pred_prob_entropy.jsonl"
            else:
                output_path = f"logits/{model_name}/gsm8k_train_pred_prob_entropy.jsonl"
            
            prompt_samples = build_prompt_samples(data_, data_generator, tokenizer)
            run(llm_engine, tokenizer, prompt_samples, output_path, prompt_batch_size)
    else:
        data_ = load_data(model_name, data_name, split = 'full')
        output_path = f"logits/{model_name}/{data_name}_pred_prob_entropy.jsonl"
        prompt_samples = build_prompt_samples(data_, data_generator, tokenizer)
        run(llm_engine, tokenizer, prompt_samples, output_path, prompt_batch_size)