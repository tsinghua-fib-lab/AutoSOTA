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


def beam_search_answer_entropy(
    prompt: str,
    llm_engine,
    tokenizer,
    beam_size: int = 5,
    max_steps: int = 10,
    logprob_top_k: int = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Beam-search-based entropy estimation over full \boxed{...} answers.

    Args:
        prompt: Prompt up to where final answer generation starts
                (e.g. "... **Final Answer** \\boxed").
        llm_engine: vLLM engine instance.
        tokenizer: HF tokenizer.
        beam_size: Beam width b.
        max_steps: Max decoding steps T.
        logprob_top_k: Number of tokens in logprobs; if None, defaults to beam_size.

    Returns:
        H_answer: Entropy over answers (natural log).
        answer_probs: Dict[answer_string, normalized_probability].
    """

    if logprob_top_k is None:
        logprob_top_k = beam_size

    # Beam is a list of (context_string, cumulative_probability)
    beam: List[Tuple[str, float]] = [(prompt, 1.0)]
    completed: Dict[str, float] = {}

    for t in range(max_steps):
        if not beam:
            break

        new_beam: List[Tuple[str, float]] = []

        prompts = [ctx for (ctx, p) in beam]
        sampling_params = [
            SamplingParams(
                max_tokens=1,
                temperature=1.0,
                logprobs=logprob_top_k,
                top_k=logprob_top_k,
            )
            for _ in prompts
        ]

        batch_outputs = llm_engine.generate(prompts, sampling_params, use_tqdm=False)

        for (context, prefix_prob), output in zip(beam, batch_outputs):
            comp: CompletionOutput = output.outputs[0]
            if not comp.logprobs or not comp.logprobs[0]:
                continue

            step_logprobs = comp.logprobs[0]  # dict: token_id -> Logprob object

            # Sort candidates by probability (descending) and take top beam_size
            candidates = sorted(
                step_logprobs.items(),
                key=lambda kv: kv[1].logprob,
                reverse=True
            )[:beam_size]

            for tok_key, logprob_obj in candidates:
                # Robustly decode the token string
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
                        completed[ans] = completed.get(ans, 0.0) + new_prob
                else:
                    new_beam.append((new_context, new_prob))
        
        if not new_beam:
            break

        # Keep top-b elements by probability
        new_beam.sort(key=lambda cp: cp[1], reverse=True)
        beam = new_beam[:beam_size]

    Z = sum(completed.values())
    if Z == 0.0:
        return 0.0, {}

    # Normalize probabilities
    for s in completed:
        completed[s] = completed[s] / Z

    # Compute entropy H = - Σ_s p_s log p_s  (in nats)
    H_answer = -sum(p_s * math.log(p_s) for p_s in completed.values())

    return H_answer, completed


# =========================
# Main logic
# =========================

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-8B')
    parser.add_argument('--data_name', type=str, default='gsm8k')
    return parser

def run(llm_engine, tokenizer, prompt_samples, output_path):

    # set the start index
    start_idx = 0
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            all_results = [json.loads(line) for line in f.readlines()]
            start_idx = len(all_results)

    # Iterate over ALL prompts
    for prompt_idx, sample in enumerate(tqdm(prompt_samples[start_idx:], desc="Processing prompts")):
        prompt = sample["prompt"]

        # Split the prompt by continue_str → prefixes
        prompt_prefixes = prefixes_until_each_continue(prompt, continue_str)

        pred_prob_list = []
        H_bits_list = []

        for _, prefix in enumerate(prompt_prefixes):
            prompt_for_prob_check = prefix + answer_prompt_str

            try:
                # ---------------------------
                # 1) Single-sample generation for pred_prob
                # ---------------------------
                prob_check_max_tokens = 20

                sampling_params_for_batch = SamplingParams(
                    max_tokens=prob_check_max_tokens,
                    stop=pred_prob_stop_tokens,  # Only predict content inside \boxed{}
                    logprobs=1,
                )

                batch_outputs = llm_engine.generate(
                    [prompt_for_prob_check],
                    [sampling_params_for_batch],
                    use_tqdm=False
                )

                completion_output = batch_outputs[0].outputs[0]

                if completion_output.logprobs:
                    pred_prob = calculate_average_max_prob_from_logprobs(
                        completion_output.logprobs,
                        policy='avg2'
                    )
                else:
                    pred_prob = 0.0

                pred_prob_list.append(pred_prob)

                # ---------------------------
                # 2) Answer-level entropy via beam search
                # ---------------------------
                H_answer, _ = beam_search_answer_entropy(
                    prompt_for_prob_check,
                    llm_engine=llm_engine,
                    tokenizer=tokenizer,
                    beam_size=5,
                    max_steps=10,
                )

                if H_answer > 0:
                    H_bits = H_answer / math.log(2.0)
                else:
                    H_bits = 0.0

                H_bits_list.append(H_bits)
            except Exception as e:
                pass

        # Store results for this prompt
        result = {
            "prompt_index": prompt_idx,
            "pred_prob_list": pred_prob_list,   # 1/ list of pred_prob over prefixes
            "H_bits_list": H_bits_list,         # 2/ list of H_bits over prefixes
            "label": sample["label"],           # 3/ label (1 for correct, 0 for incorrect)
        }
        append_jsonl(result, output_path)

if __name__ == "__main__":
    set_seeds(42)

    args = get_args_parser().parse_args()
    model_name = args.model_name
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
            run(llm_engine, tokenizer, prompt_samples, output_path)
    else:
        data_ = load_data(model_name, data_name, split = 'full')
        output_path = f"logits/{model_name}/{data_name}_pred_prob_entropy.jsonl"
        prompt_samples = build_prompt_samples(data_, data_generator, tokenizer)
        run(llm_engine, tokenizer, prompt_samples, output_path)