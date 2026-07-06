#!/usr/bin/env python3
"""Generate MATH CoT responses with Qwen3-8B and create exp2 cache JSONL.

Only keeps correct responses (answer matches ground truth).
Uses the exp2 dataset_utils for span computation.
"""

import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from exp.exp2.dataset_utils import CachedExample, attach_spans_from_answer


def extract_boxed_answer(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} in text."""
    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i - 1]
    return None


def normalize_answer(ans: str) -> str:
    """Normalize an answer string for comparison."""
    ans = ans.strip()
    # remove LaTeX formatting
    ans = ans.replace('$', '')
    # collapse whitespace
    ans = re.sub(r'\s+', ' ', ans)
    # remove leading/trailing punctuation that isn't part of the answer
    ans = ans.strip('.,;:!?')
    return ans


def answers_match(pred: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer."""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True
    # Also try numeric comparison
    try:
        p_val = float(pred_norm.replace(',', ''))
        g_val = float(gold_norm.replace(',', ''))
        return abs(p_val - g_val) < 1e-6
    except (ValueError, TypeError):
        pass
    return False


def generate_cot_response(model, tokenizer, problem: str, max_new_tokens: int = 2048) -> str:
    """Generate a CoT response for a math problem."""
    sys_prompt = (
        "You are a reasoning assistant. "
        "Before answering, engage in a chain of thought. "
        "Process this freely and naturally without using specific headers or strict formatting. "
        "When you reach the conclusion, wrap the entire final sentence containing the answer inside \\box{}. "
        "Ensure the box wraps the sentence that naturally delivers the answer."
    )
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": problem},
    ]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_json", type=str, default="/repo/data/math_problems.json")
    parser.add_argument("--out_jsonl", type=str, default="/repo/exp/exp2/data/math.jsonl")
    parser.add_argument("--model_path", type=str, default="/models/Qwen3-8B-Instruct")
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--max_attempts", type=int, default=300)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load problems
    with open(args.in_json) as f:
        problems = json.load(f)

    # Shuffle for diversity
    import random
    random.seed(args.seed)
    random.shuffle(problems)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"Model loaded on {model.device}")

    # Generate responses
    kept = []
    attempted = 0
    correct = 0
    pbar = tqdm(total=args.num_examples, desc="Correct samples")

    for prob in problems:
        if len(kept) >= args.num_examples:
            break
        if attempted >= args.max_attempts:
            break

        attempted += 1
        problem = prob["problem"]
        gold_answer = prob["answer"]

        try:
            generation = generate_cot_response(model, tokenizer, problem, args.max_new_tokens)
        except Exception as e:
            print(f"\n[ERROR] Generation failed for problem {attempted}: {e}")
            continue

        # Extract boxed answer
        pred_answer = extract_boxed_answer(generation)
        if pred_answer is None:
            continue

        # Check correctness
        if not answers_match(pred_answer, gold_answer):
            continue

        correct += 1

        # Compute token spans
        target_text = generation  # The full CoT + answer generation
        tokenizer_kwargs = dict(add_special_tokens=False)
        target_ids = tokenizer(target_text, **tokenizer_kwargs).input_ids

        # Find the boxed answer span in the target text
        # Use attach_spans_from_answer from dataset_utils
        example = CachedExample(
            prompt=problem,
            target=target_text,
            indices_to_explain=None,
            attr_mask_indices=None,
            sink_span=None,
            thinking_span=None,
            metadata={
                "dataset": "math",
                "reference_answer": gold_answer,
                "boxed_answer": pred_answer,
                "subject": prob.get("subject", ""),
                "level": prob.get("level", ""),
            },
        )
        example = attach_spans_from_answer(example, tokenizer, pred_answer)
        if not (isinstance(example.sink_span, list) and len(example.sink_span) == 2):
            continue

        indices_to_explain = list(example.sink_span)
        entry = {
            "prompt": problem,
            "target": target_text,
            "indices_to_explain": indices_to_explain,
            "attr_mask_indices": None,
            "sink_span": indices_to_explain,
            "thinking_span": example.thinking_span,
            "metadata": {
                "dataset": "math",
                "reference_answer": gold_answer,
                "boxed_answer": pred_answer,
                "subject": prob.get("subject", ""),
                "level": prob.get("level", ""),
            },
        }
        kept.append(entry)
        pbar.update(1)

    pbar.close()

    # Write cache
    out_dir = os.path.dirname(args.out_jsonl)
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for entry in kept:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nAttempted: {attempted}, Correct: {correct}, Kept: {len(kept)}")
    print(f"Cache written to {args.out_jsonl}")


if __name__ == "__main__":
    main()
