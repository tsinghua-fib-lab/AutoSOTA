"""
Baseline evaluation on MATH500 (test split of hendrycks-MATH-benchmark) using vllm.
This evaluates the base Qwen3-1.7B model without any steering.
"""
import os
import json
import torch
from argparse import ArgumentParser
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
import numpy as np
try:
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("Please install math-verify: pip install math-verify")

INSTRUCTION = r"""Solve the following math problem step by step. The last line of your response should be of the form Answer: \boxed{{$Answer}} where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:"."""


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
        print(f"[Warning] math_verify failed: {repr(e)}")
        return 0.0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="/models/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="/repo/results")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for testing")
    args = parser.parse_args()

    set_seed(42)

    model_name_display = args.model.split("/")[-1]
    if "Qwen3" in args.model and not args.enable_thinking:
        model_name_display = model_name_display + "_no_thinking"

    output_dir = os.path.join(args.output_dir, model_name_display, "vllm_generate_results")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading MATH500 (test split)...")
    ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="test")

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"Dataset size: {len(ds)}")

    def process_data(example):
        prompt = INSTRUCTION.format(problem=example["problem"])
        example["prompt"] = [{"role": "user", "content": prompt}]
        return example

    ds = ds.map(process_data)
    print(f"Sample prompt: {ds[0]['prompt'][0]['content'][:200]}...")

    prompts = [item["prompt"] for item in ds]

    print(f"Initializing vLLM with {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=42
    )

    if args.temperature == 0:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=42,
        )
    else:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            seed=42,
        )

    print(f"Running inference with temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}...")

    if args.enable_thinking:
        outputs = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=True)
    else:
        if "Qwen3" in args.model:
            outputs = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=True,
                              chat_template_kwargs={"enable_thinking": False})
        else:
            outputs = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=True)

    test_results = []
    correct = 0
    for i, output in enumerate(outputs):
        problem = ds[i]["problem"]
        model_pred = output.outputs[0].text
        ground_truth = ds[i]["answer"]
        score = compute_score(model_output=model_pred[-500:], ground_truth=ground_truth)
        if score:
            correct += 1
        test_results.append({
            "problem_id": i,
            "problem": problem,
            "model_pred": model_pred,
            "ground_truth": ground_truth,
            "score": float(score),
        })

    acc = correct / len(test_results)
    print(f"\n=== RESULTS ===")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Correct: {correct}/{len(test_results)}")

    output_file = os.path.join(output_dir, f"results_math500_t{args.temperature}_{args.max_tokens // 1024}k.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in test_results:
            f.write(json.dumps(item) + "\n")

    print(f"Results saved to {output_file}")

    # Also save summary
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "model": args.model,
            "dataset": "MATH500",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "accuracy": acc,
            "correct": correct,
            "total": len(test_results),
        }, f, indent=2)
    print(f"Summary saved to {summary_file}")
