import os
import json
from argparse import ArgumentParser
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
import numpy as np
import torch
try:
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

INSTRUCTION = r"""Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{{$Answer}} where $Answer is the answer to the problem.

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
        print(f"[Warning] math_verify failed for this sample: {repr(e)}") 
        return 0.0

def calculate_entropy_vllm(request_output):
    output = request_output.outputs[0]
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
    
    return token_entropies

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str,default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", type=str, choices=["math", "gsm8k"], default="math")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--enable_thinking", action="store_true")
    args = parser.parse_args()

    set_seed(42)

    if "Qwen3" in args.model:
        model_name = args.model + "_no_thinking" if not args.enable_thinking else args.model
    else:
        model_name = args.model
    output_dir = os.path.join(args.output_dir, f"{model_name.split('/')[-1]}", "vllm_generate_results")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.dataset == "math":
        ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
    elif args.dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
    
    def process_data(example):
        if args.dataset == "gsm8k":
            example["answer"] = example["answer"].split("####")[-1].strip()
            example["problem"] = example["question"]
            example.pop("question")
        
        prompt = INSTRUCTION.format(problem=example["problem"])
        example["prompt"] = [{"role": "user", "content": prompt}]
        return example

    ds = ds.map(process_data)
    print(ds[0])

    prompts = [item["prompt"] for item in ds]
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        seed=42
    )
    if args.temperature == 0:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            seed=42,
            logprobs=20
        )
    if args.enable_thinking:
        outputs = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=True)
    else:
        if "Qwen3" in args.model:
            outputs = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})
        else:
            outputs = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=True)
    test_results = []

    for i, output in enumerate(outputs):
        problem = ds[i]["problem"]
        prompt = prompts[i]
        model_pred = output.outputs[0].text
        entropies = calculate_entropy_vllm(output)
        assert len(output.outputs[0].token_ids) == len(entropies), f"Length mismatch: {len(output.outputs[0].token_ids)} vs {len(entropies)}"
        ground_truth = ds[i]["answer"]
        score = compute_score(model_output=model_pred[-500:], ground_truth=ground_truth)
        test_results.append({
            "problem": problem,
            "prompt": prompt,
            "model_pred": model_pred,
            "entropies": entropies,
            "ground_truth": ground_truth,
            "score": score,
        })
    
    output_dir = os.path.join(output_dir, f"results_{args.max_tokens // 1024}k.jsonl")
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    
    print(f"Test results saved to {output_dir}")

    acc = sum([1 if item["score"] else 0 for item in test_results]) / len(test_results)
    print(f"Accuracy: {acc:.4f}, {sum([1 if item['score'] else 0 for item in test_results])}/{len(test_results)}")
    with open(output_dir, "w", encoding="utf-8") as f:
        for item in test_results:
            f.write(json.dumps(item) + "\n")
    print(f"Test results saved to {output_dir}")