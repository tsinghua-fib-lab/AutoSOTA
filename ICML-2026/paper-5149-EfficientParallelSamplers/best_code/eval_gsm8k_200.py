"""Evaluate GSM8K with diffusion sampler using lm-eval - 200 samples."""
import sys
import os
import json
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/autosota_cache/hf"

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Diffusion sampler params
gen_kwargs_str = (
    "temperature=0.2,top_p=0.95,do_sample=True,"
    "headway=1,inner_recurrence=4,state_noise_mixing=0.0,"
    "ema_embeds=0.1,exit_t=0.03,max_wavefront=128,"
    "freeze_strategy=latent-diff,num_steps=32"
)

print(f"Gen kwargs: {gen_kwargs_str}")

print("Creating HFLM model wrapper...")
lm = HFLM(
    pretrained="/models/huginn-0125",
    trust_remote_code=True,
    dtype="bfloat16",
    batch_size=1,
    device="cuda",
)

print(f"Model type: {type(lm.model).__name__}")

total_start = time.time()
results = evaluator.simple_evaluate(
    model=lm,
    tasks=["gsm8k_cot"],
    num_fewshot=8,
    batch_size=1,
    limit=200,
    apply_chat_template=True,
    fewshot_as_multiturn=True,
    system_instruction="You are a helpful assistant that can assist users with mathematical reasoning.",
    gen_kwargs=gen_kwargs_str,
)
total_elapsed = time.time() - total_start

if results is not None:
    task_results = results.get("results", {}).get("gsm8k_cot", {})
    n_samples = task_results.get("sample_len", 0)
    acc_val = task_results.get("exact_match,flexible-extract", 0)
    print(f"\n=== RESULTS (limit=200) ===")
    print(json.dumps(task_results, indent=2, default=str))
    print(f"\nAccuracy (flexible-extract): {acc_val}")
    print(f"Samples: {n_samples}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    if n_samples > 0:
        print(f"Avg time per sample: {total_elapsed/n_samples:.1f}s")
    
    output = {
        "paper_id": 5149,
        "task": "gsm8k_cot",
        "model": "/models/huginn-0125",
        "sampler": "diffusion_forcing",
        "params": {
            "inner_recurrence": 4,
            "state_noise_mixing": 0.0,
            "ema_embeds": 0.1,
            "exit_t": 0.03,
            "max_wavefront": 128,
            "freeze_strategy": "latent-diff",
            "num_steps": 32,
            "temperature": 0.2,
            "top_p": 0.95,
            "fewshot": 8,
        },
        "results": task_results,
        "accuracy": acc_val,
        "total_time_s": total_elapsed,
        "n_samples": n_samples,
    }
    with open("/repo/gsm8k_diffusion_200_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to /repo/gsm8k_diffusion_200_results.json")
