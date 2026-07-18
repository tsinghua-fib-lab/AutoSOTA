"""Evaluate GSM8K with diffusion sampler using lm-eval."""
import os, json, time, torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

model_path = os.environ.get("MODEL_PATH", "/models/huginn-0125")
limit = int(os.environ.get("LIMIT", "0"))  # 0 = no limit (full)

print(f"Loading model from {model_path}...")
lm = HFLM(
    pretrained=model_path,
    trust_remote_code=True,
    dtype="bfloat16",
    batch_size=1,
    device="cuda",
)

gen_kwargs = (
    "temperature=0.2,top_p=0.95,do_sample=True,"
    "headway=1,inner_recurrence=4,state_noise_mixing=0.2,"
    "ema_embeds=0.1,exit_t=0.03,max_wavefront=128,"
    "freeze_strategy=latent-diff,num_steps=32"
)

print(f"Gen kwargs: {gen_kwargs}")
effective_limit = limit if limit > 0 else None
print(f"Running GSM8K CoT evaluation (limit={effective_limit})...")

start = time.time()
results = evaluator.simple_evaluate(
    model=lm,
    tasks=["gsm8k_cot"],
    num_fewshot=8,
    batch_size=1,
    limit=effective_limit,
    apply_chat_template=True,
    fewshot_as_multiturn=True,
    system_instruction="You are a helpful assistant that can assist users with mathematical reasoning.",
    gen_kwargs=gen_kwargs,
)
elapsed = time.time() - start

if results:
    task = results.get("results", {}).get("gsm8k_cot", {})
    n = task.get("sample_len", 0)
    acc = task.get("exact_match,flexible-extract", 0)
    acc_strict = task.get("exact_match,strict-match", 0)

    print(f"\n=== GSM8K Diffusion Sampler Results ===")
    print(f"Samples: {n}")
    print(f"Accuracy (flexible-extract): {acc:.4f} ({acc*100:.2f}%)")
    print(f"Accuracy (strict-match): {acc_strict:.4f} ({acc_strict*100:.2f}%)")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    if n > 0:
        print(f"Avg time/sample: {elapsed/n:.1f}s")

    result_json = {
        "paper_id": 5149,
        "model": model_path,
        "benchmark": "gsm8k_cot",
        "sampler": "diffusion_forcing",
        "settings": {
            "inner_recurrence": 4,
            "state_noise_mixing": 0.2,
            "ema_embeds": 0.1,
            "exit_t": 0.03,
            "max_wavefront": 128,
            "freeze_strategy": "latent-diff",
            "num_steps": 32,
            "temperature": 0.2,
            "top_p": 0.95,
            "fewshot": 8,
        },
        "accuracy_flexible_extract": acc,
        "accuracy_strict_match": acc_strict,
        "samples": n,
        "time_s": elapsed,
    }
    with open("/repo/gsm8k_diffusion_results.json", "w") as f:
        json.dump(result_json, f, indent=2)
    print("Results saved to /repo/gsm8k_diffusion_results.json")
else:
    print("ERROR: Evaluation returned None")
    exit(1)
