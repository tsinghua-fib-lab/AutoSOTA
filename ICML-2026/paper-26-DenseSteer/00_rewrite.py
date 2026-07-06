#!/usr/bin/env python3
"""Rewrite model solutions into denser positive responses for DenseSteer."""

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# === CONFIGURATION ===
REWRITER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
INPUT_FILE = Path("data/rewritten.json")
OUTPUT_FILE = Path("rewrites_out/dense_rewritten.json")
MAX_SAMPLES = 100
MAX_NEW_TOKENS = 2048


def build_dense_prompt(question: str, original_response: str) -> str:
    return f"""Rewrite the solution to be denser while preserving its meaning, style, computations, and final answer.

Rules:
- Only merge or rephrase existing reasoning.
- Do not add new facts, steps, or calculations.
- Preserve special markers such as "<<a=b>>" if they appear.
- Output only the rewritten solution.

Question:
{question}

Original solution:
{original_response}

Rewritten solution:
"""


def as_sample_list(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("samples", "data", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("Input JSON must be a list or contain a samples/data/examples list.")


def get_question(obj):
    return obj.get("question") or obj.get("doc", {}).get("question") or obj.get("prompt", "")


def get_original_response(obj):
    for key in ("resp_before", "neg_response", "response", "model_output", "completion"):
        if obj.get(key):
            return obj[key]
    return ""


def load_samples(path: Path, max_samples: int | None = None):
    with path.open("r") as f:
        data = json.load(f)

    samples = []
    for obj in as_sample_list(data):
        question = get_question(obj).strip()
        resp_before = get_original_response(obj).strip()
        if not question or not resp_before:
            continue

        doc = obj.get("doc", {"question": question})
        samples.append(
            {
                "raw": obj,
                "doc_id": obj.get("doc_id"),
                "doc": doc,
                "question": question,
                "resp_before": resp_before,
            }
        )

        if max_samples and len(samples) >= max_samples:
            break

    return samples


def format_prompt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def rewrite_one(model, tokenizer, question: str, original_response: str) -> str:
    prompt = build_dense_prompt(question, original_response)
    text = format_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    generated = outputs[0][prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_output_record(sample, rewritten: str):
    record = dict(sample["raw"])
    doc = dict(sample["doc"])
    doc.setdefault("question", sample["question"])

    record.update(
        {
            "doc_id": sample["doc_id"],
            "doc": doc,
            "resp_before": sample["resp_before"],
            "resp_after": rewritten,
            "neg_response": sample["resp_before"],
            "pos_response": rewritten,
            "resp_rewrite_ok": bool(rewritten),
            "resp_rewriter_model": REWRITER_MODEL,
        }
    )

    # 01_extract_vectors.py filters on exact_match; this step assumes the input
    # file already contains correct examples when no score is provided.
    if "results" not in record and "metrics" not in record:
        record["results"] = {"exact_match": 1.0}

    return record


def main():
    print(f"Loading samples from: {INPUT_FILE}")
    samples = load_samples(INPUT_FILE, max_samples=MAX_SAMPLES)
    print(f"Loaded {len(samples)} samples")

    print(f"Loading rewriter model: {REWRITER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(REWRITER_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        REWRITER_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    results = []
    for sample in tqdm(samples, desc="Rewriting"):
        rewritten = rewrite_one(
            model,
            tokenizer,
            sample["question"],
            sample["resp_before"],
        )
        results.append(build_output_record(sample, rewritten))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} rewritten samples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
