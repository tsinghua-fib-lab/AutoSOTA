import argparse
import json
import os
import random

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

MAX_JUDGE_TOKENS = 4096


def load_judge_model(model_id: str = "google/gemma-3-4b-it"):
    print(f"Loading judge model: {model_id}")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def get_judge_verdict(model, tokenizer, prompt: str, response_a: str, response_b: str):
    judge_prompt = f"""
You are an impartial judge evaluating the quality of two AI-generated responses to a given prompt.
Your task is to determine which response is better based on helpfulness, accuracy, and coherence.

Prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Reply with 'A' if Response A is better, 'B' if Response B is better, or 'Tie' if they are of equal quality.
Do not provide any explanation, just the single word verdict.
"""

    inputs = tokenizer(
        judge_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_JUDGE_TOKENS,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        generation = generation[0][input_len:]

    decoded = tokenizer.decode(generation, skip_special_tokens=True).strip()
    if "A" in decoded and "B" not in decoded:
        return "A"
    if "B" in decoded and "A" not in decoded:
        return "B"
    if "Tie" in decoded or "tie" in decoded:
        return "Tie"
    if decoded.startswith("A"):
        return "A"
    if decoded.startswith("B"):
        return "B"
    return "Tie"


def update_elo(elo_a: float, elo_b: float, score_a: float, k_factor: int = 32):
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    return elo_a + k_factor * (score_a - expected_a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model_id", default="google/gemma-3-4b-it")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)

    models_data = {}
    for input_file in args.input_files:
        print(f"Loading results from {input_file}...")
        with open(input_file, "r") as f:
            data = json.load(f)
            models_data.update(data["models"])

    model_names = list(models_data.keys())
    if not model_names:
        raise SystemExit("No models found in input files.")

    judge_model, judge_tokenizer = load_judge_model(args.model_id)
    elo_scores = {name: 1000.0 for name in model_names}
    comparisons = []
    num_samples = len(models_data[model_names[0]]["samples"])
    print(f"Evaluating {len(model_names)} models on {num_samples} samples...")

    for i in tqdm(range(num_samples)):
        prompt = models_data[model_names[0]]["samples"][i]["prompt"]
        for j in range(len(model_names)):
            for k in range(j + 1, len(model_names)):
                model_a = model_names[j]
                model_b = model_names[k]
                resp_a = models_data[model_a]["samples"][i]["generated"]
                resp_b = models_data[model_b]["samples"][i]["generated"]

                if random.random() > 0.5:
                    verdict = get_judge_verdict(judge_model, judge_tokenizer, prompt, resp_a, resp_b)
                    winner = model_a if verdict == "A" else model_b if verdict == "B" else "Tie"
                else:
                    verdict = get_judge_verdict(judge_model, judge_tokenizer, prompt, resp_b, resp_a)
                    winner = model_b if verdict == "A" else model_a if verdict == "B" else "Tie"

                comparisons.append(
                    {"prompt": prompt, "model_a": model_a, "model_b": model_b, "winner": winner}
                )

                score_a = 1.0 if winner == model_a else (0.5 if winner == "Tie" else 0.0)
                score_b = 1.0 if winner == model_b else (0.5 if winner == "Tie" else 0.0)
                new_elo_a = update_elo(elo_scores[model_a], elo_scores[model_b], score_a)
                new_elo_b = update_elo(elo_scores[model_b], elo_scores[model_a], score_b)
                elo_scores[model_a] = new_elo_a
                elo_scores[model_b] = new_elo_b

    with open(args.output_file, "w") as f:
        json.dump({"elo_scores": elo_scores, "comparisons": comparisons}, f, indent=2)
    print(f"Judge results saved to {args.output_file}")


if __name__ == "__main__":
    main()
