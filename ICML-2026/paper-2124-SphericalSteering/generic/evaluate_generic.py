"""
Evaluate MC tasks with Spherical Steering.

Evaluation logic (MC1 style):
  1. For each question, construct full prompts for ALL options.
  2. Calculate log_prob for the option text part only.
  3. Apply length normalization: score = mean(log_probs).
  4. Select option with max score → compare with ground truth.

Supported datasets: COPA, StoryCloze, MMLU, Winogrande, BoolQ

Usage:
    python evaluate_generic.py --model_name llama3.1-8B --dataset copa --layer 14 \
        --prototype_path ./prototypes_generic/xxx.npz --alpha 0.3 --beta -0.2
    python evaluate_generic.py --model_name llama3.1-8B --dataset mmlu_global --layer 14 \
        --prototype_path ./prototypes_generic/xxx.npz --alpha 0.3 --beta -0.2
"""

import argparse
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from functools import partial
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import TraceDict

# spherical_steering.py lives in parent directory (ICML2026/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spherical_steering import get_spherical_intervention
from utils_generic import HF_NAMES, MMLU_CATEGORIES, set_seed


# ============================================================
# Prompt Formatters
# ============================================================

def format_copa_prompt(premise, question_type, choice1, choice2):
    return (
        f"Question:\n{premise} Based on the previous passage, "
        f"choose the most reasonable {question_type}.\n"
        f"A: {choice1}\nB: {choice2}\n\nAnswer:"
    )


def format_storycloze_prompt(sentences, opt1, opt2):
    story = " ".join(sentences) if isinstance(sentences, list) else sentences
    return (
        f"{story}\n\nQuestion: Which ending makes more sense?\n"
        f"A. {opt1}\nB. {opt2}\nAnswer:"
    )


def format_mmlu_prompt(tokenizer, question, choices):
    """Format MMLU prompt using chat template (for Instruct models)."""
    user_content = (
        "You are solving a multiple-choice question.\n\n"
        "Choose the correct option AND output the option text EXACTLY as written.\n"
        "Rules:\n"
        "- Output must be a verbatim copy of ONE option's text.\n"
        "- Do NOT output the option letter (A/B/C/D).\n"
        "- Do NOT add explanations, punctuation, or extra words.\n"
        "- Do NOT paraphrase. Copy the option text exactly.\n\n"
        f"Question: {question}\n"
        "Options:\n"
        f"A) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n\n"
        "Final answer (verbatim option text only):"
    )
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_winogrande_prompt(sentence, option1, option2):
    return (
        f"Q: {sentence}\n"
        f"Which option correctly fills the blank?\n"
        f"1) {option1}\n2) {option2}\nA:"
    )


def format_boolq_prompt(passage, question):
    return f"Passage: {passage}\nQuestion: {question}\nA:"


# ============================================================
# Score Calculation
# ============================================================

def calculate_option_score(model, tokenizer, base_prompt, option_text,
                           hook_fn, layer_name, device, normalize_length=True):
    """
    Calculate log probability score for an option.
    Scores only the option tokens (after the prompt).
    """
    prompt_ids = tokenizer(base_prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    intervention_start_idx = prompt_len - 1

    # Append option text (with leading space for raw prompts, as-is for chat)
    if base_prompt.strip().endswith(":"):
        text_to_append = f" {option_text}"
    else:
        text_to_append = option_text

    option_ids = tokenizer(text_to_append, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    option_len = option_ids.shape[1]
    input_ids = torch.cat([prompt_ids, option_ids], dim=1)

    # Hook setup
    current_hook = partial(hook_fn, start_idx=intervention_start_idx) if hook_fn is not None else None

    with torch.no_grad():
        with TraceDict(model, [layer_name], edit_output=current_hook):
            outputs = model(input_ids)

    logits = outputs.logits
    log_probs_all = torch.nn.functional.log_softmax(logits[..., :-1, :], dim=-1)
    shift_labels = input_ids[..., 1:]
    token_log_probs = torch.gather(log_probs_all, 2, shift_labels.unsqueeze(2)).squeeze(2)

    option_log_probs = token_log_probs[0, intervention_start_idx:intervention_start_idx + option_len]

    if normalize_length:
        return option_log_probs.mean().item()
    else:
        return option_log_probs.sum().item()


# ============================================================
# Per-dataset Evaluation Functions
# ============================================================

def evaluate_copa(model, tokenizer, hook_fn, layer_name, device, steering_stats):
    """Evaluate on COPA validation set (100 samples)."""
    print("Loading COPA validation set...")
    dataset = load_dataset("super_glue", "copa", split="validation")
    print(f"Evaluating on {len(dataset)} samples...")

    correct, total = 0, 0
    pbar = tqdm(dataset)

    for item in pbar:
        premise = item["premise"]
        q_type = item["question"]
        choice1 = item["choice1"]
        choice2 = item["choice2"]
        correct_idx = item["label"]  # 0 or 1
        choices = [choice1, choice2]

        base_prompt = format_copa_prompt(premise, q_type, choice1, choice2)
        scores = [calculate_option_score(model, tokenizer, base_prompt, c,
                                         hook_fn, layer_name, device) for c in choices]

        if np.argmax(scores) == correct_idx:
            correct += 1
        total += 1
        pbar.set_description(f"Acc: {correct/total:.4f}")

    return correct / total


def evaluate_storycloze(model, tokenizer, hook_fn, layer_name, device, steering_stats):
    """Evaluate on XStoryCloze eval set."""
    print("Loading XStoryCloze (en) eval set...")
    dataset = load_dataset("juletxara/xstory_cloze", "en")['eval']
    print(f"Evaluating on {len(dataset)} samples...")

    correct, total = 0, 0
    pbar = tqdm(dataset)

    for item in pbar:
        story = [item[f'input_sentence_{i}'] for i in range(1, 5)]
        opt1 = item['sentence_quiz1']
        opt2 = item['sentence_quiz2']
        correct_idx = item['answer_right_ending'] - 1  # 1→0, 2→1

        base_prompt = format_storycloze_prompt(story, opt1, opt2)
        scores = [calculate_option_score(model, tokenizer, base_prompt, opt,
                                         hook_fn, layer_name, device) for opt in [opt1, opt2]]

        if np.argmax(scores) == correct_idx:
            correct += 1
        total += 1
        pbar.set_description(f"Acc: {correct/total:.4f}")

    return correct / total


def _evaluate_mmlu_items(model, tokenizer, hook_fn, layer_name, device, items, steering_stats):
    """Shared evaluation loop for MMLU items."""
    correct, total = 0, 0
    pbar = tqdm(items)

    for item in pbar:
        question = item['question']
        choices = item['choices']
        correct_idx = item['answer']

        base_prompt = format_mmlu_prompt(tokenizer, question, choices)
        scores = [calculate_option_score(model, tokenizer, base_prompt, c,
                                         hook_fn, layer_name, device) for c in choices]

        if np.argmax(scores) == correct_idx:
            correct += 1
        total += 1
        pbar.set_description(f"Acc: {correct/total:.4f}")

    return correct / total


def evaluate_mmlu_global(model, tokenizer, hook_fn, layer_name, device,
                         steering_stats, eval_split='test'):
    """
    Evaluate on MMLU Global Balanced (all 4 categories, 500/cat test).
    Returns micro-average accuracy across ~2000 samples.
    """
    print(f"Loading MMLU Global Balanced (split={eval_split})...")

    all_items = []
    for cat_name, subsets in MMLU_CATEGORIES.items():
        loaded = []
        for sub in subsets:
            try:
                loaded.append(load_dataset("cais/mmlu", sub, split='test'))
            except:
                pass
        if not loaded:
            continue

        cat_dataset = concatenate_datasets(loaded).shuffle(seed=42)
        total = len(cat_dataset)

        if eval_split == 'test':
            start_idx, end_idx = 500, min(1000, total)
        else:
            raise ValueError(f"Use eval_split='test' for mmlu_global evaluation")

        if start_idx >= total:
            continue

        dataset = cat_dataset.select(range(start_idx, end_idx))
        print(f"  {cat_name}: {len(dataset)} questions")
        all_items.extend(list(dataset))

    print(f"Total: {len(all_items)} questions")
    return _evaluate_mmlu_items(model, tokenizer, hook_fn, layer_name, device,
                                all_items, steering_stats)


def evaluate_winogrande(model, tokenizer, hook_fn, layer_name, device, steering_stats):
    """Evaluate on Winogrande validation set."""
    print("Loading Winogrande validation set...")
    dataset = load_dataset("winogrande", "winogrande_xl")['validation']
    print(f"Evaluating on {len(dataset)} samples...")

    correct, total = 0, 0
    pbar = tqdm(dataset)

    for item in pbar:
        sentence = item['sentence']
        opt1 = item['option1']
        opt2 = item['option2']
        correct_idx = int(item['answer']) - 1  # '1'→0, '2'→1

        base_prompt = format_winogrande_prompt(sentence, opt1, opt2)
        scores = [calculate_option_score(model, tokenizer, base_prompt, opt,
                                         hook_fn, layer_name, device) for opt in [opt1, opt2]]

        if np.argmax(scores) == correct_idx:
            correct += 1
        total += 1
        pbar.set_description(f"Acc: {correct/total:.4f}")

    return correct / total


def evaluate_boolq(model, tokenizer, hook_fn, layer_name, device, steering_stats):
    """Evaluate on BoolQ validation set. Options: ["no", "yes"]."""
    print("Loading BoolQ validation set...")
    dataset = load_dataset("super_glue", "boolq", split="validation")
    print(f"Evaluating on {len(dataset)} samples...")

    options = ["no", "yes"]
    correct, total = 0, 0
    pbar = tqdm(dataset)

    for item in pbar:
        passage = item["passage"]
        question = item["question"]
        correct_idx = int(item["label"])  # 0→no, 1→yes

        base_prompt = format_boolq_prompt(passage, question)
        scores = [calculate_option_score(model, tokenizer, base_prompt, opt,
                                         hook_fn, layer_name, device) for opt in options]

        if np.argmax(scores) == correct_idx:
            correct += 1
        total += 1
        pbar.set_description(f"Acc: {correct/total:.4f}")

    return correct / total


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate MC tasks with Spherical Steering")
    parser.add_argument('--model_name', type=str, default='llama3.1-8B-Instruct',
                        help=f"Model name or path. Shortcuts: {list(HF_NAMES.keys())}")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['copa', 'storycloze', 'mmlu_global', 'winogrande', 'boolq'])
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--prototype_path', type=str, required=True)
    parser.add_argument('--kappa', type=float, default=20.0)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--beta', type=float, default=-0.2)
    parser.add_argument('--disable_steering', action='store_true')
    parser.add_argument('--model_dir', type=str, default=None,
                        help="Local model directory (overrides model_name)")
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    # Load model
    model_path = args.model_dir if args.model_dir else HF_NAMES.get(args.model_name, args.model_name)
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load prototypes
    print(f"Loading prototypes: {args.prototype_path}")
    data = np.load(args.prototype_path)
    mu_T = torch.tensor(data['mu_T'], dtype=torch.float32, device=device)
    mu_H = torch.tensor(data['mu_H'], dtype=torch.float32, device=device)

    # Setup steering hook
    layer_name = f"model.layers.{args.layer}"
    steering_stats = {'total': 0, 'steered': 0}

    if args.disable_steering:
        print(">> Mode: BASELINE (No Steering)")
        hook_fn = None
    else:
        print(f">> Mode: STEERING (kappa={args.kappa}, alpha={args.alpha}, beta={args.beta})")
        hook_fn = get_spherical_intervention(
            mu_T, mu_H, args.kappa, args.alpha, args.beta, stats=steering_stats)

    # Dispatch evaluation
    if args.dataset == 'copa':
        accuracy = evaluate_copa(model, tokenizer, hook_fn, layer_name, device, steering_stats)
    elif args.dataset == 'storycloze':
        accuracy = evaluate_storycloze(model, tokenizer, hook_fn, layer_name, device, steering_stats)
    elif args.dataset == 'mmlu_global':
        accuracy = evaluate_mmlu_global(model, tokenizer, hook_fn, layer_name, device, steering_stats)
    elif args.dataset == 'winogrande':
        accuracy = evaluate_winogrande(model, tokenizer, hook_fn, layer_name, device, steering_stats)
    elif args.dataset == 'boolq':
        accuracy = evaluate_boolq(model, tokenizer, hook_fn, layer_name, device, steering_stats)

    # Print results
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Dataset:    {args.dataset}")
    print(f"Model:      {args.model_name}")
    print(f"Layer:      {args.layer}")
    print(f"Steering:   {'OFF' if args.disable_steering else 'ON'}")
    if not args.disable_steering:
        print(f"Parameters: kappa={args.kappa}, alpha={args.alpha}, beta={args.beta}")
        if steering_stats['total'] > 0:
            steer_pct = steering_stats['steered'] / steering_stats['total'] * 100
            print(f"Stats:      Steered={steering_stats['steered']}/{steering_stats['total']} ({steer_pct:.1f}%)")
    print("-" * 50)
    print(f"Accuracy:   {accuracy:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
