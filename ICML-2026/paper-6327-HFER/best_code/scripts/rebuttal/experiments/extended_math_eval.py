"""
Extended MATH Evaluation
=========================
Addresses: Reviewer QygV ("datasets are quite small and from a narrow domain")

Evaluates spectral method on 500 MATH problems across 7 categories and 5 difficulty levels
using Llama-3.2-1B with chain-of-thought generation. Requires GPU.

Usage:
    python scripts/rebuttal/extended_math_eval.py --model meta-llama/Llama-3.2-1B-Instruct --load-in-4bit
"""

import os, sys, json, re, argparse
import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'figure.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

def extract_answer(text):
    """Extract the final boxed answer from MATH-style output."""
    # Look for \boxed{...}
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if matches: return matches[-1].strip()
    # Look for "The answer is ..."
    match = re.search(r'answer\s+is\s+[:\s]*([^\.\n]+)', text, re.IGNORECASE)
    if match: return match.group(1).strip()
    # Last number
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers: return numbers[-1]
    return ''

def run_math_eval(args):
    print("=" * 70)
    print("  EXTENDED MATH EVALUATION")
    print("=" * 70)

    os.makedirs('output/rebuttal', exist_ok=True)
    os.makedirs('data/results/rebuttal', exist_ok=True)

    # Load MATH dataset
    try:
        from datasets import load_dataset
        dataset = load_dataset('hendrycks/competition_math', split='test', trust_remote_code=True)
        print(f"Loaded MATH dataset: {len(dataset)} problems")
    except Exception as e:
        print(f"ERROR: Could not load MATH dataset: {e}")
        print("Install: pip install datasets")
        sys.exit(1)

    # Stratified sampling: ~70 per category
    from collections import defaultdict
    by_type = defaultdict(list)
    for i, item in enumerate(dataset):
        by_type[item['type']].append(i)

    selected = []
    per_category = args.n_problems // len(by_type)
    np.random.seed(42)
    for cat, indices in sorted(by_type.items()):
        n = min(per_category, len(indices))
        chosen = np.random.choice(indices, size=n, replace=False)
        selected.extend(chosen)
    selected = selected[:args.n_problems]
    print(f"Selected {len(selected)} problems across {len(by_type)} categories")

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from spectral_trust import GSPDiagnosticsFramework, GSPConfig

    model_kwargs = {"output_attentions": True, "output_hidden_states": True}
    if args.load_in_4bit: model_kwargs["load_in_4bit"] = True

    config = GSPConfig(model_name=args.model, device=args.device, model_kwargs=model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    results = []

    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model(args.model)
        model = framework.instrumenter.model
        num_layers = model.config.num_hidden_layers
        target_layer = int(0.75 * num_layers)

        for idx in tqdm(selected, desc='MATH problems'):
            item = dataset[int(idx)]
            problem = item['problem']
            gt_answer = extract_answer(item['solution'])
            category = item['type']
            level = item['level']

            # Generate CoT
            prompt = f"Solve the following problem step by step.\n\nProblem: {problem}\n\nSolution:"
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            try:
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id)
                response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                pred_answer = extract_answer(response)
                is_correct = pred_answer.strip() == gt_answer.strip()

                # Spectral analysis on the full prompt + response
                full_text = prompt + response
                analysis = framework.analyze_text(full_text, save_results=False)
                layers = analysis.get('layer_diagnostics', [])
                hfer = float(getattr(layers[target_layer], 'hfer')) if target_layer < len(layers) else None

                results.append({
                    'problem_idx': int(idx), 'category': category, 'level': level,
                    'is_correct': is_correct, 'hfer': hfer, 'pred': pred_answer, 'gt': gt_answer
                })
            except Exception as e:
                print(f"  ERROR on problem {idx}: {e}")
                continue

    # Analysis by category and difficulty
    print(f"\nTotal evaluated: {len(results)}")
    overall_correct = sum(r['is_correct'] for r in results)
    overall_acc = overall_correct / len(results) if results else 0
    print(f"Overall accuracy: {overall_acc*100:.1f}%")

    # By category
    print(f"\n{'Category':<25} | {'n':>4} | {'Acc':>6} | {'d':>6}")
    print("-" * 50)
    for cat in sorted(set(r['category'] for r in results)):
        cat_results = [r for r in results if r['category'] == cat]
        correct_h = [r['hfer'] for r in cat_results if r['is_correct'] and r['hfer'] is not None]
        wrong_h = [r['hfer'] for r in cat_results if not r['is_correct'] and r['hfer'] is not None]
        d = 0
        if len(correct_h) >= 2 and len(wrong_h) >= 2:
            d = (np.mean(correct_h) - np.mean(wrong_h)) / np.sqrt(
                ((len(correct_h)-1)*np.std(correct_h, ddof=1)**2 +
                 (len(wrong_h)-1)*np.std(wrong_h, ddof=1)**2) /
                (len(correct_h) + len(wrong_h) - 2))
        cat_acc = sum(r['is_correct'] for r in cat_results) / len(cat_results)
        print(f"{cat:<25} | {len(cat_results):>4} | {cat_acc*100:>5.1f}% | {d:>5.2f}")

    # Save
    with open('data/results/rebuttal/extended_math.json', 'w') as f:
        json.dump({'results': results, 'overall_accuracy': overall_acc,
                   'n_problems': len(results)}, f, indent=2)
    print(f"\nSaved: data/results/rebuttal/extended_math.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--n-problems', type=int, default=500)
    run_math_eval(parser.parse_args())
