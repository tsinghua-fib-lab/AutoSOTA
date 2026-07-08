"""
Step 1 (Generic): Extract Hidden State Features for MC Tasks

Usage:
    python get_activations_generic.py --model_name llama3.1-8B --dataset copa --layer 14
    python get_activations_generic.py --model_name llama3.1-8B --dataset mmlu_global --layer 14
    python get_activations_generic.py --model_name llama3.1-8B --dataset storycloze --layer 14
    python get_activations_generic.py --model_name llama3.1-8B --dataset winogrande --layer 14 --num_samples 2000
    python get_activations_generic.py --model_name llama3.1-8B --dataset boolq --layer 14

Output:
    Saves a .npz file containing:
    - activations: [N, hidden_dim] array of last-token hidden states
    - labels: [N] binary labels (1=correct, 0=incorrect)
    - q_indices: [N] question indices for K-Fold splitting
"""

import argparse
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils_generic import HF_NAMES, set_seed, get_dataset_data, get_layer_activations


def main():
    parser = argparse.ArgumentParser(description="Step 1: Extract features for MC tasks")
    parser.add_argument('--model_name', type=str, default='llama3.1-8B-Instruct',
                        help=f"Model name or path. Shortcuts: {list(HF_NAMES.keys())}")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['copa', 'storycloze', 'mmlu_global', 'winogrande', 'boolq'])
    parser.add_argument('--layer', type=int, default=14, help="Layer index to extract")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--num_samples', type=int, default=None, help="Limit number of questions")
    parser.add_argument('--save_dir', type=str, default='./features_generic')
    parser.add_argument('--model_dir', type=str, default=None,
                        help="Local model directory (overrides model_name)")
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    # 1. Load Model
    model_path = args.model_dir if args.model_dir else HF_NAMES.get(args.model_name, args.model_name)
    print(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Load Data
    prompts, labels, q_indices = get_dataset_data(
        args.dataset, split=args.split, num_samples=args.num_samples,
        seed=args.seed
    )

    print(f"\n=== Data Statistics ===")
    print(f"Total samples: {len(prompts)}")
    print(f"Correct (label=1): {sum(labels)}, Incorrect (label=0): {len(labels) - sum(labels)}")
    print(f"Example prompt: {prompts[0][:200]}...")

    # 3. Extract Features
    activations = get_layer_activations(model, tokenizer, prompts, args.layer, device)

    # 4. Save
    os.makedirs(args.save_dir, exist_ok=True)
    save_name = f"{args.model_name}_{args.dataset}_{args.split}_l{args.layer}.npz"
    save_path = os.path.join(args.save_dir, save_name)

    np.savez(save_path, activations=activations, labels=labels, q_indices=q_indices)
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    main()
