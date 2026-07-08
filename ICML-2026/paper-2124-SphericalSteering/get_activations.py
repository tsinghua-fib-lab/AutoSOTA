"""
Step 1: Extract Hidden State Features from TruthfulQA

This script extracts last-token hidden state activations from a specified
transformer layer for all TruthfulQA question-answer pairs.

Usage:
    python get_activations.py llama3.1-8B --layer 14 --save_dir ./features

Output:
    Saves a .npz file containing:
    - activations: [N, hidden_dim] array of hidden states
    - labels: [N] binary labels (1=truthful, 0=hallucination)  
    - q_indices: [N] question indices for K-Fold splitting
"""

import argparse
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import get_truthfulqa_data, extract_layer_activations


HF_NAMES = {
    'llama3.1-8B': 'meta-llama/Llama-3.1-8B',
    'llama3.1-8B-Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'Qwen2.5-7B-Instruct': 'Qwen/Qwen2.5-7B-Instruct',
}


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Extract hidden state features from TruthfulQA"
    )
    parser.add_argument(
        'model_name', 
        type=str,
        help=f"Model name. Available: {list(HF_NAMES.keys())}"
    )
    parser.add_argument(
        '--layer', 
        type=int, 
        default=14,
        help="Layer index to extract hidden states from"
    )
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default='./features',
        help="Directory to save extracted features"
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help="Local directory with model data (overrides model_name)"
    )
    
    args = parser.parse_args()

    # Get model path
    MODEL = HF_NAMES.get(args.model_name, args.model_name) if not args.model_dir else args.model_dir
    
    # 1. Load Model and Tokenizer
    print(f"Loading model: {MODEL}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    
    # Set pad_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 2. Load TruthfulQA Data
    prompts, labels, q_indices = get_truthfulqa_data(tokenizer, style="standard")
    
    print("\n=== Data Statistics ===")
    print(f"Total samples: {len(prompts)}")
    print(f"Truthful samples: {sum(labels)}")
    print(f"Hallucination samples: {len(labels) - sum(labels)}")
    print(f"Example prompt: {prompts[0][:100]}...")
    
    # 3. Extract Features
    device = "cuda" if torch.cuda.is_available() else "cpu"
    activations = extract_layer_activations(model, tokenizer, prompts, args.layer, device)
    
    # 4. Save Features
    os.makedirs(args.save_dir, exist_ok=True)
    
    save_name = f"{args.model_name}_layer{args.layer}.npz"
    save_path = os.path.join(args.save_dir, save_name)
    
    print(f"\nSaving features to {save_path}...")
    np.savez(
        save_path, 
        activations=activations,
        labels=labels,
        q_indices=q_indices
    )
    
    print(f"Feature shape: {activations.shape}")
    print("Step 1 Complete!")


if __name__ == '__main__':
    main()
