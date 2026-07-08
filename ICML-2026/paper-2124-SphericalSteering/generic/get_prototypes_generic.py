"""
Step 2 (Generic): Compute Contrastive Prototypes

Computes:  mu_T = normalize(mean(correct) - mean(incorrect)),  mu_H = -mu_T

Usage:
    python get_prototypes_generic.py --feature_file ./features_generic/llama3.1-8B_mmlu_global_train_l14.npz
"""

import argparse
import numpy as np
import os


def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def main():
    parser = argparse.ArgumentParser(description="Step 2: Compute contrastive prototypes")
    parser.add_argument('--feature_file', type=str, required=True, help="Path to feature .npz file")
    parser.add_argument('--save_dir', type=str, default='./prototypes_generic')
    args = parser.parse_args()

    data = np.load(args.feature_file)
    X, y = data['activations'], data['labels']
    print(f"Loaded {len(X)} samples ({sum(y)} correct, {len(y)-sum(y)} incorrect), dim={X.shape[1]}")

    diff = np.mean(X[y == 1], axis=0) - np.mean(X[y == 0], axis=0)
    mu_T = normalize(diff)
    mu_H = -mu_T

    os.makedirs(args.save_dir, exist_ok=True)
    base_name = os.path.basename(args.feature_file).replace('.npz', '')
    save_path = os.path.join(args.save_dir, f"{base_name}_proto.npz")
    np.savez(save_path, mu_T=mu_T, mu_H=mu_H)
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    main()
