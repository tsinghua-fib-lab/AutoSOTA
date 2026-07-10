"""
Find steering circuit using gradient-based attribution with WEIGHTED SAMPLING.
OPTIMIZED: Computes attribution ONCE, then samples multiple trials.
"""
import argparse
import json
import torch
import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
vocab_path = os.path.join(script_dir, '../function_circuit')
circuit_utils_path = os.path.join(script_dir, '../circuit_utils')
steering_path = script_dir

sys.path.append(vocab_path)
from function_utils import CNNProbe

sys.path.append(steering_path)
from steering_utils import score_seqs, rank_nodes_by_attribution

sys.path.append(circuit_utils_path)
from clt_circuit import CircuitDiscovererCLT
from plt_circuit import CircuitDiscovererPLT


def main():
    parser = argparse.ArgumentParser(description='Find steering circuit using attribution sampling')
    parser.add_argument('--supp', type=int, required=True,
                        help='|Support| of steering circuit (number of nodes to sample)')
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to JSON file containing circuit data')
    parser.add_argument('--esm_weights', type=str, required=True,
                        help='Path to ESM weights')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['clt', 'plt', 'clt_direct', 'clt_local'],
                        help='Model type')
    parser.add_argument('--scoring_model', type=str, required=True,
                        help='Path to scoring model state_dict (.pt file)')
    parser.add_argument('--freeze_attention', action='store_true',
                        help='Freeze attention layers during attribution')
    parser.add_argument('--wt', type=str, required=True,
                        help='Wildtype sequence')
    
    # NEW ARGUMENTS FOR BATCHING
    parser.add_argument('--output_prefix', type=str, required=True,
                        help='Prefix for output files (e.g., path/to/probe_fold0_steering)')
    parser.add_argument('--trials', type=int, default=1, 
                        help='Number of independent trials to sample')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Base Random seed')

    args = parser.parse_args()

    # Set base seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.freeze_attention and args.model_type == "clt_direct":
        raise ValueError("Freeze attention not needed for clt_direct")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Base Seed: {args.seed} | Trials: {args.trials}")

    # 1. Load Circuit & Model (ONCE)
    with open(args.json_path, 'r') as f:
        data = json.load(f)
    nodes = {int(k): v for k, v in data['nodes'].items()}
    k = data['k']
    
    # Flatten available nodes
    available_pool = []
    for layer, latents in nodes.items():
        for latent in latents:
            available_pool.append((layer, latent))
    
    total_nodes = len(available_pool)
    if args.supp > total_nodes:
        print(f"WARNING: Support ({args.supp}) > Available nodes ({total_nodes}). Using all nodes.")
        args.supp = total_nodes

    if args.model_type == 'clt' or args.model_type == 'clt_direct':
        discoverer = CircuitDiscovererCLT(device, ckpt_path=args.ckpt, esm_weights_path=args.esm_weights)
        sequential = (args.model_type == 'clt')
    elif args.model_type == 'plt':
        discoverer = CircuitDiscovererPLT(device, ckpt_path=args.ckpt, esm_weights_path=args.esm_weights)
        sequential = True

    embed_dim = discoverer.esm.embed_dim
    scoring_model = CNNProbe(input_dim=embed_dim).to(device)
    scoring_model.load_state_dict(torch.load(args.scoring_model, map_location=device))
    scoring_model.eval()

    # 2. Compute Attribution Scores (ONCE)
    print(f"\nComputing attribution scores (Generic)...")
    ranked_nodes = rank_nodes_by_attribution(
        discoverer, scoring_model, args.wt, nodes,
        sequential=sequential,
        freeze_attention=args.freeze_attention
    )

    # 3. Prepare Probabilities (ONCE)
    scores = np.array([item[2] for item in ranked_nodes], dtype=np.float64)
    scores = np.maximum(scores, 0.0) 
    sum_scores = np.sum(scores)
    
    if sum_scores > 1e-9:
        probs = scores / sum_scores
    else:
        print("Warning: Sum of scores is effectively 0. Using uniform distribution.")
        probs = np.ones_like(scores) / len(scores)

    # 4. Loop to Generate Trials
    print(f"\nStarting generation of {args.trials} trials...")
    
    for t in range(args.trials):
        # Deterministic seed per trial based on base seed
        trial_seed = args.seed + t 
        np.random.seed(trial_seed)
        
        # Sample Indices
        selected_indices = np.random.choice(
            len(ranked_nodes), 
            size=args.supp, 
            replace=False, 
            p=probs
        )

        selected_items = [ranked_nodes[i] for i in selected_indices]
        selected_items.sort(key=lambda x: (x[0], x[1])) # Sort for cleanliness

        # Build dict & string
        steering_circuit = {}
        node_str_list = []
        
        for layer, latent, score in selected_items:
            if layer not in steering_circuit:
                steering_circuit[layer] = []
            steering_circuit[layer].append(latent)
            node_str_list.append(f"{layer}:{latent}")

        layer_latent_str = ", ".join(node_str_list)

        output_data = {
            'nodes': {str(k): v for k, v in steering_circuit.items()},
            'k': len(selected_items),
            'original_k': k,
            'model_type': args.model_type,
            'method': 'weighted_attribution_sampling',
            'base_seed': args.seed,
            'trial_seed': trial_seed,
            'trial_index': t,
            'layer_latent_string': layer_latent_str
        }
        
        # Save to specific trial file
        # Format: {prefix}_trial{t}.json
        output_path = f"{args.output_prefix}_trial{t}.json"
            
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
    print(f"Successfully saved {args.trials} trial files starting at {args.output_prefix}_trial0.json")

if __name__ == '__main__':
    main()