"""Evaluation script for TOP1-PG (CA-PG-SwR) on KuaiRec K=50.

Reproduces the Policy Value metric from Table 4 of:
"Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking"
"""
import sys, os, json, argparse
import torch
import numpy as np

sys.path.insert(0, "/repo")
os.chdir("/repo/experiments/synthetic")

from experiments.synthetic.function_kuairec import (
    setup_data_generation_process,
    initialize_trainable_policy,
    train_online_pg_policy,
)

def run_single_seed(seed, n_epoch, K, device_str, dim_emb=10, n_moe=1, lr=0.01):
    device = torch.device(device_str)

    env = setup_data_generation_process(
        dataset_path="/repo/experiments/synthetic/data/kuairec_small_matrix.csv",
        n_output_action=1,
        device=device,
        random_seed=12345,
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    policy, _ = initialize_trainable_policy(
        env=env, dim_model_emb=dim_emb, n_moe_model=n_moe,
        device=device, random_seed=seed,
    )

    _, logs = train_online_pg_policy(
        env=env, early_stage_policy=policy, early_stage_lr=lr,
        late_stage_optimality="optimal", credit_assignment_type="TOP1",
        is_vanilla_replacement=False, n_epoch=n_epoch, n_epochs_per_log=100,
        n_candidate_action_train=K, n_candidate_action_eval=K,
        device=device, random_seed=seed, use_wandb=False,
    )

    final_val = logs["policy_values"][-1].item()
    return final_val, logs["policy_values"].cpu().tolist()

def main():
    parser = argparse.ArgumentParser(description="Evaluate TOP1-PG (CA-PG-SwR) on KuaiRec")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds")
    parser.add_argument("--start-seed", type=int, default=0, help="Starting seed")
    parser.add_argument("--n-epoch", type=int, default=5000, help="Training epochs (5000=@50K, 50000=@500K)")
    parser.add_argument("--K", type=int, default=50, help="Candidate set size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--dim-emb", type=int, default=10, help="Embedding dimension")
    parser.add_argument("--n-moe", type=int, default=1, help="Number of MoE experts")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()
    
    results = {}
    all_seed_vals = []
    
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        val, history = run_single_seed(seed, args.n_epoch, args.K, args.device,
                                        dim_emb=args.dim_emb, n_moe=args.n_moe, lr=args.lr)
        all_seed_vals.append(val)
        results[f"seed_{seed}"] = {"policy_value": val, "history": history}
        print(f"Seed {seed}: policy_value = {val:.4f}")
        # Intermediate save to protect against mid-run crashes.
        if args.output:
            partial = {
                "method": "TOP1-PG (CA-PG-SwR)",
                "benchmark": "KuaiRec",
                "K": args.K,
                "n_epoch": args.n_epoch,
                "gradient_steps": args.n_epoch * 10,
                "n_seeds": len(all_seed_vals),
                "policy_value_mean": float(np.mean(all_seed_vals)),
                "policy_value_std": float(np.std(all_seed_vals)) if len(all_seed_vals) > 1 else 0.0,
                "per_seed": results,
            }
            with open(args.output, "w") as f:
                json.dump(partial, f, indent=2)
    
    mean_val = float(np.mean(all_seed_vals))
    std_val = float(np.std(all_seed_vals))
    
    output = {
        "method": "TOP1-PG (CA-PG-SwR)",
        "benchmark": "KuaiRec",
        "K": args.K,
        "n_epoch": args.n_epoch,
        "gradient_steps": args.n_epoch * 10,
        "n_seeds": args.seeds,
        "policy_value_mean": mean_val,
        "policy_value_std": std_val,
        "per_seed": results,
    }
    
    print(f"\n=== FINAL RESULT ===")
    print(f"Method: TOP1-PG (CA-PG-SwR), K={args.K}, Steps={args.n_epoch * 10}")
    print(f"Policy Value: {mean_val:.4f} +/- {std_val:.4f}")
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {args.output}")
    
    return output

if __name__ == "__main__":
    try:
        main()
    finally:
        # Skip PyTorch atexit CUDA cleanup to avoid SIGSEGV.
        # PyTorch 2.1 + CUDA 12.1 crashes during interpreter teardown
        # when CUDA resources are freed in undefined order.
        # os._exit(0) bypasses atexit handlers; the OS reclaims GPU memory.
        os._exit(0)
