"""
Batch reproduction runner for GVALID paper (HNL dataset).
Runs GVALID and ABC3 with 20 seeds on HardNonLinear8D at N=350.
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "/repo")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HardNonLinear8D")
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--samplers", type=str, default="GVALID,ABC3")
    parser.add_argument("--n_total", type=int, default=350)
    parser.add_argument("--n_init", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--n_pool", type=int, default=1750)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--t_grid_size", type=int, default=101)
    parser.add_argument("--n_candidates", type=int, default=500)
    parser.add_argument("--gpu_batch_size", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.96)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--seed_offset", type=int, default=0)
    args = parser.parse_args()

    sampler_names = [s.strip() for s in args.samplers.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    import torch
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    from src.experiment import run_single_seed

    output_dir = f"results/reproduction_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    all_results = []
    sep = "=" * 60

    for sampler_name in sampler_names:
        print(f"\n{sep}")
        print(f"Running {sampler_name} on {args.dataset} with {args.n_seeds} seeds")
        print(f"{sep}")

        for seed in tqdm(range(args.seed_offset, args.seed_offset + args.n_seeds), desc=sampler_name):
            try:
                results = run_single_seed(
                    dataset_name=args.dataset,
                    sampler_name=sampler_name,
                    seed=seed,
                    N_total=args.n_total,
                    B=args.batch_size,
                    N_init=args.n_init,
                    beta=args.beta,
                    num_threads=args.num_threads,
                    n_pool=args.n_pool,
                    test_ratio=args.test_ratio,
                    n_candidates=args.n_candidates,
                    gpu_batch_size=args.gpu_batch_size,
                    t_grid_size=args.t_grid_size,
                    init_strategy="lhs",
                    validate_theory=False,
                    lr_x=0.1,
                    lr_t=0.1,
                )

                for r in results:
                    r["seed"] = seed
                    r["sampler"] = sampler_name
                    r["dataset"] = args.dataset
                all_results.extend(results)

                # Save incremental CSV
                df = pd.DataFrame(all_results)
                final_n = df[df["N"] == args.n_total]
                summary = final_n.groupby("sampler").agg(
                    mean_subopt=("policy_suboptimality", "mean"),
                    std_subopt=("policy_suboptimality", "std"),
                    count=("policy_suboptimality", "count"),
                )
                summary.to_csv(os.path.join(output_dir, f"summary_{timestamp}.csv"))

            except Exception as e:
                print(f"Seed {seed} for {sampler_name} failed: {e}")
                import traceback
                traceback.print_exc()

    # Final summary
    df = pd.DataFrame(all_results)
    final_n = df[df["N"] == args.n_total]
    print(f"\n{sep}")
    print(f"FINAL RESULTS at N={args.n_total}")
    print(f"{sep}")
    for sname in sampler_names:
        sub = final_n[final_n["sampler"] == sname]["policy_suboptimality"]
        if len(sub) > 0:
            print(f"{sname}: mean={sub.mean():.4f}, std={sub.std():.4f}, n={len(sub)}")
        else:
            print(f"{sname}: NO RESULTS")

    pkl_path = os.path.join(output_dir, f"full_results_{timestamp}.pkl")
    df.to_pickle(pkl_path)
    print(f"\nFull results saved to {pkl_path}")

if __name__ == "__main__":
    main()
