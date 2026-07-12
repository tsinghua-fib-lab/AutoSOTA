"""
Evaluation script for GVALID on HNL (HardNonLinear8D) dataset.
Reproduces the policy_suboptimality metric from Table 1 of the paper.

Usage: python3 eval.py [--n_seeds 20] [--gpu_id 0]

Paper settings:
  - Dataset: HardNonLinear8D (8-dim covariates, smooth unimodal dose-response)
  - Budget: N_total=350, N_init=35, batch_size=5
  - Noise: sigma=0.1, Test ratio: 0.3 (7:3 split)
  - Init: LHS, t_grid=101 points, GP with Matern 2.5 kernel
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, "/repo")

def main():
    parser = argparse.ArgumentParser(description="GVALID Evaluation on HNL")
    parser.add_argument("--n_seeds", type=int, default=20, help="Number of random seeds")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--n_total", type=int, default=350, help="Total query budget")
    parser.add_argument("--n_init", type=int, default=35, help="Initial labeled samples")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size per round")
    parser.add_argument("--n_pool", type=int, default=1750, help="Pool size for dataset")
    parser.add_argument("--sampler", type=str, default="GVALID", help="Sampler name")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    from src.experiment import run_single_seed

    print("GVALID Evaluation on HardNonLinear8D (HNL)")
    print("Settings: N_total=%d, N_init=%d, B=%d" % (args.n_total, args.n_init, args.batch_size))
    print("Sampler: %s, Seeds: %d, GPU: %d" % (args.sampler, args.n_seeds, args.gpu_id))
    print("=" * 60)

    all_final_values = []
    for seed in range(args.n_seeds):
        try:
            results = run_single_seed(
                dataset_name="HardNonLinear8D",
                sampler_name=args.sampler,
                seed=seed,
                N_total=args.n_total,
                B=args.batch_size,
                N_init=args.n_init,
                beta=1.96,
                num_threads=1,
                n_pool=args.n_pool,
                test_ratio=0.3,
                n_candidates=500,
                gpu_batch_size=8,
                t_grid_size=101,
                init_strategy="lhs",
                validate_theory=False,
                lr_x=0.1,
                lr_t=0.1,
            )
            final = results[-1]
            psub = final["policy_suboptimality"]
            all_final_values.append(psub)
            print("  Seed %2d: policy_suboptimality=%.6f" % (seed, psub))

        except Exception as e:
            print("  Seed %2d: FAILED - %s" % (seed, str(e)))
            import traceback
            traceback.print_exc()

    if all_final_values:
        arr = np.array(all_final_values)
        print()
        print("=" * 60)
        print("FINAL RESULT (%s on HNL, N=%d, n=%d seeds)" % (args.sampler, args.n_total, len(arr)))
        print("  policy_suboptimality: mean=%.6f, std=%.6f" % (arr.mean(), arr.std()))
        print("  95%% CI: [%.6f, %.6f]" % (
            arr.mean() - 1.96 * arr.std() / np.sqrt(len(arr)),
            arr.mean() + 1.96 * arr.std() / np.sqrt(len(arr))
        ))
        print("=" * 60)

if __name__ == "__main__":
    main()
