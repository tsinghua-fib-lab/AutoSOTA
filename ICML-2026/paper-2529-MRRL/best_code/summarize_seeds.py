import argparse
import os
from functools import reduce

import pandas as pd
from evaluate import find_and_load_best_context, run_evaluation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, required=True)
    parser.add_argument("--exp_grp", type=str, default="default_group")
    parser.add_argument("--data_seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--ckpt_strategy",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Whether to load the best-*.ckpt or last.ckpt for evaluation",
    )
    parser.add_argument("--use_test_data", action="store_true")
    parser.add_argument("--n_test", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=3000)
    parser.add_argument(
        "--compute_extras", action="store_true", help="Compute GCM and reconstruction error"
    )
    parser.add_argument("--metric_key", type=str, default="val_tot_loss")
    parser.add_argument("--selection_mode", type=str, default="min")
    parser.add_argument("--last_k", type=int, default=20)
    parser.add_argument("--exclude_sim_ids", type=int, nargs="*", default=[])
    parser.add_argument(
        "--method",
        type=str,
        default="tsls",
        choices=["tsls", "liml"],
        help="IV estimation method for compute_estimates (default: tsls)",
    )

    args = parser.parse_args()

    all_estimates = []
    all_extras = []

    # We don't need to instantiate ModelSelector manually here anymore
    # find_and_load_best_context handles it internally per seed

    for seed in args.data_seeds:
        exp_prefix = f"{args.exp_id}-ds{seed}"
        print(f"\n>>> Processing Seed: {seed} | Prefix: {exp_prefix}")

        try:
            # 1. Unified Discovery & Loading
            # This handles Selector -> Config -> Model Loading
            model, cfg, best_sim = find_and_load_best_context(args, exp_prefix)

            # 2. Run Evaluation
            # Now passing 'model' directly
            estimates_df, extra_metrics = run_evaluation(
                model=model,
                cfg=cfg,
                sim_id=best_sim,
                exp_prefix=exp_prefix,
                args=args,
            )

            # 3. Tag and Store Results
            estimates_df["data_seed"] = seed
            all_estimates.append(estimates_df)

            if args.compute_extras and extra_metrics:
                # extra_metrics is a Dict of DataFrames (hsic, r2, etc.)
                # We need to tag each dataframe with the seed before merging

                # Tag 'recon' if it exists (it's usually in long format)
                if "recon" in extra_metrics:
                    extra_metrics["recon"]["data_seed"] = seed

                # For merging, we want to flatten the dict into one wide DataFrame per seed
                dfs_to_merge = []

                # Handle specific metrics that are returned
                for metric_name, df in extra_metrics.items():
                    # Make sure 'pop' column is int for clean merging
                    if "pop" in df.columns:
                        df["pop"] = df["pop"].astype(int)
                    # Add data_seed
                    df["data_seed"] = seed
                    dfs_to_merge.append(df)

                # Merge all metrics for this seed based on 'pop' and 'data_seed'
                if dfs_to_merge:
                    extras_df_seed = reduce(
                        lambda left, right: pd.merge(
                            left, right, on=["pop", "data_seed"], how="outer"
                        ),
                        dfs_to_merge,
                    )
                    all_extras.append(extras_df_seed)

        except Exception as e:
            print(f"Skipping seed {seed} due to error: {e}")
            # print(traceback.format_exc()) # Optional for debugging

    # --- Combine and Save ---

    metric_key = args.metric_key.replace("/", "_")
    suffix = "outsample" if args.use_test_data else "insample"
    if args.method != "tsls":
        suffix += f"_{args.method}"
    if args.exclude_sim_ids:
        suffix += "_no" + "".join(map(str, args.exclude_sim_ids))

    exp_results_dir = os.path.join("results", args.exp_grp)
    os.makedirs(exp_results_dir, exist_ok=True)

    # Save Estimates Summary
    if all_estimates:
        final_df = pd.concat(all_estimates, ignore_index=True)
        out_name = f"summary_{args.exp_id}_{args.ckpt_strategy}_{metric_key}_{suffix}.csv"
        out_path = os.path.join(exp_results_dir, out_name)

        final_df.to_csv(out_path, index=False)
        print(f"\nDONE! Saved master summary to {out_path}")
    else:
        print("\nNo estimates collected. Check your experiment IDs or paths.")

    # Save Extras Summary
    if args.compute_extras and all_extras:
        extras_final_df = pd.concat(all_extras, ignore_index=True)
        extras_out_name = f"extras_{args.exp_id}_{args.ckpt_strategy}_{metric_key}_{suffix}.csv"
        extras_out_path = os.path.join(exp_results_dir, extras_out_name)

        extras_final_df.to_csv(extras_out_path, index=False)
        print(f"DONE! Saved extra metrics to {extras_out_path}")


if __name__ == "__main__":
    main()
