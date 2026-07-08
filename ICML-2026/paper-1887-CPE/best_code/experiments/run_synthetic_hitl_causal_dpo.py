#!/usr/bin/env python3
import subprocess
import os

# seeds = list(range(1, 31))   # 30 seeds
seeds = list(range(1, 11))
policies = ["eig", "uncertainty", "random", "static_eig"]
# policies = ["static_random", "static_uncertainty", "static_eig"]

for policy in policies:
    outdir = os.path.join("./results/synthetic", policy)
    os.makedirs(outdir, exist_ok=True)

    common_args = [
        "python", "synthetic_hitl_causal_dpo.py",
        "--outdir", outdir,
        "--D", "20",
        "--S", "10000",
        "--T", "190",
        "--edge_prob_true", "0.25",
        "--flip_prob", "0.1",
        "--add_remove_prob", "0.05",
        "--weight_noise", "0.2",
        "--beta_edge", "10.0",
        "--beta_dir", "10.0",
        "--lam", "0.0",
        "--screen_k", "200",
        "--resample_threshold", "0.6",
        "--policy", policy,
        "--rejuvenate_samples",
        "--rejuvenate_steps", "2",
        "--mutate_fraction", "0.5",
    ]

    for seed in seeds:
        prefix = f"{policy}_seed{seed}"
        args = common_args + ["--seed", str(seed), "--save_prefix", prefix]
        print("Running:", " ".join(args))
        subprocess.run(args)

    # Generate plots for this policy
    # print(f"Generating plots for {policy} ...")
    # plot_args = [
    #     "python", "plot_logs.py",
    #     "--outdir", outdir#,
    #     #"--pick_run", f"{policy}_seed1"  # pick one run for heatmaps/orient
    # ]
    # subprocess.run(plot_args)

