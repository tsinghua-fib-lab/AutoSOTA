#!/usr/bin/env python3
import subprocess
import os

policies = ["eig", "uncertainty", "random"]
rootdir = "./results/synthetic"

# Individual plots for each policy
for policy in policies:
    outdir = os.path.join(rootdir, policy)

    # Generate plots for this policy
    print(f"Generating plots for {policy} ...")
    plot_args = [
        "python", "plot_logs.py",
        "--outdir", outdir  # ,
        # "--pick_run", f"{policy}_seed1"  # pick one run for heatmaps/orient
    ]
    subprocess.run(plot_args)

# Summary plots for all policies
plot_args = ["python", "compare_policies_synthetic.py", "--rootdir", rootdir]
subprocess.run(plot_args)


