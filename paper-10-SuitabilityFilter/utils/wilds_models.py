#!/usr/bin/env python
import time
import os

datasets = ["fmow", "civilcomments", "rxrx1"]
seeds = [0, 1, 2]
algorithms = ["ERM", "groupDRO", "IRM"]
root_dir = "/path/to/wilds/data/" # Change this to your root directory
log_dir = "/path/to/experiment/logs/" # Change this to your log directory
wandb_api_key_path = "/path/to/wandb_key.txt" # Change this to your wandb API key path

if not os.path.isdir("./job_files"):
    os.mkdir("./job_files")

for data in datasets:
    for seed in seeds:
        for algorithm in algorithms:
            job_file = f"./job_files/wilds_exps_{data}_{seed}_{algorithm}.slrm"
            with open(job_file, "w+") as fh:
                fh.writelines("#!/bin/bash\n")
                # Modify the following lines according to your SLURM configuration
                fh.writelines("#SBATCH -A nic\n") 
                fh.writelines("#SBATCH -q nic\n")
                fh.writelines("#SBATCH -p nic\n")
                fh.writelines("#SBATCH --gres=gpu:1\n")
                fh.writelines("#SBATCH -c 4\n")
                fh.writelines("#SBATCH --mem=40GB\n")
                fh.writelines("#SBATCH --output=./out_err_slurm/%j.out\n")
                fh.writelines("#SBATCH --error=./out_err_slurm/%j.err\n")
                fh.writelines(
                    f"python wilds/examples/run_expt.py --dataset {data} --seed {seed} --algorithm {algorithm} --root_dir {root_dir} --log_dir {log_dir}{algorithm}/{data} --use_wandb --wandb_api_key_path {wandb_api_key_path} --wandb_kwargs project=WILDS name={data} \n"
                )
            os.system("sbatch %s" % job_file)
            time.sleep(0.3)
