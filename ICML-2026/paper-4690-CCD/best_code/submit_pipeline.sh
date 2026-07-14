#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH --partition=node2
#SBATCH --gres=gpu:1
#SBATCH --output=output/pipeline_%A.out
#SBATCH --error=output/pipeline_%A.err
#SBATCH -t 2-00:00:00

# Execute the entire pipeline from Step 1 to Step 3
./run_pipeline.sh
