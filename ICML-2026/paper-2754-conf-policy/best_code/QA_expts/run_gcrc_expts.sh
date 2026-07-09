#!/bin/sh
#SBATCH -t 24:00:00 
#SBATCH --mem=32G
source .venv/bin/activate
python run_gcrc_expts.py --method_names monotized_losses_crc ltt gcrc --score_names selfevals --n_trials 50 # logprobs frequency selfevals
