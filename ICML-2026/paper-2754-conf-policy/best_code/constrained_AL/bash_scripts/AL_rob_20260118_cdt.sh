#!/bin/sh
#SBATCH -t 48:00:00 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --partition=parallel
#SBATCH --gpus-per-task=0
source ../../.venv/bin/activate
python ../run_SplitCP_MFCS_ALexpts.py --dataset robot_arm --n_steps 100 --n_initial_all 80 --n_seed 200 --lmbda 10.0 --p_split_train 0.8 --noise_magnitude 0.05 --weight_depth_maxes 1 --initial_sampling_bias 8.0 --add_to_train_split_prob 0.5 --noise_level 0.05 --sigma_0 0.05 --prob_bound_inf 1.0 --n_queries_ann 1 --constrain_vs_init Y --pc_alpha 0.2  --risk_control Y --cdt_risk_control Y --cdt_step_size 0.5
