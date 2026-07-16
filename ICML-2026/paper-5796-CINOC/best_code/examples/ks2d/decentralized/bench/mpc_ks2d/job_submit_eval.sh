#!/bin/bash
#SBATCH --job-name=KS2D_MPC_eval
#SBATCH --time=24:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --account=sgoswam4
#SBATCH --mail-type=end
#SBATCH --mail-user=droysar1@jhu.edu

source $HOME/jax_torch2.venv/bin/activate

cd /home/droysar1/scr4_sgoswam4/Dibakar/multi_agent_dpc/CINOC/examples/ks2d/decentralized/bench/mpc_ks2d/

echo "Starting KS2D MPC Evaluation with Timing..."
echo "======================================="

python3 -u evaluate_mpc_ks2d.py --num-samples 10

echo ""
echo "MPC Evaluation Complete!"