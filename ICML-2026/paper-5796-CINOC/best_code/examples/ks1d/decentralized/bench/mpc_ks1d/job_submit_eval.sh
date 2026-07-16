#!/bin/bash
#SBATCH --job-name=KS1D_MPC_eval
#SBATCH --time=12:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --account=sgoswam4
#SBATCH --mail-type=end
#SBATCH --mail-user=droysar1@jhu.edu

source $HOME/jax_torch2.venv/bin/activate

cd /home/droysar1/scr4_sgoswam4/Dibakar/multi_agent_dpc/CINOC/examples/ks1d/decentralized/bench/mpc_ks1d/

echo "Starting KS1D MPC Evaluation with Timing..."
echo "======================================="

python3 -u evaluate_mpc_ks1d.py --num-samples 10

echo ""
echo "MPC Evaluation Complete!"
