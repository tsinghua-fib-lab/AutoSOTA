#!/bin/sh
#SBATCH --job-name=proc
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=01:00:00
#SBATCH --output=652_output.out

pwd; hostname; date

echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
Rscript 653_ICMLPlots_SuppMat_Full.R
Rscript 654_ICMLPlots_Sensitivity.R
Rscript 655_TimePlots_Full.R
echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
echo a
echo a

date

