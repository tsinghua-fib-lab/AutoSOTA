#!/bin/bash
#SBATCH -N 1                       # Total number of nodes
#SBATCH -C gpu                    # Constraint: use GPU nodes
#SBATCH -G 1                      # Total GPUs across all nodes
#SBATCH -q regular                # Queue name
#SBATCH -J Protein_Training                # Job name
#SBATCH --mail-user=kkeega3@emory.edu
#SBATCH --mail-type=ALL
#SBATCH -t 14:00:00               # Max wall time
#SBATCH -A m1266                 # Project allocation

./training/protein.sh