#!/bin/bash
#SBATCH -N 1                       # Total number of nodes
#SBATCH -C gpu                    # Constraint: use GPU nodes
#SBATCH -G 1                      # Total GPUs across all nodes
#SBATCH -q regular                # Queue name
#SBATCH -J 3D_Training_NF_Sphere                # Job name
#SBATCH --mail-user=kkeega3@emory.edu
#SBATCH --mail-type=ALL
#SBATCH -t 8:00:00               # Max wall time
#SBATCH -A m1266                 # Project allocation

./training/smileyface_sphere_nf.sh