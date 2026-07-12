#!/bin/bash

# Create the environment from the yml file
# We use --prune to ensure it matches the file exactly
echo "Creating/Updating Conda environment: pirate_env..."
conda env create -f environment.yml || conda env update -f environment.yml

echo "Setup complete! Activate your environment with: conda activate pirate_env"
