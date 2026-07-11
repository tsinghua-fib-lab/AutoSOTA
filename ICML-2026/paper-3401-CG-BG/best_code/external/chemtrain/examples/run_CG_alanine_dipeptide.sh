#!/usr/bin/env bash

# Disable everything related to fusions to potentially avoid issue
# common_utilization <= producer_output_utilization

# export XLA_FLAGS=""

export DATA_PATH="../../examples/data"

/usr/bin/env

jupytext --output ../docs/examples/execute.ipynb CG_alanine_dipeptide.md
cd ../docs/examples && papermill execute.ipynb CG_alanine_dipeptide.ipynb --progress-bar --request-save-on-cell-execute
cp CG_alanine_dipeptide.ipynb ../../examples/CG_alanine_dipeptide.ipynb
