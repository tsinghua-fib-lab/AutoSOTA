#!/usr/bin/env bash

# Disable everything related to fusions to potentially avoid issue
# common_utilization <= producer_output_utilization

# export XLA_FLAGS=""

/usr/bin/env

jupytext --output ../docs/examples/execute_TI.ipynb AT_titanium_fused_training.md

cd ../docs/examples && papermill execute_TI.ipynb AT_titanium_fused_training.ipynb --progress-bar --request-save-on-cell-execute

cp AT_titanium_fused_training.ipynb ../../examples/AT_titanium_fused_training.ipynb
