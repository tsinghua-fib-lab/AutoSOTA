#!/bin/bash

# remove any contmination from the host first
# unset LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/.singularity.d/libs

# deactivate

#########################
# CONFIGURATION
#########################

# Set investigation json.
investigation_file="./example_settings_maskedit.json"

# Set the base directory where the output will be saved.
sample_dir="./output"
base_inv_root="./output/total_mass"

# Set the dataset and project root
project_root="./training_data/maskedit/data/"
checkpoint="./training_data/maskedit/model/MaskeditModel"
data_csv="train_set.csv"
indices="train_indices.pkl"

# Set up the mixed diffusion root and checkpoint
mixed_root="./training_data/mixedit/model"
mixed_token="./training_data/mixedit/data/"
mixed_checkpoint_path="/checkpoint_ep1000.pth"
mixed_sde_path="/sde_stats.pkl"
mixed_out_path_cond="./output/total_mass/generated_samples_cond_mass_t1.txt"
mixed_out_path_marg="./output/total_mass/generated_samples_marg_t0.txt"

#########################
# RUN INVESTIGATION
#########################

nvidia-smi || echo "No GPU visible!"

echo "=== GPU diagnostic ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "which python=$(which python)"

# Extract the GPU ID
GPU_ID=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "${sample_dir}"

echo "--------------------------------------------------"
echo "Processing Investigation: ${investigation_file}"
echo "Sample Directory:         ${sample_dir}"
echo "--------------------------------------------------"

#!/usr/bin/env bash

echo "Running standard investigation pipeline..."

python -m maskedit.sample_maskedit \
    --cuda-visible-devices "${GPU_ID}" \
    --project-root "${project_root}" \
    --data-csv "${data_csv}" \
    --indices-pkl "${indices}" \
    --checkpoint-dir "${checkpoint}" \
    --investigations "${investigation_file}" \
    --output-root "${sample_dir}" \
    --Nsamples 500 \
    --sample-steps 500 \
    --nlayers 16

# Pick the newest investigation directory (sorted by name)
inv_root=$(ls -1d "${base_inv_root}"/*/ 2>/dev/null | sort | tail -n 1)

if [ -z "$inv_root" ]; then
    echo "ERROR: No investigation directories found in ${base_inv_root}"
    exit 1
fi

echo "Newest investigation root: $inv_root"

python plot_script.py \
    --run-dir ${inv_root} \
    --project-root ${project_root} \
    --data-csv ${data_csv} \
    --cols '[
        {
          "name": "cruise lift",
          "x": "/res_metrics/cruise_l",
          "function": "lambda x: x"
        },
        {
          "name": "lift rotor radius (m)",
          "x": "/design_tree/lift_prop/tip_radius",
          "function": "lambda x: x"
        },
        {
          "name": "lift rotor mass (kg)",
          "x": "/res_metrics/comp_weights/lift_rotors",
          "function": "lambda x: x"
        },
        {
          "name": "battery mass (kg)",
          "x": "/design_tree/battery/mass",
          "function": "lambda x: x"
        },
        {
          "name": "wing area (sq. m)",
          "x": "/design_tree/main_wings/0/chord_root",
          "y": "/design_tree/main_wings/0/span_proj",
          "function": "lambda x,y: x*y"
        },
        {
          "name": "wing mass (kg)",
          "x": "/res_metrics/comp_weights/total_wing_weight",
          "function": "lambda x: x"
        }
    ]' \
    --legend-config '{
        "title": "Total mass",
        "format_string": "{:.0f} kg",
        "keys": ["/res_metrics/comp_weights/total"]
    }' \
    --downsample 500 \
    --save-png \
    --out-png all_investigations_corner.png \
    --extra-plots '[[3,3],[4,3],[1,3]]'\
    --format-labels '["Cruise Lift","Lift Rotor\nRadius [m]","Lift Rotor\nMass [kg]","Battery Mass\n[kg]","Wing Area\n[m$^2$]","Wing Mass\n[kg]"]'