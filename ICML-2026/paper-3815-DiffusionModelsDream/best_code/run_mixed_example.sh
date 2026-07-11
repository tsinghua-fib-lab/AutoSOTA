#!/bin/bash

# remove any contmination from the host first
# unset LD_LIBRARY_PATH
# export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"

# deactivate

#########################
# CONFIGURATION
#########################

# Set investigation json.
investigation_file="./example_settings_mixed.json"

# Set the base directory where the output will be saved.
sample_dir="./output"
base_inv_root="./output/mass_mixed"

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
mixed_out_path_cond="./output/mass_mixed/generated_samples_cond_mass_t1.txt"
mixed_out_path_marg="./output/mass_mixed/generated_samples_marg_t0.txt"

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

echo "Running mixed investigation pipeline..."

python sample_mixed.py \
    --cuda-visible-devices "${GPU_ID}" \
    --project-root "${project_root}" \
    --data-csv "${data_csv}" \
    --indices-pkl "${indices}" \
    --checkpoint-dir "${checkpoint}" \
    --investigations "${investigation_file}" \
    --output-root "${sample_dir}" \
    --mixed-root "${mixed_root}" \
    --mixed-ckpt "${mixed_checkpoint_path}" \
    --mixed-sde "${mixed_sde_path}" \
    --mixed-out-cond "${mixed_out_path_cond}" \
    --mixed-out-marg "${mixed_out_path_marg}" \
    --mixed-token "${mixed_token}" \
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
          "name": "num wing",
          "x": "/design_tree/main_wings/1/span_proj",
          "function": "lambda x: 1+np.where(np.isnan(x), 0, 1)"
        },
        "/res_metrics/comp_weights/total_wing_weight",
        "/design_tree/battery/mass",
        {
          "name": "Number of forward rotors",
          "x": "/design_tree/main_wings/0/prop_arms/0/[ForwardPropArm]/origin_z_rel",
          "y": "/design_tree/main_wings/1/prop_arms/0/[ForwardPropArm]/origin_z_rel",
          "function": "lambda x,y: 1+2*np.where(np.isnan(x), 0, 1)+2*np.where(np.isnan(y), 0, 1)"
        },
        {
          "name": "Number of lift rotors",
          "x": "/design_tree/main_wings/0/prop_arms/0/[LiftRotorArmSymmetric]/origin_z_rel",
          "y": "/design_tree/main_wings/1/prop_arms/0/[LiftRotorArmSymmetric]/origin_z_rel",
          "z": "/design_tree/main_wings/0/prop_arms/1/[LiftRotorArmSymmetric]/origin_z_rel",
          "a": "/design_tree/main_wings/1/prop_arms/1/[LiftRotorArmSymmetric]/origin_z_rel",
          "function": "lambda x,y,z,a: 4*np.where(np.isnan(x), 0, 1)+4*np.where(np.isnan(y), 0, 1)+4*np.where(np.isnan(z), 0, 1)+4*np.where(np.isnan(a), 0, 1)"
        },
        "/design_tree/forward_prop/number_of_blades",
        "/design_tree/forward_prop/tip_radius"
    ]' \
    --legend-config '{
        "format_string": "Total Mass: {:.0f} kg",
        "keys": ["/res_metrics/comp_weights/total"]
    }' \
    --downsample 500 \
    --save-png \
    --out-png all_investigations_corner.png \
    --categorical '["Number of Wings","Number Of Forward Rotors","Number Of Lift Rotors"]' \
    --extra-plots '[[1,0],[3,1],[-1,0],[-1,2]]'\
    --format-labels '["Number of Wings","Total Wing\nMass [kg]","Battery\nMass [kg]","Number Of Forward Rotors","Number Of Lift Rotors","Number Of\nBlades","Lift Rotor\nRadius [m]"]'