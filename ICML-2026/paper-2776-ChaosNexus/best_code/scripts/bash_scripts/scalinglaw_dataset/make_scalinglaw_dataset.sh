#!/bin/bash

params_dir=$WORK/data/improved/scalinglaw/params_disjoint/

param_dicts_splits=(
    params_dict_split_0-163.json
    params_dict_split_163-327.json
    params_dict_split_327-655.json
    params_dict_split_655-1311.json
    params_dict_split_1311-2622.json
    params_dict_split_2622-5244.json
    params_dict_split_5244-10489.json
)

n_ics_splits=(
    128
    64
    32
    16
    8
    4
    2
)

# Assert that the number of elements in n_ics_splits matches param_dicts_splits
if [ ${#n_ics_splits[@]} -ne ${#param_dicts_splits[@]} ]; then
    echo "Error: n_ics_splits and param_dicts_splits must have the same length"
    echo "n_ics_splits has ${#n_ics_splits[@]} elements"
    echo "param_dicts_splits has ${#param_dicts_splits[@]} elements"
    exit 1
fi

for i in "${!param_dicts_splits[@]}"; do
    params_json_fname=${param_dicts_splits[$i]}
    n_ics=${n_ics_splits[$i]}
    echo "params_json_fname: $params_json_fname"
    # Extract the numerical suffix (num1-num2) from params_json_fname
    suffix=$(echo "$params_json_fname" | grep -o "[0-9]\+-[0-9]\+")
    echo "Numerical suffix: $suffix"
    echo "n_ics: $n_ics"
    # Check if params_json_path exists
    params_json_path="${params_dir}/${params_json_fname}"
    if [ ! -f "$params_json_path" ]; then
        echo "Error: Parameter file does not exist: $params_json_path"
        echo "Skipping this split"
        continue
    fi
    echo "Parameter file exists: $params_json_path"

    python scripts/make_dataset_from_params.py \
        restart_sampling.params_json_path=$params_json_path \
        restart_sampling.systems_batch_size=128 \
        sampling.data_dir=$WORK/data/improved/scalinglaw/split_${suffix}_ic${n_ics} \
        sampling.rseed=1000 \
        sampling.num_ics=$n_ics \
        sampling.num_points=5120 \
        sampling.num_periods=40 \
        sampling.num_periods_min=40 \
        sampling.num_periods_max=40 \
        sampling.split_coords=false \
        sampling.atol=1e-10 \
        sampling.rtol=1e-8 \
        sampling.silence_integration_errors=true \
        events.verbose=false \
        events.max_duration=400 \
        validator.transient_time_frac=0.2 \
        wandb.log=false \
        wandb.project_name=dyst_data \
        "$@"
done