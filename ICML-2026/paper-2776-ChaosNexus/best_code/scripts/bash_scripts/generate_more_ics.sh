#!/bin/bash

data_dir=$WORK/data/improved/final_skew40
split_name=test_zeroshot

params_json_path=$data_dir/parameters/${split_name}/filtered_params_dict.json

# directory to save the new dataset
data_split_dir_more_ics=$data_dir/${split_name}_more_ics

n_ics=16

python scripts/make_dataset_from_params.py \
    if [ ! -f "$params_json_path" ]; then
        echo "Error: Parameter file does not exist: $params_json_path"
        echo "Skipping this split"
        exit 1
    fi
    echo "Parameter file exists: $params_json_path"

    python scripts/make_dataset_from_params.py \
        restart_sampling.params_json_path=$params_json_path \
        restart_sampling.systems_batch_size=128 \
        restart_sampling.batch_idx_low=0 \
        restart_sampling.batch_idx_high=16 \
        sampling.data_dir=$data_split_dir_more_ics \
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