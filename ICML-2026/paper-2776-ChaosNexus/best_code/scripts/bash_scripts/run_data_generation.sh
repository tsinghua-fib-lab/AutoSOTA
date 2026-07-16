#!/bin/bash

# read debug flag
DEBUG=0
while getopts "d" flag; do
        case "${flag}" in
                d) DEBUG=1;;
        esac
done
shift $((OPTIND - 1))

ulimit -n 99999
if [ "$DEBUG" -eq 0 ]; then

    TOTAL_CORES=$(nproc)
    CORES_PER_GROUP=$(( $TOTAL_CORES / 2 ))
    CORES_PER_JOB=$(( $CORES_PER_GROUP / 4 ))

    # skew systems
    WANDB_DISABLE_GPU=true python -W ignore scripts/make_skew_systems.py \
        sampling.num_points=5120 \
        sampling.num_periods=40 \
        sampling.num_periods_min=50 \
        sampling.num_periods_max=50 \
        sampling.num_param_perturbations=100 \
        sampling.num_ics=1 \
        sampling.param_scale=1.0 \
        sampling.atol=1e-10 \
        sampling.rtol=1e-8 \
        sampling.silence_integration_errors=true \
        sampling.data_dir=./data_scaling/19962-1 \
        sampling.rseed=414328 \
        sampling.ic_rseed=414 \
        sampling.test_split=0.3 \
        sampling.verbose=false \
        multiprocess_kwargs.processes=128 \
        events.verbose=false \
        events.max_duration=300 \
        skew.normalization_strategy=flow_rms \
        skew.transform_scales=true \
        skew.randomize_driver_indices=true \
        skew.num_pairs=19968 \
        skew.pairs_rseed=328 \
        skew.sys_idx_low=0 \
        skew.sys_idx_high=16512 \
        validator.transient_time_frac=0.2 \
        run_name=new_skew40 \
        wandb.log=true \
        wandb.project_name=dyst_data \
        "$@"


    # base dysts
    # WANDB_DISABLE_GPU=true python -W ignore scripts/make_dyst_data.py \
    #    sampling.sys_class=continuous \
    #    sampling.num_points=8192 \
    #    sampling.num_periods=40 \
    #    sampling.num_periods_min=20 \
    #    sampling.num_periods_max=60 \
    #    sampling.num_param_perturbations=200 \
    #    sampling.param_scale=1.0 \
    #    sampling.atol=1e-10 \
    #    sampling.rtol=1e-8 \
    #    sampling.silence_integration_errors=true \
    #    sampling.data_dir=./scaleformer/data/big_base_mixedp \
    #    sampling.rseed=21433 \
    #    sampling.verbose=false \
    #    events.verbose=false \
    #    events.max_duration=600 \
    #    validator.transient_time_frac=0.5 \
    #    run_name=big_base_mixedp \
    #    "$@"

else
    WANDB_DISABLE_GPU=true python -W ignore scripts/make_skew_systems.py \
        sampling.num_points=5120 \
        sampling.num_periods=40 \
        sampling.num_periods_min=25 \
        sampling.num_periods_max=75 \
        sampling.num_param_perturbations=2 \
        sampling.num_ics=3 \
        sampling.param_scale=1.0 \
        sampling.atol=1e-10 \
        sampling.rtol=1e-8 \
        sampling.silence_integration_errors=true \
        sampling.data_dir=./scaleformer/data/skew_debug \
        sampling.rseed=414328 \
        sampling.ic_rseed=414 \
        sampling.verbose=false \
        sampling.multiprocessing=true \
        multiprocess_kwargs.processes=64 \
        multiprocess_kwargs.maxtasksperchild=4 \
        events.verbose=false \
        events.max_duration=400 \
        skew.normalization_strategy=flow_rms \
        skew.transform_scales=true \
        skew.randomize_driver_indices=true \
        skew.num_pairs=2048 \
        skew.pairs_rseed=328 \
        skew.sys_idx_low=0 \
        skew.sys_idx_high=128 \
        validator.transient_time_frac=0.2 \
        run_name=skew_debug \
        wandb.log=false \
        wandb.project_name=dyst_data \
        "$@"
fi
