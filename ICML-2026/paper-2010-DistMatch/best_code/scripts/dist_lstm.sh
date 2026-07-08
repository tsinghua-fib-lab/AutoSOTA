export HYDRA_FULL_ERROR=1

uc_model=dist_match
task=default
fc_model="global_lstm"

for gamma in 0.01 0.025 0.05 0.075 0.1
do
    for dataset in "enbPI_electric" "enbPI_solar_atlanta" "enbPI_wind" "stock_meta_5m" "stock_nflx_5m"
    do
        key="${fc_model}_${uc_model}_${dataset}"
        log_dir="./logs/${uc_model}_lstm_${gamma}/"
        mkdir -p $log_dir

        nohup \
        python ./code/run_sweep_eval.py \
            "config/model_fc=$fc_model" \
            "config/model_uc=$uc_model" \
            "config/task=$task" \
            "config/dataset=$dataset" \
            "config/evaluation=plot_test.yaml" \
            "config.experiment_data.experiment_name=$key" \
            "config.experiment_data.base_proj_dir=ABSOLUTE_PATH_TO_PROJECT_ROOT" \
            "config.experiment_data.project_name=dist_match" \
            "config.model_uc.match_method=ks_stat" \
            "config.model_uc.match_threshold=$gamma" \
        > "./$log_dir/$key.log" 2> "./$log_dir/$key.err" &
    done
done