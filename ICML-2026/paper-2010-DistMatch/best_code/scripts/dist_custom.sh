export HYDRA_FULL_ERROR=1

uc_model=dist_match
n_recent=300
custom_dir_key=linear


for gamma in 0.01 0.015 0.025 0.05
do
    for seed in 20 30 40 50
    do
        for dataset in "enbPI_electric" "enbPI_solar_atlanta" "enbPI_wind" "stock_meta_5m" "stock_nvda_5m"
        do
            for fc_model in "darts_forest"
            do
                key="${fc_model}_${uc_model}_${dataset}_s${seed}"
                log_dir="./logs/${uc_model}_${custom_dir_key}_${gamma}/"
                mkdir -p $log_dir

                nohup \
                python ./code/main.py \
                    "config/model_fc=$fc_model" \
                    "config/model_uc=$uc_model" \
                    "config/task=default" \
                    "config/dataset=$dataset" \
                    "config/evaluation=plot_test.yaml" \
                    "config.experiment_data.experiment_name=$key" \
                    "config.experiment_data.base_proj_dir=ABSOLUTE_PATH_TO_PROJECT_ROOT" \
                    "config.experiment_data.project_name=dist_match" \
                    "config.experiment_data.seed=$seed" \
                    "config.model_uc.match_method=ks_stat" \
                    "config.model_uc.qrf_param.quantile_regressor=linear" \
                    "config.experiment_data.offline=true" \
                    "config.model_uc.match_threshold=$gamma" \
                > "$log_dir/$key.log" 2> "$log_dir/$key.err" &
            done
        done
    done
done