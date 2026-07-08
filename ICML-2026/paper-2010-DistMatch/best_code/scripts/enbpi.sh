export HYDRA_FULL_ERROR=1

uc_model=enbpi
seed=30

for dataset in "enbPI_electric" "enbPI_solar_atlanta" "enbPI_wind" "stock_meta_5m" "stock_nvda_5m"
for dataset in "enbPI_electric" "enbPI_solar_atlanta" "enbPI_wind" "stock_meta_5m" "stock_nvda_5m"
do
    for fc_model in "darts_forest" "darts_lightgbm" "darts_tcn"
    do
        key="${fc_model}_${uc_model}_${dataset}_s${seed}"
        log_dir="./logs/$uc_model"
        mkdir -p $log_dir

        nohup \
        python ./code/main.py \
            "config/model_fc=$fc_model" \
            "config/model_uc=$uc_model" \
            'config/task=default' \
            "config/dataset=$dataset" \
            'config/evaluation=plot_test.yaml' \
            "config.experiment_data.experiment_name=$key" \
            "config.experiment_data.base_proj_dir=ABSOLUTE_PATH_TO_PROJECT_ROOT" \
            "config.experiment_data.project_name=dist_match" \
            "config.experiment_data.seed=$seed" \
        > "$log_dir/$key.log" 2> "$log_dir/$key.err" &
    done
done