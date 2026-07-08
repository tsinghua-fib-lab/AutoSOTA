export HYDRA_FULL_ERROR=1

uc_model=adaptiveci

for dataset in "enbPI_electric" "enbPI_solar_atlanta" "enbPI_wind"
for dataset in "enbPI_electric" "enbPI_solar_atlanta" "enbPI_wind"
do
    for fc_model in "darts_lightgbm_quantile"
    do
        key="${fc_model}_${uc_model}_${dataset}"
        nohup \
        python ./code/main.py \
            "config/model_fc=$fc_model" \
            "config/model_uc=$uc_model" \
            'config/task=default' \
            "config/dataset=$dataset" \
            'config/evaluation=plot_test.yaml' \
            "config.experiment_data.experiment_name=${fc_model}_${uc_model}_${dataset}" \
            "config.experiment_data.base_proj_dir=ABSOLUTE_PATH_TO_PROJECT_ROOT" \
            "config.experiment_data.project_name=dist_match" \
        > logs/$key.log 2> logs/$key.err &
    done
done