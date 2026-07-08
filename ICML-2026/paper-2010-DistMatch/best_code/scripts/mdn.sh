export HYDRA_FULL_ERROR=1

uc_model="mdn"
fc_model="global_lstmmdn"


for dataset in "enbPI_electric" "enbPI_solar_atlanta" "enbPI_wind"
do
    key="${fc_model}_${uc_model}_${dataset}"
    log_dir="./logs/${uc_model}/"
    mkdir -p $log_dir

    nohup \
    python ./code/run_sweep_mdn_train_and_eval.py \
        "config/model_fc=$fc_model" \
        'config/task=default' \
        "config/dataset=$dataset" \
        "config.trainer.trainer_config.n_epochs=150" \
        "config.trainer.trainer_config.optim.lr=0.005" \
        "config.experiment_data.experiment_name=$key" \
        "config.experiment_data.base_proj_dir=ABSOLUTE_PATH_TO_PROJECT_ROOT" \
        "config.experiment_data.project_name=dist_match" \
        "config.experiment_data.offline=true" \
    > "./$log_dir/$key.log" 2> "./$log_dir/$key.err" &
done