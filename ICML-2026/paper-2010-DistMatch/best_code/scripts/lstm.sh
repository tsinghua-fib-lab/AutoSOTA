export HYDRA_FULL_ERROR=1

fc_model=global_lstm

for dataset in "enbPI_electric" "enbPI_solar_atlanta" "enbPI_wind"
do
    python ./code/run_sweep_lstm_train.py \
        "config/model_fc=$fc_model" \
        'config/task=default' \
        "config/dataset=$dataset" \
        "config.trainer.trainer_config.n_epochs=150" \
        "config.model_fc.model_params.plot_eval_after_train=false" \
        "config.model_fc.model_params.lstm_conf.hidden_size=256" \
        "config.model_fc.model_params.dropout=0.1" \
        "config.model_fc.model_params.batch_size=512" \
        "config.trainer.trainer_config.optim.lr=0.0001" \
        "config.experiment_data.experiment_name=${fc_model}_${dataset}" \
        "config.experiment_data.base_proj_dir=ABSOLUTE_PATH_TO_PROJECT_ROOT" \
        "config.experiment_data.project_name=dist_match" \
        "config.experiment_data.offline=true"
done