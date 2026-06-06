DATA_DIR=/path/to/datasets
LOG_DIR=/path/to/log_dir


# short term
DATASET='solar_nips' # select from ['etth1', 'etth2', 'ettm1', 'ettm2', 'traffic_nips', 'electricity_nips', 'exchange_rate_nips', 'solar_nips']
MODEL=k2vae
CTX_LEN=24 # 30 for exchange_rate_nips and 24 for the others
PRED_LEN=24 # 30 for exchange_rate_nips and 24 for the others

python run.py --config config/stsf/${DATASET}/${MODEL}.yaml --seed_everything 1  \
    --data.data_manager.init_args.path ${DATA_DIR} \
    --trainer.default_root_dir ${LOG_DIR} \
    --data.data_manager.init_args.split_val true \
    --data.data_manager.init_args.dataset ${DATASET} \
    --data.data_manager.init_args.context_length ${CTX_LEN} \
    --data.data_manager.init_args.prediction_length ${PRED_LEN} \
    --trainer.max_epochs 50


DATASET='exchange_ltsf' # select from ['etth1', 'etth2', 'ettm1', 'ettm2', 'traffic_ltsf', 'electricity_ltsf', 'exchange_ltsf', 'weather_ltsf', 'ILI_ltsf']
MODEL=k2vae
CTX_LEN=96 # 36 for ILI-ltsf and 96 for the others
PRED_LEN=720 # [24, 36, 48, 60] for ILI-ltsf and [96, 192, 336, 720] for the others
# long term
python run.py --config config/ltsf/${DATASET}/${MODEL}.yaml --seed_everything 1  \
    --data.data_manager.init_args.path ${DATA_DIR} \
    --trainer.default_root_dir ${LOG_DIR} \
    --data.data_manager.init_args.split_val true \
    --data.data_manager.init_args.dataset ${DATASET} \
    --data.data_manager.init_args.context_length ${CTX_LEN} \
    --data.data_manager.init_args.prediction_length ${PRED_LEN} \
    --trainer.max_epochs 50