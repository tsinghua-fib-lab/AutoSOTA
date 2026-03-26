# Running Inference with TimesFM

[Original Repository](https://github.com/google-research/timesfm) | [Paper](https://arxiv.org/abs/2310.10688)

Follow these steps to set up and run inference using TimesFM:

1. Set up the [environment](../README.md#results-reproduction).
2. Run the inference script with the following commands:

```bash
MODEL='timesfm'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2'; do
    for CTX_LEN in 96; do
        for PRED_LEN in 24 48 96 192 336 720; do
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --data.test_batch_size 64
        done
    done
done
```

## Hyper-param in Inference

`frequency` (default: 0): Chose from {0, 1, 2}.


- **0 (default):** High frequency, long horizon time series. We recommend using this for time series up to daily granularity.
- **1:** Medium frequency time series. We recommend using this for weekly and monthly data.
- **2:** Low frequency, short horizon time series. We recommend using this for anything beyond monthly, e.g., quarterly or yearly.


`window size` (default: None):  Window size of trend + residual decomposition


