# Running Inference with Lag-Llama

[Original Repository](https://github.com/time-series-foundation-models/lag-llama) | [Paper](https://arxiv.org/abs/2310.08278)

Follow these steps to set up and run inference using Lag-Llama:

1. Set up the [environment and initialize submodules](../README.md#results-reproduction).
2. Run the inference script with the following commands:

```bash
# Lag-Llama
MODEL='lag_llama'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2' 'weather_ltsf'; do
    for CTX_LEN in 512; do
        for PRED_LEN in 24 48 96 192 336 720; do
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --model.forecaster.init_args.ckpt_path './checkpoints/lag-llama/lag-llama.ckpt' \
                --data.test_batch_size 1
        done
    done
done
```
