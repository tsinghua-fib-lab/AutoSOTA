# Running Inference with MOIRAI

[Original Repository](https://github.com/SalesforceAIResearch/uni2ts) | [Paper](https://arxiv.org/abs/2402.02592)

Follow these steps to set up and run inference using MOIRAI:

1. Set up the [environment and initialize submodules](../README.md#results-reproduction).
2. Run the inference script with the following commands:

```bash
MODEL='moirai'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2' 'weather_ltsf' 'electricity_ltsf'; do
    for CTX_LEN in 5000 96; do
        for PRED_LEN in 24 48 96 192 336 720; do
            python run.py --config config/tsfm/${MODEL}/context_${CTX_LEN}/${DATASET}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN}
        done
    done
done
```

## Hyper-param in Inference

`patch size` (default: `auto`): Specifies the patch size used during inference. When set to `auto`, the model selects the patch size that minimizes validation loss based on historical data.

`variate_mode` (default: `S`): Determines whether the model operates in univariate (`S`) or multivariate mode (`M`) during inference.