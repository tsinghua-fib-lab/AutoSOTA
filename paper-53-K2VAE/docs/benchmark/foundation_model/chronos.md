# Running Inference with Chronos

[Original Repository](https://github.com/amazon-science/chronos-forecasting) | [Paper](https://arxiv.org/abs/2403.07815)

Follow these steps to set up and run inference using Chronos:

1. Set up the [environment](../README.md#results-reproduction).
2. Run the inference script with the following commands:

```bash
MODEL='chronos'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2' 'weather_ltsf'; do
    for CTX_LEN in 5000 96; do
        for PRED_LEN in 24 48 96 192 336 720; do
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --data.test_batch_size 1
        done
    done
done
```


## Hyper-param in Inference


`Temperature` (default: 1): If Temperature=0, the output is consistent. The bigger the more diverse

`top_k`(default: 50): Only conduct softmax for top-k logits.

`top-p` (default: 1): Nucleus sampling. The model sums the probabilities of the most likely next value in descending order and stops when the sum reaches p.

