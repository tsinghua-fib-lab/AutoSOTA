# Running Inference with Timer

[Original Repository](https://github.com/thuml/Large-Time-Series-Model) | [Paper](https://arxiv.org/abs/2402.02368)

Follow these steps to set up and run inference using Timer:

1. Set up the [environment](../README.md#results-reproduction).
2. Run the inference script with the following commands:

```bash
MODEL='timer'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2' 'weather_ltsf' 'electricity_ltsf'; do
    for CTX_LEN in 96; do
        for PRED_LEN in 24 48 96 192 336 720; do
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --model.forecaster.init_args.ckpt_path './checkpoints/timer/Timer_67M_UTSD_4G.pt' \
                --data.test_batch_size 64
        done
    done
done
```

## Hyper-param in Inference

`use_ims` (default: false): Evaluate decoder-only models in the Iterative Multi-step (IMS) way or encoder-only forecasters in Direct Multi-step (DMS) approach

`sub_rand_ratio`: The ratio of training samples in few-shot scenarios.