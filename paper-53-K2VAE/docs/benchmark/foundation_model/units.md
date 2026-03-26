# Running Inference with UniTS

[Original Repository](https://github.com/mims-harvard/UniTS) | [Paper](https://arxiv.org/pdf/2403.00131)

Follow these steps to set up and run inference using UniTS:

1. Set up the [environment](../README.md#results-reproduction).
2. Run the inference script with the following commands:

```bash
MODEL='units'
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
                --model.forecaster.init_args.ckpt_path './checkpoints/units/units_x128_pretrain_checkpoint.pth' \
                --data.test_batch_size 64
        done
    done
done
```
