# GLUE Pipeline Quick Start

Fine-tune a single model:

```bash
PYTHONPATH="$PWD" python experiments/downstreamGLUE/GLUE_PIPELINE/GLUE_TRAINER.py \
  --model protoattn \
  --device cuda:0 \
  --seed 123 \
  --tasks cola sst2 mrpc qqp stsb mnli qnli rte wnli \
  --tokenizer_path tok/fineweb_bpe_16000.json \
  --data_cache data \
  --output_root outputs
```

Generate test submissions:

```bash
PYTHONPATH="$PWD" python experiments/downstreamGLUE/GLUE_PIPELINE/GLUE_PREDICT.py \
  --model protoattn \
  --checkpoint_root outputs \
  --submission_root submissions \
  --tokenizer_path tok/fineweb_bpe_16000.json \
  --data_cache data \
  --split test \
  --include_ax \
  --device cuda:0
```

Run all configured models with the helper script:

```bash
bash experiments/downstreamGLUE/GLUE_PIPELINE/RUN_GLUE_EXPERIMENTS.sh
```

Useful overrides:

```bash
MODELS="protoattn llama" PROTO_GPU=0 LLAMA_GPU=1 bash experiments/downstreamGLUE/GLUE_PIPELINE/RUN_GLUE_EXPERIMENTS.sh
```

```bash
GLUE_CHECKPOINT_ROOT=/path/to/checkpoints \
GLUE_DATA_CACHE=/path/to/glue_cache \
GLUE_OUTPUT_ROOT=/path/to/outputs \
bash experiments/downstreamGLUE/GLUE_PIPELINE/RUN_GLUE_EXPERIMENTS.sh
```
