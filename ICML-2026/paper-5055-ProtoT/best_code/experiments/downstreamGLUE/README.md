# Downstream GLUE Experiments

This folder contains the GLUE fine-tuning and prediction code used for downstream evaluation of Prototype Attention and baseline sequence models.

## Setup

Install the core dependencies:

```bash
pip install -r experiments/downstreamGLUE/requirements.txt
```

Optional acceleration or baseline dependencies:

```bash
pip install flash-linear-attention
pip install flash-attn
pip install mamba-ssm
```

## Checkpoint Layout

Place pretrained checkpoints under `checkpoints/` using this layout:

```text
checkpoints/
  ProtoAttn_large_FineWeb_scheduler/
    args.json
    model_state_dict.pth
  LLaMA_large_FineWeb_scheduler/
    args.json
    model_state_dict.pth
  mamba1_large_fineweb/
    args.json
    model_state_dict.pth
  deltanet_large_fineweb/
    args.json
    model_state_dict.pth
```

## Fine-Tuning

Run one model on selected tasks:

```bash
PYTHONPATH="$PWD" python experiments/downstreamGLUE/GLUE_PIPELINE/GLUE_TRAINER.py \
  --model protoattn \
  --tasks sst2 mnli \
  --tokenizer_path tok/fineweb_bpe_16000.json \
  --data_cache data \
  --output_root outputs \
  --device cuda:0
```

Run all configured models:

```bash
bash experiments/downstreamGLUE/GLUE_PIPELINE/RUN_GLUE_EXPERIMENTS.sh
```

You can override paths without editing code:

```bash
GLUE_CHECKPOINT_ROOT=/path/to/checkpoints \
GLUE_DATA_CACHE=/path/to/glue_cache \
GLUE_OUTPUT_ROOT=/path/to/outputs \
bash experiments/downstreamGLUE/GLUE_PIPELINE/RUN_GLUE_EXPERIMENTS.sh
```

## Prediction and Submission

Generate GLUE test files from fine-tuned checkpoints:

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

The script writes task TSV files and a zipped submission archive under `submissions/<model>/`.
