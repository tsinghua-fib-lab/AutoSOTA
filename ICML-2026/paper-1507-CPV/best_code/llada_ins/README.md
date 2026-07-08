# COVER LLaDA Reproduction

This package contains the COVER evaluation code for LLaDA-8B-Instruct and LLaDA-1.5 on HumanEval, MBPP, GSM8K, and MATH500.

## Environment

The tested environment is exported in `requirement.txt` from the `dllm` conda environment.

```bash
conda create -n cover python=3.12 -y
conda activate cover
pip install -r requirement.txt
```

The original runs used CUDA 12.6, cuDNN 9.7, NVIDIA driver 570.x, and 4 NVIDIA H200 GPUs. Other A100/H100/H200 setups should work, but small accuracy differences can occur because the runs use bfloat16 multi-GPU inference.

## Layout

```text
.
├── code/
│   ├── evaluation_script.py
│   ├── metrics/                    # task-specific post-processing metrics
│   └── dllm_eval/
│       ├── models/cover/           # COVER model and generation implementation
│       └── tasks/                  # task YAMLs and helpers
├── scripts/
│   ├── run_cover_256.sh
│   └── run_cover_512.sh
├── requirement.txt
└── README.md
```

## Run

Run all four benchmarks with LLaDA-8B-Instruct:

```bash
cd /path/to/COVER_OS/llada_ins
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_cover_256.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_cover_512.sh
```

Run all four benchmarks with LLaDA-1.5:

```bash
MODEL_PATH=GSAI-ML/LLaDA-1.5 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_cover_256.sh
MODEL_PATH=GSAI-ML/LLaDA-1.5 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_cover_512.sh
```

Run a single benchmark:

```bash
ONLY_TASK=humaneval CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_cover_256.sh
ONLY_TASK=math500 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_cover_256.sh
```

Results are written under:

```text
results/<model_key>/cover/<task>_<run_tag>/
```

Each run writes aggregated JSON, sample JSONL, and a task-specific `metrics_<task>.txt` file.

## Implementation Notes

COVER is registered as `LLaDA_cover`. The core implementation is in:

- `code/dllm_eval/models/cover/LLaDA_cover.py`
- `code/dllm_eval/models/cover/contextual_cover.py`
- `code/dllm_eval/models/cover/modeling_llada_kv_cover.py`

The implementation includes the practical progress safeguards used for the reported runs: at least one token is drafted per step when masked positions remain, and same-step ReMask operations are capped so the block makes positive net progress.
