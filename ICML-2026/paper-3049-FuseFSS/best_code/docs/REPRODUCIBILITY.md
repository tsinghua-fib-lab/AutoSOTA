# Reproducibility

This document lists the maintained artifact commands.

## Build

```bash
GPU_ARCH=120 bash scripts/build_sigma_pair.sh --variant all
```

This produces:

```text
build/gpu_mpc_upstream/sigma
build/gpu_mpc_vendor/sigma
ezpc_upstream/GPU-MPC/experiments/sigma/sigma
third_party/EzPC_vendor/GPU-MPC/experiments/sigma/sigma
```

## Unit and Canary Tests

```bash
cmake -S . -B build \
  -DFUSEFSS_ENABLE_CUDA=ON \
  -DFUSEFSS_ENABLE_TESTS=ON \
  -DFUSEFSS_ENABLE_BENCH=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Strict production canary:

```bash
SIGMA_MEMPOOL_DISABLE=1 SIGMA_PINNED_COMM_BUFS=0 SIGMA_PINNED_KEYBUF=0 SIGMA_COMM_BUF_MB=1024 \
  SUF_PRIMITIVE_KEYBUF_MB=4096 \
  ./build/gpu_mpc_vendor/sigma fusefss-primitive-canary 1048576 0 127.0.0.1 19401 37
SIGMA_MEMPOOL_DISABLE=1 SIGMA_PINNED_COMM_BUFS=0 SIGMA_PINNED_KEYBUF=0 SIGMA_COMM_BUF_MB=1024 \
  SUF_PRIMITIVE_KEYBUF_MB=4096 \
  ./build/gpu_mpc_vendor/sigma fusefss-primitive-canary 1048576 1 127.0.0.1 19401 37
```

The private-model guard canary is single-process:

```bash
./build/gpu_mpc_vendor/sigma fusefss-private-model-failfast-canary
```

## End-to-End Evaluation

Paper-style evaluation:

```bash
python3 scripts/repro_eval_latest.py \
  --tasks bert-tiny:128,bert-base:128,bert-large:128,gpt2:128,gpt-neo:128 \
  --runs 5 \
  --warmup 1 \
  --results-dir results/paper_eval
```

Artifact validation on a two-GPU server:

```bash
python3 scripts/repro_eval_latest.py \
  --tasks bert-tiny:128,bert-base:128,bert-large:128,gpt2:128,gpt-neo:128,\
llama7b:16,llama7b:32,llama7b:64,llama3-8b:16,llama3-8b:32,llama3-8b:64 \
  --runs 2 \
  --warmup 1 \
  --gpu0 0 \
  --gpu1 1 \
  --threads 96 \
  --auto-cpu-affinity \
  --assert-targets \
  --results-dir results/artifact_validation
```

The Llama extension target is:

The key-size column follows the Sigma artifact convention used in the raw logs:
the printed `GB` value is `bytes / 1024^3`. The raw byte counts are also
preserved in `results.json`.
The script reports `PASS_CLOSE` for Llama extension rows that are within the
small tolerance recorded in the JSON metadata (5% relative speedup tolerance and
2% relative key-size tolerance); raw values are not rounded before being written
to the result files.

| Model | Seq | Minimum speedup | Maximum FuseFSS key |
|---|---:|---:|---:|
| LLaMA-7B | 16 | 1.09x | 68.95 GB |
| LLaMA-7B | 32 | 1.17x | 90.17 GB |
| LLaMA-7B | 64 | 1.22x | 134.19 GB |
| LLaMA-3.1-8B | 16 | 1.05x | 83.09 GB |
| LLaMA-3.1-8B | 32 | 1.11x | 108.74 GB |
| LLaMA-3.1-8B | 64 | 1.17x | 160.04 GB |

Accepted aliases for `llama3-8b` include `llama8b`, `llama3.1-8b`, and
`llama-3.1-8b`.

Low-cost single-GPU sanity:

```bash
python3 scripts/repro_eval_latest.py \
  --tasks bert-tiny:128,gpt2:128 \
  --runs 2 \
  --warmup 1 \
  --single-gpu \
  --results-dir results/sanity_single_gpu
```

Strict generic sanity:

```bash
python3 scripts/repro_eval_latest.py \
  --tasks bert-tiny:128,gpt2:128 \
  --runs 2 \
  --warmup 1 \
  --single-gpu \
  --fusefss-sigma-generic-strict \
  --results-dir results/strict_generic_sanity
```

## Batch Scaling

```bash
python3 scripts/run_batch_scaling.py \
  --models bert-base,gpt2 \
  --seq 128 \
  --batches 1,2,4,8 \
  --out results/batch_scaling.json

python3 scripts/plot_batch_scaling.py \
  --inputs results/batch_scaling.json \
  --out-dir artifacts/batch_plots \
  --table-out artifacts/batch_scaling_tables.md
```

## Accuracy Emulation

```bash
python3 bench/accuracy_compare.py \
  --config bench/configs/accuracy_table4.json \
  --out-json results/accuracy_table4.json \
  --out-md artifacts/accuracy_table4.md
```

Accuracy emulation is cleartext fixed-point emulation. It checks numerical
impact and does not measure two-party runtime or communication.
