# Paper 2947 - SOTA Preparation Repair Analysis

## Original Failure

The orchestrator's preparation failed at the git initialization step:
1. `git` was not installed in the base Docker image
2. `apt-get install git` failed because the proxy returned HTTP 502 for `archive.ubuntu.com` URLs
3. The fallback container hit the same proxy issue

## Repair Applied

1. **Git installation**: Bypassed proxy: `apt-get install -y git -o Acquire::http::Proxy=false`
2. **Git repo initialization**: Initialized git, committed baseline, tagged `_baseline`
3. **Record score script**: Copied from host to `/tools/record_score.sh`

## Corrected In-Container Evaluation Command

```
cd /repo
MODEL_DIR=/models python3 main.py \
  --retrievers attention --generators attention \
  --retriever-models /models/bce-embedding-base_v1 \
  --generator-models /models/Qwen2.5-0.5B-Instruct \
  --datasets marco --triggers president \
  --results-dir /repo/results --seed 42
```

## Baseline Verification

- **E2E-ASR**: 79.61% (82/103 queries succeeded)
- **R-ASR**: 79.61%
- **G-ASR**: 79.61%
- **Manifest baseline**: 79.61% - exact match

## Container State

- Container: `autosota_sota_paper_2947` (from `autosota/paper-2947:reproduced`)
- GPU devices: 2,3
- Models: `/models/bce-embedding-base_v1`, `/models/Qwen2.5-0.5B-Instruct`, `/models/gpt2`
- Correlation data: `/repo/data/correlation/`
- Python: 3.10.13, Torch: 2.8.0+cu128

## Known Config Parameters (Levers)

- `RET_CORRELATION_THRESHOLD` (0.9) - Pearson correlation threshold
- `GEN_CORRELATION_THRESHOLD` (0.9) - same for generator
- `RET_NUM_EPOCHS` (50) - HotFlip optimization epochs
- `GEN_NUM_EPOCHS` (50) - generator epochs
- `RET_PREFIX_LEN`/`RET_SUFFIX_LEN` (10/10) - attractor token lengths
- `GEN_PREFIX_LEN`/`GEN_SUFFIX_LEN` (5/5) - generator attractor lengths
- `RET_PATIENCE`/`GEN_PATIENCE` (3) - early stopping patience
- `RETRIEVER_TOP_K` (5) - number of retrieved documents
- `NUM_TRAINING_PASSAGES` (1) - malicious documents
- `PREFIX_INIT_MODE` ("random") - token initialization
- `TRIGGER_RATIO_MIN`/`MAX` (0.005/0.01) - frequency range
- `ATTN_NUM_STEER_LAYERS` (6) - attention layers to steer

## Safe Optimization Targets

- `retriever.py:_hotflip_multi_head_retriever()` - core HotFlip optimization
- `retriever.py:_filter_heads_by_threshold()` - head selection
- `generator.py:_hotflip_one_step_multi()` - generator HotFlip
- `generator.py:_filter_heads_by_threshold()` - generator head selection
- `dataloader.py:Dataset.__init__()` - trigger word selection
- `config.py` - all parameter defaults

## Objective

Maximize E2E-ASR within 6-12 optimization iterations. Baseline: 79.61%.
