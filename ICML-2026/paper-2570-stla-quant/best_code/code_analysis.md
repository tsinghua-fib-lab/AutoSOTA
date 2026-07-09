# STLA Code Analysis for SOTA Optimization (Paper 2570)

## Evaluation Path
- **Entry**: `main.py` — parses args, loads model, calibrates, quantizes, evaluates, saves results
- **Quantization**: `main.py` line 27 → `aespa_fwrd()` in `quantize.py` line 12
- **Per-block**: `quantize.py` lines 39-60 — for each transformer block: compute Hessian → quantize each linear layer → update inputs
- **Adaround**: `aespa.py` line 723 — `adaround()` method with Adam (line 739), CosineAnnealingLR (line 740), RoundLoss (line 863)
- **GPTAQ**: `aespa.py` line 433 — compensation with `alpha=0.25`, Spin order
- **Evaluation**: `eval_utils.py` line 7 — evaluates wikitext2 and c4-new perplexity

## Key Configuration
- **Default eval command** (baseline):
  ```
  python main.py --model_path /models/opt-125m --calib_data c4 --nsamples 128 --seqlen 2048 --seed 0 \
    --w_bits 3 --groupsize 256 --blocksize 256 --clustersize 256 \
    --loss_option global --order_option spin --comp_method GPTAQ \
    --learn_rounding --num_iters 200 --lr 1.1 --round_weight 1.0 --block_v \
    --cache_dir /autosota_cache/stla_cache
  ```
- **Paper defaults** (utils.py):
  - `--lr 0.015` (repro uses 1.1 — 73x higher)
  - `--num_iters 2000` (repro uses 200)
  - `--round_weight 1.0`, `--round_weight_qkv 1.5`

## Metric Parsing
- **Output**: stdout lines like `wikitext2 : 31.606` and `c4-new : 28.028`
- **CSV**: `quantization_results.csv` columns: `wikitext2`, `c4-new`, `process_time`

## Key Files and Safe Modification Targets

| File | Lines | Function | Safe Modifications |
|------|-------|----------|-------------------|
| `aespa.py` | 739 | `torch.optim.Adam([sb], lr=lr)` | Replace with Adamax (ALGO-01) |
| `aespa.py` | 740 | `CosineAnnealingLR(optimizer, T_max=num_iters, eta_min=lr*0.3)` | Replace with WarmRestarts (ALGO-06) |
| `aespa.py` | 433 | `alpha = 0.25` | Tune per layer type (ALGO-05) |
| `aespa.py` | 742 | `RoundLoss(max_count=num_iters, b_range=(20, 2), ...)` | Change b_range (CODE-02) |
| `quant_utils.py` | 39 | `idx_dead = (diagonals == 0)` | Change to threshold (CODE-04) |
| `quant_utils.py` | 156-222 | `refine_qparams_zfold` | Already implemented, needs `--use_zfold` flag (ALGO-02) |
| `data_utils.py` | 129-142 | `get_c4()` calib data | Multi-scale sequences (ALGO-04) |
| `utils.py` | 45-55 | Default args | `--num_iters` increase (ALGO-03) |
| `main.py` | 12+ | Main script | Multi-seed evaluation (CODE-03) |

## Risky Files (DO NOT MODIFY)
- `eval_utils.py` — evaluation protocol, metric computation
- `data_utils.py` — test data loading (`get_testdata`)
- `main.py` — metric output format (stdout lines)
- Test data, labels, splits, scoring scripts

## Available caches
- `/autosota_cache/stla_cache/calib_opt_c4_128_2048_0.cache` — calibration data
- `/autosota_cache/stla_cache/testloader_opt_wikitext2.cache` — WikiText-2 test data
- `/autosota_cache/stla_cache/testloader_opt_c4-new.cache` — C4 test data

## GPU
- CUDA_VISIBLE_DEVICES=0 used for evaluation
- OPT-125M is small (~125M params), fits in <2GB GPU memory
