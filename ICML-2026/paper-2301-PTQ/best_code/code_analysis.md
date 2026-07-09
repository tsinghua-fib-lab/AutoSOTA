# Code Analysis — SRR (Paper 2301)

## Evaluation Path
- Entry: `ptq_pipeline.py` (main CLI with argparse)
- Loads model via `transformers`, calibrates on SlimPajama-6B, evaluates on WikiText2
- Metric: perplexity from `evaluate_perplexity()` or `perplexity_results.yaml`

## Key Source Files
| File | Role | Safe to Modify? |
|------|------|-----------------|
| `src/qera/ptq_pipeline.py` | Main pipeline, Hessian/scale computation, Cholesky | Yes — add shrinkage before Cholesky, modify calibration |
| `src/qera/approximate.py` | Quantization + low-rank correction (compute_AB_and_approximation_error) | Yes — modify quantizer, add multi-round |
| `src/qera/approximate_with_init.py` | SRR init (srr_init, find_optimal_k) | Yes — MC variance reduction, rank allocation |
| `src/qera/quantize/quantizers/mxint.py` | MXINT quantizer | Yes — MSE-optimal range |
| `src/qera/statistic_profiler/scale.py` | Scale/Hessian hooks | Yes — streaming accumulation |
| `src/qera/datasets/slim_pajama.py` | Calibration data loading | Yes — diversity sampling |
| `src/qera/quantized_layers/linear.py` | Forward pass with quantized weights + LR correction | Yes — sparse correction |
| `src/qera/evaluate/evaluate_lm.py` | Perplexity evaluation | **NO** — metric definition |
| `experiments/configs/srr_3bit_rank32_repro.yaml` | Reproduction config | Yes — parameter tuning |

## Config Path
- `experiments/configs/srr_3bit_rank32_repro.yaml` — primary config
- Overridden via CLI: `--model-name`, `--perplexity-eval-batch-size`, `--max-position-embeddings`, `--perplexity-max-seq-length`, `--lr-scaling-mode`, `--num-calibration-samples`, `--srr-seed`, `--disable-lm-eval`, `-ow`

## Metric Parser
- Stdout line: `Perplexity after approximation: <float>` 
- OR: `checkpoints/ptq/srr/cholesky/_models_TinyLlama-1.1B/slim_pajama_6b_256/mxint_3/32_1/seed_42/perplexity_results.yaml`
- Mean over 3 seeds (42, 1234, 4321) for paper reporting

## Risky Files (DO NOT MODIFY)
- `src/qera/evaluate/` — metric computation
- `src/qera/datasets/wikitext2.py` — test data loading
- `/tools/record_score.sh` — score recording

## Safe Modification Targets
1. `get_precomputed_scale_dict()` in ptq_pipeline.py — add Ledoit-Wolf shrinkage
2. `find_optimal_k()` in approximate_with_init.py — MC variance reduction
3. `_mxint_quantizer()` in mxint.py — MSE-optimal clipping
4. `_compute_scales_and_error_for_fc()` in approximate.py — multi-round refinement
5. Config YAML — rank, iter, block_size parameter tuning
