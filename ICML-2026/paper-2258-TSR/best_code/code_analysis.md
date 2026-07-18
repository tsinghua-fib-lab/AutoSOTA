# Code Analysis — Paper 2258 (TSR: Temporal Score Rescaling)

## Evaluation Path
- **Main eval script:** `/repo/run_final.py` — runs DDIM baseline, then TSR (k=1.5, σ=1.0) with steps=10, seed=42, ensemble_size=1
- **Flexible eval script:** `/repo/eval_marigold_tsr_eth3d.py` — CLI args for k, sigma, steps, seed, no_tsr
- **Output:** stdout lines + `/repo/output/eth3d_eval/results.json`
- **Data:** 82 images (courtyard + delivery_area from ETH3D), list in `/repo/eth3d_filename_list_available.txt`
- **Data path:** `/datasets/marigold_eval/eth3d`

## Inference Path
- Pipeline: `MarigoldDepthPipeline` (diffusers) with model from `/models/marigold-v1-0` (fp16 variant)
- Scheduler: `DDIMScheduler` (baseline) or `TSR_DDIMScheduler` (TSR)
- Key hyperparameters: k (variance scaling), sigma (rescaling onset), num_inference_steps (DDIM steps), ensemble_size, processing_resolution, seed
- TSR modifies the epsilon prediction in the DDIM step by multiplying with `psr_ratio = (α·σ² + 1-α) / (α·σ²/k + 1-α)`

## Config Path
- No external config file; all hyperparameters are inline in `run_final.py`
- Model config loaded from `/models/marigold-v1-0/scheduler/scheduler_config.json`

## Metric Parser
- `compute_errors()` computes abs_rel and delta1 from aligned predictions vs ground truth
- Alignment: least_square fit (scale + shift), Marigold standard
- Primary metric: `AbsRel` = mean(|gt-pred|/gt) * 100, lower is better
- Guardrail: `delta1` = fraction with max(gt/pred, pred/gt) < 1.25, higher is better

## Reusable Resources
- Model weights: `/models/marigold-v1-0` (pre-downloaded, fp16 variant)
- Evaluation data: `/datasets/marigold_eval/eth3d` (82 images, 2 scenes)
- Cache: `/autosota_cache`, `/datasets`, `/models`

## Safe Modification Targets
1. **`run_final.py`** — Modify hyperparameters (k, sigma, steps, ensemble_size, processing_resolution, seed, timestep_spacing)
2. **`TSR_diffusers/TSR_schedulers.py`** — Modify `get_psr_ratio()` for adaptive k scheduling or spatial k modulation
3. **`eval_marigold_tsr_eth3d.py`** — For grid search and multi-config evaluation

## Risky Files (do not modify without strong justification)
- `run_final.py` metric computation functions (`compute_errors`, `align_depth_ls`, `load_depth_binary`)
- `/tools/record_score.sh` — scoring harness
- `/autosota_artifacts/paper-2258/sota/scores.jsonl` — scores database

## Red-Line Boundaries
- Do NOT change: metric definitions, test data/splits, alignment protocol, scoring scripts
- Do NOT hard-code metrics or predictions
- Do NOT modify `/models/marigold-v1-0` weights

## Optimization Strategy
Priority: CODE-01 (ensemble_size) → CODE-02 (trailing timestep) → ALGO-01 (adaptive k) → ALGO-02 (multi-k ensemble) → CODE-03 (resolution) → ALGO-06 (forward-backward) → PARAM-01 (grid search)
