# Code Analysis — Paper 4707: Progressive Cramming

## Evaluation Path
- Script: `/repo/bench_v5.py` (extended from bench_v4.py)
- In-container eval: `source /autosota_cache/paper-4707-env/bin/activate && unset ALL_PROXY all_proxy && CUDA_VISIBLE_DEVICES=0 python3 /repo/bench_v5.py --max_samples 100`
- Output: stdout lines `Conv%=VALUE%` and `Acc (converged): N/M (VALUE%)`
- Also saved at `results.json` in output_dir

## Train/Inference Path
- ProgressiveCrammingTrainer in `src/progressive_cramming/train/trainers/progressive_cramming.py`
- BaseTrainer in `src/progressive_cramming/train/trainers/base.py` handles loss
- Loss in `src/progressive_cramming/train/loss.py` — CE, hybrid CE+alignment, leading token weighting

## Config Path
- MyTrainingArguments in `src/progressive_cramming/train/arguments.py`
- All flags exposed via bench_v5.py CLI args

## Metric Parser
- Parse stdout: Conv% and Acc (converged) lines
- Also from results.json: conv_pct and acc fields

## Safe Modification Targets
- bench_v5.py: CLI flags — parameter-only, no protocol change
- No source code changes needed for initial ALGO/CODE ideas (all parameterized)

## Risky Files
- loss.py, base.py, progressive_cramming.py — only modify for ALGO-06/07

## Evaluation Protocol
- Conv%: Fraction with final_convergence >= 1.0
- Acc: HellaSwag accuracy on converged subset with compression embedding prepended
- Model: pythia-1.4b, frozen. Dataset: HellaSwag validation (100 samples)
