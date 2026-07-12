# Code Analysis - Paper 4416 RegimeFlow

## Evaluation Path
- Entry: `exp/train_bioTFM.py` with `test_only=true`
- Creates model from config via `instantiate(cfg.model)` then loads checkpoint
- Metrics: `test/mse`, `test/mae`, `test/CRPS` from PyTorch Lightning logs
- Results saved to `<save_path_dir>/checkpoints/<experiment_name>/<seed>/final_test_results_only_test.csv`

## Config Path
- Main: `exp/configs/config.yaml`
- Model: `exp/configs/model/ProbabilityForecasting/RegimeFlow.yaml`
- Hydra overrides via command line

## Model Architecture
- `models/FlowMatching/RegimeFlow/RegimeFlow.py::RegimeFlowCond` - main model
- `models/FlowMatching/RegimeFlow/arch/_base.py::RegimeFlowBase` - base flow matching
- `models/FlowMatching/RegimeFlow/arch/backbone.py::BackboneModel` - Mamba backbone
- `models/FlowMatching/RegimeFlow/arch/source_BLR.py` - BLR prior
- `models/FlowMatching/RegimeFlow/arch/bio_cond_layers.py` - Condition encoder

## Key Findings
1. `guidance_scale = 0` hardcoded (line 81 RegimeFlow.py) - CFG disabled
2. `num_steps: 1` in config, default 16 in code - likely suboptimal
3. `num_harmonics: 1` in config, default 4 in code - likely suboptimal
4. `configure_optimizers()` returns only optimizer, no LR scheduler
5. `warmup_epochs: 50` exists in config but unused
6. `p_losses` uses uniform t sampling (line 393)
7. EMA checkpoint `last_ema.ckpt` exists and is preferred for eval
8. Baseline checkpoint: epoch 39 (early stopped), seed 53

## Inference-only changes (no retraining)
- I-02: num_steps (add `model.num_steps=N` to eval cmd)
- I-06: CFG guidance_scale (needs code change + config param)

## Training changes (require retraining)
- I-01: LR scheduler
- I-03: num_harmonics
- I-05: Beta timestep distribution
- I-08: Pattern-based sampling
- I-09: Model capacity
