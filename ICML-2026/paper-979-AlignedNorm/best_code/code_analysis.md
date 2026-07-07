# Code Analysis — AlignedNorm (Paper 979) SOTA Optimization

## Evaluation Path
- **Eval script**: /repo/scripts/eval_eurosat.sh
  - Trains 3 seeds (1,2,3) on base classes (B2N task)
  - Tests each seed on novel classes
  - Computes weighted average via compute_metrics.py
- **Metrics parser**: /repo/scripts/compute_metrics.py
  - Parses "* accuracy: XX.XX%" from log files
  - Computes HM = 2*Base*New/(Base+New) per seed, then averages
- **Time**: ~15 min per seed train, ~1 min test -> ~50 min total

## Training Path
- **Entry point**: train.py -> Dassl TrainerX framework
- **Trainer**: trainers/alignednorm.py — ALIGNEDNORM class (572 lines)
- **Config**: trainers/alignednorm_config.py — dataset-specific overrides
- **Train config**: configs/trainers/ALIGNEDNORM/vit_b16.yaml
- **Loss**: ALIGNEDNORM_Loss with 6 components (CE x2, self-consistency x2, norm alignment, layer-wise norm)
- **KEY FINDING (CODE-01)**: norm_ln_loss and stg_norm_att_loss computed but NOT added to loss_sum at line 428

## Model Architecture
- **Backbone**: CLIP ViT-B/16
- **Prompt tokens**: N_REP_TOKENS=5, REP_LAYERS=[6..12]
- **Trainable params**: representation_learner, proj_rep, A, B matrices
- **Attention hook**: need_weights=True at line 212 model_alignednorm.py

## Current Hyperparameters (EuroSAT B2N)
- ALPHA=0.7, BETA=0.2, FINAL_NORM=0.1, SEQ_NORM=0.05, REG_WEIGHT=0.01
- N_REP_TOKENS=5, REP_LAYERS=[6..12], MAX_EPOCH=10
- LR=1e-3, cosine scheduler, warmup_epoch=1, warmup_cons_lr=1e-5, AMP

## Seed Determinism (CODE-02)
- cudnn.benchmark = True (non-deterministic), cudnn.deterministic NOT set
- No worker_init_fn for dataloader

## Risky Files (DO NOT MODIFY)
- scripts/compute_metrics.py, scripts/eval_eurosat.sh

## Safe Modification Targets
- trainers/alignednorm.py: loss, gradient clipping, MixUp
- trainers/alignednorm_config.py: hyperparameters
- configs/trainers/ALIGNEDNORM/vit_b16.yaml: base config
- clip/model_alignednorm.py: attention, inference calibration
