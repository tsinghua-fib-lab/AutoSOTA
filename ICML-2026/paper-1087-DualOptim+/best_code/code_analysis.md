# SOTA Preparation Repair — Paper 1087 (DualOptim+)

## Original Failure

The preparation step failed because:
1. Docker overlay storage pool (200G) was full, preventing checkpoint saves.
2. The manifest eval_command uses save_checkpoint=false which prevents the HF Trainer from saving checkpoints needed by eval.py.
3. Git init failed after fresh container start due to residual disk space issues.

## Repair Actions

1. Freed disk space by removing 9 dangling Docker images (~37GB freed).
2. Fixed checkpoint saving: changed save_checkpoint=false to save_checkpoint=true. The HF Trainer now saves checkpoints at save_steps=last.
3. Implemented CODE-01: Fixed unused alpha parameter in custom_optimizer_plus.py.
   - Added alpha to adam_decouple_plus_step() function signature
   - Changed exp_avg computation to use alpha * state1_base_normalized
   - Applied alpha to denominator computation
   - Expanded alpha validation range from [0,1] to [0,2]
4. Reinitialized git repo with baseline commit and _baseline tag.
5. Installed /tools/record_score.sh for score recording.

## Corrected Evaluation Command

Working single-task command from /repo with save_checkpoint=true.

## Baseline Verification

Single-task baseline (task 1): UFE=70.14, TFE=64.01, MU=50.52, OVR=58.80.
Close to manifest 5-task baseline (UFE=67.04, TFE=63.86, MU=50.94, OVR=58.20).

## Optimization Targets

Safe parameter targets (all red-line checks passed):
- forget_coeff: forgetting aggressiveness (default 1.0)
- forget_lr: forget step learning rate (default 1e-5)
- alpha: base/delta mixing ratio (default 1.0, now functional via CODE-01)
- beta1/beta2: Adam betas for delta state (default 0.9/0.95)
- base_beta1/base_beta2: Adam betas for base state (default 0.9/0.95)
- retain_freq: retain step frequency (default 5)
