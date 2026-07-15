# Code Analysis: RobOP-CAP (Paper 5797)

## Evaluation Path
- robop_cap.py main() -> load model -> eval dense -> load calibration -> compute Fisher -> prune -> eval pruned
- Output: JSON files in --output_dir with pruned_accuracy, dense_accuracy, actual_sparsity
- Metric parsing: pruned_accuracy from JSON output is the primary metric

## Key Files
- robop_cap.py — Main entry point with Fisher estimation, RobOP regularization, CAP pruning, evaluation
- pruning_handle_standalone.py — CAP OBS pruning algorithm (CAPHandle class). Imported via exec()

## Config Path
- Argparse in robop_cap.py main() (lines 336-350)
- Key params: model, sparsity, uncertainty_set, gamma, num_grads, fisher_block_size, damp, seed

## Key Components
1. EmpiricalBlockFisherInverse (lines 108-195): Block-diagonal Fisher inverse with RobOP regularization
   - add_grad(): Updates F_inv with per-sample gradient via rank-1 update
   - apply_robop_regularization(): Applies uncertainty set regularization
2. compute_fisher_inverse() (lines 202-280): Collects per-sample gradients, builds Fisher inverse
3. prune_model_cap() (lines 282-312): Applies CAP OBS pruning using Fisher inverse
4. CAP targets: attn.qkv, attn.proj, mlp.fc1, mlp.fc2 weights only

## Known Issues
- Actual sparsity ~55.7% vs target 60% (CAP algorithm property)
- Dense model accuracy 71.53% vs expected 72.2% (validation set differences)

## Safe Modification Targets
- apply_robop_regularization(): Add new uncertainty set types, per-block gamma scaling
- compute_fisher_inverse(): Modify Fisher estimation strategy, add multi-seed averaging
- EmpiricalBlockFisherInverse.__init__(): Accept new parameters
- prune_model_cap(): Add per-layer sparsity allocation
- main(): Add new CLI args, modify pipeline flow

## Risky Files (do not modify)
- evaluate() — evaluation protocol
- load_imagenet_val(), load_imagenet_calibration() — data split/transform logic
- CAPHandle in pruning_handle_standalone.py — core OBS algorithm

## Baseline Config
python3 robop_cap.py --model deit_tiny_patch16_224 --sparsity 0.6 --uncertainty_set trace --gamma 0.005 --num_grads 4096 --fisher_block_size 192 --damp 1e-8 --seed 0 --batch_size 64 --val_batch_size 128 --workers 4 --data_dir /datasets/imagenet1k --output_dir /repo/results_robop_cap
