# MOON Code Analysis for SOTA Optimization

## Evaluation Path
- **Script**: `/repo/run_moon_eval.py`
- **Entry**: `python3 run_moon_eval.py --seed 1 --n_tasks 1000 --batch_size 64 --num_class_eff_min 1 --num_class_eff_max 4 --device 0 --alpha 1.0 --lambda_laplacian 1.0 --lambda_y_hat 1.0 --n_neighbors 3 --max_iter 10`
- **Flow**: Load cached features → BatchSampler generates indices → MOON_solver per batch → accumulate accuracy
- **Output**: stdout grep "MOON Accuracy:" after "RESULTS" block

## Core Algorithm Path
- **Solver**: `/repo/solvers/MOON.py` → `MOON_solver()`
- **Sampler**: `/repo/sampler.py` → `BatchSampler` with Keff range [1,4]
- **Utils**: `/repo/utils.py` → `cls_acc()`

## Cached Data (read-only, do not modify)
- `/repo/caches/imagenet/test_f.pt`: 50000×512 float16 CLIP ViT-B/16 image features
- `/repo/caches/imagenet/test_l.pt`: 50000 int64 labels
- `/repo/caches/imagenet/clip_prototypes.pt`: 1×512×1000 float16 CLIP text prototypes

## MOON Algorithm Summary
1. Normalize query features and clip prototypes to unit hypersphere
2. Compute zero-shot logits: `100 * features @ prototypes` (temperature=100)
3. Build k-NN affinity graph (n_neighbors=3)
4. Initialize class weights: geometric mean of mean and max zero-shot confidence
5. EM iterations (max_iter=10):
   - **E-step**: Update z via vMF likelihood + prior + Laplacian, with temperature 1/50
   - **M-step**: Beta (shrinkage) → Mu (means) → Kappa (concentration)
   - Sample weights: entropy-based (1 - entropy/log(K))

## Hard-coded Constants (Tunable Targets)
| Constant | File:Line | Value | Role |
|----------|-----------|-------|------|
| Temperature in zero-shot logits | MOON.py:get_zero_shot_logits | 100 | Controls prior sharpness |
| Laplacian numerator | MOON.py:update_z | 50 | Laplacian scaling |
| Softmax temperature | MOON.py:update_z | 1/50 | E-step update temperature |
| Kappa clamp max | MOON.py:update_kappa | 500 | Max concentration |
| r_bar clamp max | MOON.py:Ad_inverse_approx | 0.999 | Max resultant length |
| Epsilon values | MOON.py (various) | 1e-6, 1e-10, 1e-12 | Numerical stability |

## Safe Modification Targets
- **MOON.py**: vMF adapter, kappa init/update, beta update, z update (algorithm parameters only)
- **run_moon_eval.py**: CLI args only (add new hyperparameters)
- **sampler.py**: Read-only, no changes needed

## Risky Files (Do Not Modify)
- `/repo/caches/imagenet/*` — cached features/labels (evaluation data)
- `/repo/utils.py` → `cls_acc()` — accuracy metric computation
- `/repo/sampler.py` → `BatchSampler.generate_indices()` — evaluation protocol

## Red Line Checks (per idea)
1. Eval command unchanged or justified extension
2. cls_acc() metric untouched
3. Test features/labels unchanged
4. No hard-coded outputs
5. Optimization objective respected
6. All metrics reported
7. Git rollback point exists

## Optimization Headroom
- Baseline: 82.83% Top-1 Accuracy
- Paper CI upper bound: 83.16%
- Maximum theoretical gain: ~0.33 percentage points
- Zero-shot baseline: ~66.48%
- Strongest non-MOON baseline: 79.2% (Dirichlet)

## Known Levers
alpha (1.0), lambda_laplacian (1.0), lambda_y_hat (1.0), n_neighbors (3), soft_beta (False), max_iter (10)
