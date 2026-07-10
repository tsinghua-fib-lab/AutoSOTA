# Code Analysis - Paper 2235: Online Adversarial Image Segmentation

## Repository Structure
- Main script: reproduce_online_seg.py
- Original notebook: Submodular_concave/Image_seg.ipynb

## Evaluation Path
- Entry point: python3 -u reproduce_online_seg.py --n_runs 10 --seed 42
- 10 independent trials, each: 10800 frames of 50x50 synthetic dumbbell video
- Metrics: IoU, Precision, Recall, F1 (averaged over frames 120-10799)
- Output format: lines matching Average (IoU—precision—recall—f1 score):\s+([\d.]+)

## Key Configuration Parameters
- h = 3e-2: step size (shared for w and y)
- mu = 1e-3: Gaussian smoothing parameter
- Y_sm = 10: smoothing samples for ZO gradient
- rho = 25.0: adversarial budget
- lam = 10.0: seed loss weight
- sigma_I = 20.0: bilateral filter intensity sigma
- sigma_x = 1.0: bilateral filter spatial sigma
- skip_frames = 120: warmup frames excluded from metrics
- n_seeds = 50: seeds per cluster (hardcoded)

## Algorithm Structure
1. transform_frame(): Generates synthetic 50x50 dumbbell image
2. build_graph(): 4-connected graph with bilateral edge weights
3. Algorithm 1 (ExtraGradient): Two half-steps per frame
4. lovasz_cut_subgradient(): Exact subgradient for submodular term
5. gaussian_smooth_y(): ZO gradient estimate (Y_sm samples)
6. proj_x() / proj_y(): Projection operators

## Safe Modification Targets
- Configuration parameters (h, mu, Y_sm, rho, lam, sigma_I, sigma_x)
- ExtraGradient step logic (momentum, adaptive step, multi-step)
- Graph construction (edge connectivity, epsilon stabilization)
- Post-processing threshold (tau)
- Gaussian smoothing loop (Y_sm count, schedule)

## Red-Line Constraints
- Do NOT change: metric computation (compute_iou, compute_prf)
- Do NOT change: ground truth generation (transform_frame)
- Do NOT change: evaluation protocol (n_runs, seed, skip_frames)
- Do NOT change: projection operators (proj_x, proj_y semantics)
- Do NOT change: lovasz objective/surrogate definition

## No External Data
- All data is synthetic (generated on-the-fly)
- No datasets, checkpoints, or pre-trained models needed
