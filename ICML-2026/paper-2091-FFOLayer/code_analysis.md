# Code Analysis - Paper 2091 (FFOLayer)

## Evaluation Path
- synthetic_task/main_synthetic.py - main training script
- Eval: python3 synthetic_task/main_synthetic.py --method ffocp_eq --ydim 800 --epochs 1 --batch_size 8 --seed 0 --device cpu --backward_eps 1e-6
- Output CSV: synthetic_results_8/ffocp_eq/ffocp_eq_ydim800_lr0.001_seed0_backwardTol1e-06.csv
- Training Loss = col5 (test_df_loss), Total Time = col6+col7 (forward_time + backward_time)

## Key Source Files
- synthetic_task/main_synthetic.py - training loop, loss computation
- synthetic_task/models.py - MLP predictor (2 hidden layers of 128)
- synthetic_task/data.py - synthetic data generation (2048 samples, 80/20 split)
- src/ffolayer/ffocp_eq.py (1528 lines) - FFOLayer implementation with SCS solver

## Solver Configuration (ffocp_eq.py)
- Line 726: Forward solver warm_start=False
- Line 747: Backward solver warm_start=True (already enabled)
- Lines 842-952: Direct SCS warm-start path for forward solves (exists, disabled)
- Lines 1286-1363: Direct SCS warm-start path for backward solves

## Safe Modification Targets
1. Loss weights: ts_weight, norm_weight
2. Solver warm_start for forward
3. MLP hidden sizes
4. Output clamp bounds
5. LR schedule
6. Two-phase training (ts_weight schedule)
