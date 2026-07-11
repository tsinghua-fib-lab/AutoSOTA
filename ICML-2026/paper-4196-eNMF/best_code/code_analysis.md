# Code Analysis: eNMF (paper-4196) SOTA Optimization

## Evaluation Path
- Eval script: /repo/eval_enmf_face.py
  - Phase 1: run_within_fixed_time(target_run_time=300) -> Reconstruction Error
  - Phase 2: run_to_target_error(target_error=recon_error) -> Runtime
- Output: JSON dict with Reconstruction Error (float) and Runtime (float) on stdout
- Timeout: 30 minutes (1800s)

## Core Algorithm Files
- src/nmf_algos/algorithms/NMF_ENMF.py — ENMF class with 4-phase pipeline
- src/nmf_algos/utils/ENMF_utils.py — Core helper: gen_svd_sol, admm_rotation, move_to_positive_orthant, HALS_pos
- src/nmf_algos/utils/algo_utils.py — Low-level: HALS_iter_solver, calculate_obj_NMF
- src/nmf_algos/NMF_base.py — Base class with set_params() and reset_status()

## Configuration Defaults
- rho=5, epsilon=1e-4, max_iter=4000, tau_inc=1.1, tau_dec=1.1, mu=2
- tol_asc=0.2, inner_iter_asc=2, num_steps=100
- hals_rounds=100

## Safe Modification Targets
1. NMF_ENMF.py enmf_config_init() — parameter changes
2. NMF_ENMF.py core_run() — profiling hooks
3. ENMF_utils.py gen_svd_sol() — initialization
4. ENMF_utils.py admm_rotation() — ADMM convergence
5. ENMF_utils.py move_to_positive_orthant() — PBCD scheduling
6. ENMF_utils.py HALS_pos() — early stopping, product caching
7. algo_utils.py HALS_iter_solver() — acceleration, sparse ops

## DO NOT MODIFY
- eval_enmf_face.py — evaluation protocol
- Dataset/face_id_4.npy — test data
- /tools/record_score.sh — scoring script
- calculate_obj_NMF() — metric computation
