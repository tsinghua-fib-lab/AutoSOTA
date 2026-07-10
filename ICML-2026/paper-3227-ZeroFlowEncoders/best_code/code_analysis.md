# Code Analysis for Paper 3227 (Zero-Flow Encoders) SOTA Optimization

## Evaluation Path
- **Script**: `/repo/reproduce_nonparanormal_mlp.py`
- **Command**: `CUDA_VISIBLE_DEVICES=0 python3 reproduce_nonparanormal_mlp.py`
- **Output**: stdout line `RESULTS: Nonparanormal MLP AUC = X.XXXX +/- Y.YYYY`
- **JSON output**: `/repo/data/reproduction_nonparanormal_mlp.json` key `mlp_auc_mean`
- **Per-seed AUC**: key `mlp_auc_individual`
- **Timeout**: 15 minutes

## Train/Inference Path
- **Training loop**: `BaseExperiment.py:67-101` (train method)
- **Forward step**: `reproduce_nonparanormal_mlp.py:56-84` (_forward_step)
- **Encoder**: `models/nntoy.py:131-157` (Encoder class)
- **Vector Field**: `models/nntoy.py:160-178` (VectorField class)
- **Dataset loader**: `datasets/ToyChainNonpra.py` (ToyNonParanormalLoader)

## Config Path
- Config dict at top of `reproduce_nonparanormal_mlp.py:132-136`:
  - batch_size: 400
  - lr: 1e-3 (paper says 1e-4 in Appendix D)
  - l1_lambda: 1e-9 (paper says 3e-9 in Appendix D)
- Zero-flow penalty weight: 0.1 (hardcoded in `_forward_step:83`: `1e-1 * penalty`)
- Time sampling: uniform t ~ U(0,1) in `_forward_step:61`
- Training iterations: 5000
- Seeds: 10

## Metric Parser
- Mean AUC parsed from stdout: `RESULTS: Nonparanormal MLP AUC = X.XXXX +/- Y.YYYY`
- Also available from JSON file: `data/reproduction_nonparanormal_mlp.json`
- Individual seed AUCs printed per seed

## Safe Modification Targets
1. **`models/nntoy.py:131-157`** (Encoder class): Safe to modify MLP architecture (gating network)
2. **`reproduce_nonparanormal_mlp.py:56-84`** (_forward_step): Safe to modify loss computation, time sampling, penalty formulation
3. **`reproduce_nonparanormal_mlp.py:132-136`** (config): Safe to modify hyperparameters
4. **`BaseExperiment.py`**: Safe to modify training loop (but use caution - shared across experiments)
5. **`reproduce_nonparanormal_mlp.py:156-195`** (evaluation): DO NOT modify metric computation

## Risky Files (DO NOT MODIFY)
1. **`utils/roc.py`**: ROC curve computation - metric definition
2. **`datasets/ToyChainNonpra.py`**: Data generation and mask creation - test data/splits
3. **`data/glasso_loader_Theta_true.npy`**: Ground truth precision matrix - test labels
4. **Metric parsing in `main()`**: AUC computation from all_gates vs true_prec

## Red-line Boundaries
- Do NOT change `compute_roc_curve`, `auc_trapezoid`, or the ground truth comparison
- Do NOT modify `ToyNonParanormalLoader` data generation
- Do NOT hardcode gate values or AUC results
- Do NOT change the number of seeds or evaluation protocol
