# Code Analysis — Paper 609: OASR Circuit Discovery

## Evaluation Path
- Script: eval_ioi_circuits.py
- Flow: Loads GPT-2 Small, loads IOI test data, evaluates circuits from circuits_discovered/oasr_ioi_circuits/
- Default circuits: low_iou_0.pt, low_iou_1.pt (pre-computed OASR circuits)
- Accepts optional path arg

## Training/Inference Path
- Algorithm: DiscoGP (circuit_discovery/algorithms/discogp.py)
- Config: circuit_discovery/configs.yaml -> notebook 01_oasr_alternative_sheaves
- Experiment runner: run_oasr_experiment.py
- Model: GPT-2 Small via circuit_discovery/models.py
- Losses: fidelity (good-vs-bad CE), completeness, edge density (sigmoid sum), overlap

## Key Optimization Levers
1. lambda_sparse_e and its schedule -> main sparsity control
2. lambda_overlap_e and its schedule -> sheaf diversity
3. gs_temp_edge -> mask sharpness
4. n_epochs_e -> convergence time

## Baseline Gap
- Paper OASR: edge_density=2.86%, edges=928.5, acc=99.59%
- Our baseline: edge_density=3.75%, edges=1217, acc=100%
- Room for ~0.9pp edge density and ~288 edges improvement via sparsity tuning

## Safe Modification Targets
- run_oasr_experiment.py: training hyperparams, overlap schedule
- discogp.py: algorithm internals (losses, schedules, mask init)
- New training scripts using existing DiscoGP/run APIs
- circuits_discovered/oasr_ioi_circuits/: replace with better-trained circuits

## Risky Files (do not modify)
- eval_ioi_circuits.py (evaluation script)
- circuit_discovery/metrics.py evaluate_good_bad_accuracy() (metric computation)
- circuit_discovery/datasets/ (dataset files)
- /tools/record_score.sh
