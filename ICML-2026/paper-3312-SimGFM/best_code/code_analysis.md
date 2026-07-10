# SimGFM Code Analysis for SOTA Optimization

## Evaluation Path
- Entry: src/main.py via Hydra config
- Command: python -m src.main +experiment=qm9_no_h dataset=qm9 general.test_only=/repo/model_weights/qm9-qm9_no_h/epoch_249.ckpt general.num_folds=5
- Mode: test_only triggers sampling (not training), runs 5-fold sampling via _sample_and_log
- Sampling: sample_batch() builds Sampler, runs flow matching chain
- Metrics: compute_molecular_metrics() in src/analysis/rdkit_functions.py
- FCD: from src/metrics/molecular_metrics.py via fcd_torch

## Key Files
- src/main.py - entry point
- src/graph_discrete_flow_model.py - model, sampling, eval orchestration
- src/flow_matching/sampler.py - inference sampling loop
- src/flow_matching/rate_matrix.py - vf/rvf/defog rate matrix strategies
- src/flow_matching/kappa_scheduler.py - time scheduling
- src/models/transformer_model.py - GraphTransformer backbone
- src/analysis/rdkit_functions.py - molecular metric computation
- src/metrics/molecular_metrics.py - metric orchestration including FCD
- configs/experiment/qm9_no_h.yaml - experiment config
- configs/sample/sample_default.yaml - sampling defaults
- configs/train/train_default.yaml - training defaults

## Safe Modification Targets (Inference-Only)
1. src/flow_matching/sampler.py - sampling loop, self-conditioning, SID-style
2. src/flow_matching/rate_matrix.py - rate strategies, numerical stability
3. configs/experiment/qm9_no_h.yaml - config-level changes

## Risky Files (Do Not Modify)
- src/analysis/rdkit_functions.py - metric definitions
- src/metrics/molecular_metrics.py - metric orchestration
- configs/dataset/qm9.yaml - dataset splits
