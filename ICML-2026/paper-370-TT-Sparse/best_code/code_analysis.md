# Code Analysis: TT-Sparse (paper 370)

## Evaluation Path
- `eval_diabetes.py` — main evaluation script
  - Loads Pima Indians diabetes dataset (OpenML id=37)
  - 80-20 test split (seed=42)
  - 5 training seeds: [0, 1, 2, 3, 4]
  - For each seed: encode → train → prune → explain → predict → compute AUC & complexity
  - Outputs `outputs/metrics.json` and prints JSON to stdout

## Key Files
- `src/tt_sparse/model.py` — TTSparseModel, train(), prune()
- `src/tt_sparse/encoder.py` — TabularEncoder (thermometer encoding)
- `src/tt_sparse/rules.py` — RuleSet, explain(), predict_rules()
- `src/tt_sparse/qmc.py` — Quine-McCluskey logic minimization

## Config Path (in eval_diabetes.py)
- N_BITS=4, NUM_NODES=30, TAU=0.05
- EPOCHS=200, BATCH_SIZE=1024, LR=0.005, PATIENCE=25, VAL_SPLIT=0.2
- NUM_BITS_ENC=9, MAX_DROP=2.0, FT_EPOCHS=30, MAX_ITER=80, MAX_FANIN=16

## Metric Parser
- Metrics parsed from stdout JSON (also written to outputs/metrics.json)
- AUC: mean across 5 seeds, parsed from roc_auc_score
- Complexity: sum of rules + Boolean operators from RuleSet.complexity
- No modification to metric computation allowed

## Reusable Resources
- No /paper_data mounted
- Diabetes dataset cached at /root/scikit_learn_data/openml/
- pip cache at /autosota_cache/pip

## Safe Modification Targets
- `eval_diabetes.py`: parameter constants, training loop config, data features
- `src/tt_sparse/model.py`: train() (LR schedule, tau schedule, loss function), prune() (pruning schedule, saliency, edge selection)
- `src/tt_sparse/encoder.py`: num_bits, encoding strategy
- `src/tt_sparse/rules.py`: ensemble logic

## Risky Files (do NOT modify)
- `src/tt_sparse/model.py` metric computation (_eval_metric)
- sklearn metric functions (roc_auc_score)
- Test split logic
- `src/tt_sparse/qmc.py` (complexity computation logic)
