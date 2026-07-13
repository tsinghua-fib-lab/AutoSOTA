# Code Analysis — Paper 5224 (Exact GNN)

## Evaluation path
- `cd /repo/training_flooding_trees && bash run_eval.sh`
- Generates test samples + cases (deterministic), trains ensemble, reports metrics
- Final metrics parsed from `results/n{N}_D{D}_l{L}/trial_0/results.csv` last line
- Stdout emits `METRIC:case_accuracy=X.XXXX`

## Train/inference path
- `train_flooding.py:train_and_evaluate()` — main training loop
- `train_flooding.py:train_model()` — trains one NTKMLP for `epochs` epochs
- `model.py:NTKMLP` — 2-layer MLP (input_dim -> hidden_dim -> output_dim), NTK init
- `dataset.py:FloodingDataset` — one-hot encoded template vectors (13 samples)
- `dataset.py:load_test_samples/cases` — loads pre-generated JSON test data

## Config path
- CLI args in train_flooding.py: n, D, l, epochs, hidden_dim, sigma_w, sigma_b, max_models, delta, trials, batch_size, device, seed, lr (hardcoded 1e-3)
- run_eval.sh defaults: n=7, D=2, l=1, TRIALS=1, EPOCHS=7000, MAX_MODELS=300
- hidden_dim auto-computed: k*2000 = (l + 4*(D+1)*l)*2000 = 13*2000 = 26000

## Metric parser
- case_accuracy: fraction of test cases where ALL samples within the case are correctly classified
- sample_accuracy: fraction of individual test samples correctly classified
- ensemble_size: number of models trained before convergence (case_accuracy >= 1-delta)

## Key observations
- Baseline reaches case_accuracy=1.0 at model 51
- First 25 models contribute nearly zero improvement (stuck at 0.5)
- Models 50-51 provide the critical jump from 0.643 to 0.905 to 1.0
- 13 training samples, one-hot encoded, extremely small dataset
- float64 precision used everywhere (.double())
- DataLoader used for 13 samples — unnecessary overhead

## Risky files (do not modify)
- train_flooding.py:evaluate_cases() — metric computation
- dataset.py:load_test_cases/load_test_samples — test data loading
- test_samples/*.json — pre-generated deterministic test data

## Safe modification targets
- train_flooding.py:train_model() — optimizer, LR schedule, precision, early stopping
- train_flooding.py:train_and_evaluate() — aggregation method, ensemble selection
- model.py:NTKMLP — architecture, precision, initialization
- dataset.py:FloodingDataset — data loading optimization
- run_eval.sh — parameter defaults
