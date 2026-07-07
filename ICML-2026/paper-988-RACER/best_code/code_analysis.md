# RACER Code Analysis

## Evaluation pipeline
- Entry: `main.py` → `run_repeated_racer_evaluation()`
- Router built via `routers/factory.py` → `MLPModule`
- Data loaded by `MLPModule.load_datasets()` using `RouterMLPDataset`
- Router trained in `MLPModule.fit()` 
- Calibration: `RACER_Module.calibrate()` on cal split
- Temperature tuning on held-out split
- Evaluation: `evaluate_racer()` on test split
- Metrics: risk, coverage, avg_set_size, base_router_accuracy, aggregation accuracy

## Training path
- `routers/mlp_router.py::MLPModule.fit()` - trains MLPClassifier
- Collects embeddings and labels from dataset
- 2-layer MLP: 768→256→7 (input_dim, hidden_size, num_models)
- BCEWithLogitsLoss, AdamW(lr=1e-4, weight_decay=0.01), 100 epochs, batch_size=32

## Inference path
- `MLPModule.forward()` - gets logits from MLP classifier
- `load_dataset_scores_and_labels()` - extracts probs (with softmax) and labels
- RACER calibration, prediction set construction, aggregation

## Config path
- All hyperparameters via argparse in `main.py`
- Router-specific params passed to `MLPModule.__init__`

## Metric parser
- Output JSON: `results/repro/gsm8k_repeated_racer_results.json`
- Primary metric: `summary.mean_racer_agg_acc_weighted_p_true`
- Also available: risk, coverage, avg_set_size, base_router_accuracy

## Reusable resources
- `/models/mdeberta-v3-base` - DeBERTa backbone
- `data/*.json` - pre-computed LLM outputs with probabilities and labels

## Risky files
- `main.py` - evaluation pipeline, metric computation
- `RACER.py` - calibration, aggregation, metric computation
- `routers/mlp_router.py` - router architecture and training

## Safe modification targets
- `routers/mlp_router.py::MLPClassifier` - router architecture
- `routers/mlp_router.py::MLPModule.fit()` - training loop, loss function, optimizer
- `routers/mlp_router.py::RouterMLPDataset` - data preprocessing, soft labels
- `RACER.py::calibrate()` - calibration mechanism
- `RACER.py::evaluate_racer()` - aggregation weights
- `main.py` - temperature tuning, hyperparameter defaults

## Red-line constraints
- Do not modify test data, labels, evaluation protocol
- Do not change metric computation in RACER.py evaluate_racer
- Do not hard-code predictions or dataset-specific shortcuts
