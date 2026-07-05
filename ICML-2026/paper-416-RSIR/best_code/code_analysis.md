# Code Analysis: Paper 416 - RSIR + SASRec

## Evaluation Path
- Entry: run.py -> quickstart.run_recommender() -> BaseModel.fit()
- Evaluation: BaseModel.test_epoch() -> evaluation.get_eval_metrics()
- Metrics: ndcg, recall, precision, f1, mrr in evaluation/__init__.py
- Early stopping: utils/callbacks.py:EarlyStopping monitors ndcg@20, patience=20

## Config Paths
- Main: configs/basemodel.yaml (train, model, eval, generation)
- Model: configs/sasrec.yaml (hidden_size, head_num, dropout_rate, activation)
- CLI overrides: utils/arguments.py

## Key Files
- model/basemodel.py: BaseModel class, fit_loop, training_step, _neg_sampling
- model/sasrec.py: SASRec transformer encoder, alignment/uniformity
- model/loss_func.py: SampledSoftmaxLoss, BinaryCrossEntropyLoss, GeneralizedBCELoss
- utils/utils.py: seed_everything, anomaly detection flag (line 11)
- utils/callbacks.py: EarlyStopping, Analyzer

## Critical Finding: Anomaly Detection
- utils/utils.py line 11: set_detect_anomaly(True) MUST stay enabled
- Disabling causes ~2% regression (ndcg@10 drops 0.0293 -> 0.0286)
- Affects CUDA non-determinism even with fixed seed

## Data
- amazon-sport: ~18K items, ~300K interactions
- Pre-generated RSIR: train.pth through train_8th.pth
- Baseline trainfile: _1th

## Safe Modifications
- Gradient clipping in training loop (max_norm=1.0)
- neg_num increase (256->512 optimal, +1.0% ndcg@10)
- learning_rate, weight_decay config tuning
- _neg_sampling target exclusion

## Risky Modifications
- Disabling anomaly detection (regression)
- Architecture scaling (overfitting on small dataset)
- gBCE loss (major regression, beta=0.75 gave -31%)
