# RDBLearn Code Analysis for SOTA Optimization

## Evaluation Path
- Entry: `evaluate_cvr.py` → `evaluate(depth, seed)`
- Dataset: RDBDataset.from_4dbinfer("retailrocket"), task="cvr"
- Model: TabPFNClassifier (v2, finetuned checkpoint) wrapped in RDBLearnClassifier
- Metric: roc_auc_score (sklearn) → JSON stdout: {"auc": float, "depth": int, "seed": int}

## Train/Inference Path
1. Load dataset → RDBDataset.from_4dbinfer("retailrocket")
2. Initialize TabPFNClassifier with config → base_model
3. Wrap in RDBLearnClassifier(base_estimator, config)
4. clf.fit(X_train, y_train, rdb, key_mappings, cutoff_time_column)
   - Inside fit: _downsample() → _prepare_rdb() → compute_dfs_features() → TabularPreprocessor → base_estimator.fit()
5. clf.predict_proba(X_test) → ROC AUC

## Config Path
- `rdblearn/constants.py`: TABPFN_DEFAULT_CONFIG, RDBLEARN_DEFAULT_CONFIG
- `rdblearn/config.py`: RDBLearnConfig (pydantic model)
- `evaluate_cvr.py`: overrides model_path, max_depth, max_train_samples

## Metric Parser
- Parses JSON from stdout: {"auc": <float>, "depth": <int>, "seed": <int>}

## Reusable Resources
- /models/tabpfn-v2-clf/: 16 TabPFN v2 checkpoints (various finetuned and base variants)
- /datasets/: fastdfs-cached Retailrocket data

## Risky Files (DO NOT MODIFY)
- Test data loading: `rdblearn/datasets.py` (RDBDataset.from_4dbinfer)
- Metric computation: sklearn.metrics.roc_auc_score
- Evaluation protocol: dataset splits, labels, task definitions

## Safe Modification Targets
- `evaluate_cvr.py`: Config overrides (max_train_samples, n_estimators, model_path, sampling params)
- `rdblearn/constants.py`: Default config values
- `rdblearn/estimator.py`: _downsample() method for sampling strategies
- `rdblearn/config.py`: Add new config fields if needed

## Key Levers (from manifest + code analysis)
1. max_train_samples: 10000 → 20000+ (TabPFN with ignore_pretraining_limits=True)
2. n_estimators: 8 → 16/32 (ensemble size)
3. dfs.max_depth: 2 → 3 (deeper feature traversal)
4. stratified_sampling: False → True (class-balanced downsampling)
5. model_path: different finetuned checkpoints available
6. enable_target_augmentation: False → True
7. predict_batch_size: 5000 (GPU memory tradeoff)
8. dfs.agg_primitives: add var, skew

