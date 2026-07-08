# Code Analysis — Paper 1982 SDR Optimization

## Evaluation Path
- run_eval.sh -> utils/get_slm_embeddings.py (pre-compute hidden states) -> sdr/train.py (train + test)
- AUC parsed from stdout: "auc: <float>" after "--- Starting Testing ---"
- Test data: data/qwen-7b/mmlu-test.parquet (SLM answers) + data/gpt-4o/mmlu-test.parquet (LLM answers)
- Train/test split: 80/20 within the single MMLU file, shuffle with seed=42

## Training Path
- sdr/train.py:main() -> parse_args -> load_model_and_tokenizer -> get_dataset -> CustomTrainer.train() -> do_test()
- Loss: compute_loss() in CustomTrainer — pointwise MSE between tanh(score) and advantage
- Advantage label: remote_metric - local_metric (0/1 correctness -> {-1, 0, 1})
- Optimizer: AdamW via HuggingFace Trainer (default weight_decay=0)
- Scheduler: cosine with warmup_steps=100

## Inference Path
- sdr/train_utils.py:do_test() -> DataLoader -> model(input_ids, attention_mask, index) -> score = scorer(hidden_states)
- Score applied through tanh in compute_loss(), but in do_test() the raw score is stored and np.tanh() is applied in result
- AUC computation: sort advantages by descending scores, cumulative sum, normalized

## Model Architecture
- sdr/model.py:LLMWithScorer — loads base model, extracts embeddings from layer -2 (penultimate)
- With embed_path: loads pre-computed embeddings -> scorer(hidden_states)
- GeneralHead (multi_remote_strategy=head): MLP 3584->1024->512->256->1, GELU+Dropout(0.1)
- BUG in else branch: nn.Linear(input_dim, 256) followed by nn.Linear(1024, 512) — dimension mismatch

## Config Path
- All args via argparse in sdr/train_utils.py:parse_args()
- Defaults: lr=5e-5, epochs=2, batch=1, grad_accum=2
- run_eval.sh overrides: lr=1e-4, epochs=3, batch=32, grad_accum=1

## Metric Parser
- sdr/train_utils.py:compute_metrics(): returns auc_norm
- do_test(): computes advantage, score arrays -> auc_norm
- No AUC printed during training eval (eval_steps=100); only final auc: in stdout

## Reusable Resources
- Pre-computed embeddings: /repo/output/qwen_mmlu_hidden.pt (penultimate layer, last-token-only)
- Pre-computed answers: data/qwen-7b/mmlu-test.parquet, data/gpt-4o/mmlu-test.parquet
- Model: /models/Qwen2.5-7B-Instruct
- No /paper_data mount available

## Safe Modification Targets
- sdr/model.py: GeneralHead architecture (BatchNorm, dimension fixes)
- sdr/train.py: loss function, compute_loss, CustomTrainer
- sdr/train_utils.py: training args, do_test, compute_metrics (not the metric logic itself)
- run_eval.sh: hyperparameters, flags
- utils/get_slm_embeddings.py: layer selection for multi-layer extraction

## Risky Files (DO NOT MODIFY)
- Metric computation: compute_metrics() in train_utils.py (AUC formula)
- Data loading: get_dataset() and preprocess functions (train/test split, advantages)
- Test data: parquet files in data/
- record_score.sh tool
- scores.jsonl
