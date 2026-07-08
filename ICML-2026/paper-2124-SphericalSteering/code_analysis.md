# Code Analysis: Spherical Steering (Paper 2124)

## Evaluation Path

run_eval.sh executes four sequential steps:
1. get_activations.py — extracts last-token hidden states from specified layer for all TruthfulQA QA pairs
2. get_prototypes.py — computes antipodal (mu_H = -mu_T) prototypes via difference-of-means with 2-fold CV
3. evaluate_mc.py — MC evaluation (MC1/MC2/MC3) with spherical steering per fold
4. evaluate_llm_judge.py — generates answers with steering, scores via LLM judges (TRUE/INFO)

Final metrics are averaged across folds.

## Key Files

| File | Role | Safe to Modify |
|------|------|----------------|
| spherical_steering.py | Core steering algorithm | YES |
| get_prototypes.py | Prototype computation | YES |
| evaluate_mc.py | MC evaluation loop | YES (hook usage, not metrics) |
| evaluate_llm_judge.py | LLM judge evaluation | YES (generation, not judge) |
| utils.py | Data loading, activation extraction | YES |
| get_activations.py | Activation extraction entry point | YES |
| run_eval.sh | Orchestration script | YES |
| TruthfulQA/ | Third-party eval library | NO |

## Configuration

- Model: Qwen2.5-7B-Instruct at /models/Qwen2.5-7B-Instruct
- Default layer: 19 (out of 28)
- Default params: kappa=20, alpha=0.6, beta=0.4
- Fold splitting: 2-fold CV at question level (sequential, not random)
- Judge models at /models/truthfulqa-truth-judge-llama2-7B and /models/truthfulqa-info-judge-llama2-7B

## Metric Parsing

MC metrics from evaluate_mc.py stdout and saved JSON files.
LLM judge metrics (TRUE/INFO) from evaluate_llm_judge.py stdout and summary.csv.
Final TRUE_x_INFO = avg_true * avg_info across folds.

## Safe Modification Targets

1. spherical_steering.py — numerical precision, steering strength, multi-layer support
2. get_prototypes.py — prototype computation method
3. evaluate_mc.py — layer iteration, hook configuration
4. evaluate_llm_judge.py — token-position-dependent alpha
5. utils.py — activation normalization
6. run_eval.sh — parameter changes

## Key Observations

1. Prototypes are antipodal (mu_H = -mu_T) — vMF gate is effectively sigmoid
2. No multi-layer support exists
3. Batch processing is single-sample
4. 2-fold CV uses sequential split
5. Layer 19 chosen for Qwen based on Llama-3.1 analysis
