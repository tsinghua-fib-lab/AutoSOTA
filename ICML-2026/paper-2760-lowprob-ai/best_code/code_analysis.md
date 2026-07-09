# Code Analysis — Paper 2760: Uncertainty-AIGT

## Evaluation Path
- **Script:** `/repo/scripts/eval_target.py`
- **Flow:** loads XSum GPT-2 XL data → tokenizes 150 human/AI text pairs → runs GPT-J-6B proxy → extracts uncertainty features → computes fused score → AUROC
- **Metric parser:** Regex `AUROC: ([\\d.]+)%` from stdout

## Key Files
- `/repo/scripts/eval_target.py` — single-file evaluation; no separate training/inference split
- `/repo/dataset/xsum_gpt2_xl.raw_data.json` — 150 human/AI text pairs
- `/repo/Proxy_LLMs/gpt-j-6b` — proxy model (symlink to /models/gpt-j-6b-ms)

## Safe Modification Targets
1. **Temperature:** `log_softmax(chunk_logits / T)` at line ~60
2. **Feature extraction:** `extract_features()` — new features, weighting schemes
3. **Hyperparameter constants:** X_TAIL, RENYI_Q, WZ
4. **MAX_LENGTH, CHUNK_SIZE, MODEL_DTYPE** — configuration changes

## Risky Files (DO NOT MODIFY)
- `/repo/dataset/xsum_gpt2_xl.raw_data.json` — test data
- `roc_auc_score` call — metric computation
- `/tools/record_score.sh` — scoring infrastructure
- Labels (0=human, 1=AI) — evaluation protocol

## Baseline Config
- X_TAIL=7, RENYI_Q=2.0, WZ=0.8, n=150 samples, GPT-J-6B proxy
- AUROC=84.88% (matches paper Table 12)
- eval timeout: 30 min; typical runtime: ~5 min
