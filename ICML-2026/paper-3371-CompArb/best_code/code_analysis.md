# Code Analysis — Paper 3371: Computational Arbitrage in AI Model Markets

## Evaluation Path
- Entry point: eval_arbitrage.py
- Core library: notebooks/utils.py
- Data: data/swebench/*.jsonl (pre-computed SWE-bench pass/fail records)
- Inline eval: python3 eval_arbitrage.py

## Architecture
1. Load per-problem pass/fail data from JSONL (attempts list + mean_cost per model)
2. Compute per-problem pass-vs-budget curves via pass_at_k estimator
3. Grid search optimal tau (threshold) allocation across models
4. Compose cascade pass curve via cascade() (sequential threshold gating)
5. Compute cost-for-pass at target solve rates via survival integration
6. Report Cost at 75% solve rate and max Profit Margin

## Key Configuration Parameters (eval_arbitrage.py)
| Parameter | Current Value | Line |
|---|---|---|
| MODELS | [gpt5mini, deepseek] | 35 |
| BUDGET_POINTS | 250 | 39 |
| PERFORMANCE_POINTS | 1000 | 41 |
| COMPUTE_SEARCH_POINTS | 100 | 42 |
| SCALE_FACTOR | 500 | 43 |
| PERFORMANCE_RANGE | (0.68, None) | 40 |
| BUDGET_RANGE | (1e-4, 1.0) | 37-38 |

## Metric Parser
- Parses JSON_METRICS: line from stdout
- Cost: metrics_json cost_at_75 arbitrageur (primary, lower is better)
- Profit Margin: metrics_json profit_margin max_pct (guardrail, higher is better)

## Available Model Data
| Model | Problems | Avg Cost/Attempt | Pass Rate | File |
|---|---|---|---|---|
| deepseek | 445 | 0.0428 | 18.1pct | deepseek.jsonl |
| gpt5mini | 445 | 0.0219 | 10.5pct | gpt5mini.jsonl |
| sonnet | 445 | 1.9004 | 56.8pct | sonnet.jsonl |
| qwen3-coder-30b | 446 | 0.0061 | 11.0pct | qwen3-coder-30b.jsonl |
| qwen3-coder-480b | 445 | 0.0364 | 42.5pct | qwen3-coder-480b.jsonl |
| mini-coder-4b | 445 | 0.0044 | 7.9pct | mini-coder-4b.jsonl |
| minicoder4b | 445 | 0.0027 | 7.9pct | minicoder4b.jsonl |

## Key Optimization Levers
1. Cascade model selection and ordering: mini-coder-4b is 5x cheaper than gpt5mini
2. Tau threshold grid search resolution and distribution
3. Budget and performance grid resolution
4. Performance range lower bound
