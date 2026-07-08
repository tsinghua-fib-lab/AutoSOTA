# Code Analysis - Paper 1679 (SERPANT)

## Evaluation Path
- `run_reproduction.py` → `main.experiment_real_mode()` → `real_world/runner.run_real_mode_experiment()`
- Runner calls `core.algorithm.serpant_algorithm()` with `observation_fn` that wraps `env.compare()`
- Results saved to `/repo/reproduction_results/serpant_results_*.json`
- **Metric parsing**: `summary.discovered_partial_order_pairs` from serpant_results JSON

## Config Path
- Eval command: `python3 /repo/run_reproduction.py /repo/config_small.yaml`
- config_small.yaml: 3 models (Qwen2.5-1.5B, TinyLlama-1.1B-Chat, gpt2), tournament sampling, t=2000
- paper_config_triviaqa.yaml: 10 models, same settings

## Key Files
- `core/algorithm.py`: Main SERPANT algorithm (also contains `serpant_algorithm_covariate` - unused)
- `core/e_value.py`: E-value computation (`compute_e_value`, `compute_e_value_covariate`)
- `core/transitivity.py`: Transitivity propagation
- `real_world/runner.py`: Real experiment orchestration - calls `serpant_algorithm` (NOT covariate version)
- `real_world/environment.py`: LLMComparisonEnvironment - model loading, comparison, judge
- `real_world/model_clients.py`: HFModelClient - model loading with NaN fallback for multinomial
- `real_world/judge.py`: OpenAIJudge using DeepSeek API
- `real_world/config.py`: RealModeConfig dataclass
- `real_world/questions.py`: QuestionProvider from file/HF

## Metric Parser
From `runner.py:_save_serpant_results()`:
```python
discovered = len(serializable_results.get("rejected_hypotheses", []))
serializable_results["summary"]["discovered_partial_order_pairs"] = discovered
```
Parse from JSON: `summary.discovered_partial_order_pairs`

## Ceiling Analysis
- 3 models: max 3 pairs → baseline 3/3 = 100% AT CEILING
- Need more models to improve metric
- 5 models: max 10 pairs
- 10 models: max 45 pairs (paper target: 32)

## Cached Models
- Qwen/Qwen2.5-1.5B-Instruct ✓
- Qwen/Qwen2.5-3B-Instruct ✓ 
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 ✓
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B ✓
- openai-community/gpt2 ✓

## Safe Modification Targets
- `config_small.yaml`: Add more models (only change model list)
- `real_world/model_clients.py`: NaN attention fix, answer cache
- `real_world/judge.py`: Position swap, ensemble judge
- `real_world/runner.py`: Wire covariate mode, use covariate algorithm
- `core/algorithm.py`: UCB priority, adaptive exploration
- `core/e_value.py`: Hedged e-values

## Risky Files (avoid modifying)
- `core/transitivity.py`: Core theoretical guarantee
- `core/confidence_sets.py`: Core theoretical guarantee
- `run_reproduction.py`: Evaluation wrapper

## Red-Line Boundaries
- Do NOT change evaluation protocol, metric definition, data splits
- Do NOT hard-code results
- FWER must remain ≤ alpha=0.1
- Same TriviaQA questions, same evaluation command format
