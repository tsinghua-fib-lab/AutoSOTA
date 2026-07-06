# Code Analysis - Paper 369: Profiling the Irrational Agent

## Evaluation Path
- Entry: eval.sh runs run_experiments.py then analyze.py
- Experiment script: scripts/experiment/run_experiments.py (batch executor)
- MAB core: src/mab/cigt_environment.py (state, rewards), src/mab/runner.py (prompt building)
- LLM client: src/mab/models.py (OpenAI-compatible wrapper with retries)
- Analysis: src/analysis/metrics.py (ASR, NTF, IAR computation)

## Config Path
- src/mab/config.py: SYSTEM_PROMPTS, FEEDBACK_STYLES, SCENARIO_REGISTRY
- configs/regret_experiment.json: Model config, dataset, trials, scenarios
- src/core/settings.py: API platform definitions, dotenv loading

## Metric Parser
- src/analysis/metrics.py MetricsCalculator
- SingleFileEvaluator computes per-instruction: NTF, ASR, Switch Rate, Recovery Rate
- BatchAggregator.generate_summary() aggregates: IAR, Mean_ASR, Avg_NTF
- IAR = Broken_Count / Total_Instructions
- Output: stdout table + CSV at metrics_summary_{model}.csv

## Safe Modification Targets
1. System prompts (SYSTEM_PROMPTS in src/mab/config.py)
2. Feedback styles (FEEDBACK_STYLES in src/mab/config.py)
3. Scenario probabilities (SCENARIO_REGISTRY probs)
4. Experiment config (configs/regret_experiment.json)
5. Prompt construction (runner.py build_prompt)
6. Option framing (runner.py _format_options_text)

## Risky Files (DO NOT MODIFY)
- src/analysis/metrics.py (metric definitions)
- eval.sh (evaluation protocol)
- Test dataset CSV
- /tools/record_score.sh

## Baseline
- IAR: 0.90, NTF: 6.2, Mean_ASR: 0.56
