# Code Analysis for Paper 678 SOTA Optimization

## Evaluation Path
- Entry: `/repo/eval.py`
- Simulation: `decoupledmarket/src/decoupledmarket/main.py` → `overall_test()`
- Mock LLM: `eval.py` monkey-patches `decoupledmarket.content.gpt_structure._request_by_model`
- Metrics: Computed from SQLite DB (`save/sim01/data.db`) for Agent-class traders only (person_id range 13-27)
- Eval timeout: 20 minutes per run

## Metric Parser
- Regex in eval.py: `TR`, `Rp`, `sigma_p`/`σp`, `SR`, `WR`, `MD` from stdout
- DB-based: wealth time series → daily returns → TR, Rp, sigma_p, SR, WR, MD

## Safe Modification Targets
1. **eval.py — mock_request_by_model**: Replace pattern-based mock with market-data-informed signals
2. **constant.py**: Agent counts, market parameters, simulation days
3. **virtual_agent.py**: Trading probabilities, risk_appetite range
4. **content/our_prompt_template/*.txt**: Prompt templates for LLM agents
5. **our_run_gpt_prompt.py**: Prompt assembly, information integration
6. **behavior.py**: Reflection and agent behavior logic

## Risky Files (Do Not Modify)
- Database schema (database.py, database_utils.py)
- Metric computation logic in eval.py
- Stock initialization prices, DPS, quantities (paper Table 1)
- Market matching engine (Market.py)

## Red-Line Constraints
- Do not change eval command or metric computation
- Do not hard-code predictions or metric values
- Do not change test data, splits, or labels
- Keep changes auditable via git
