# Code Analysis for Paper 3912 SOTA Optimization

## Evaluation Path
- **Entry point**: `/repo/code/eval_exchange_rate.py`
- **Data loading**: `load_data()` reads CSV files from `/repo/datasets/Exchange Rate/price/` and attribute data
- **Prediction**: `ese_predict()` per test point (130 test points, each with 100-step input window)
- **Metrics**: `compute_metrics()` computes RMSE, MAE, RMSE*, MAE* 
- **Output**: stdout prints metric table with per-currency breakdown

## Key Source Files
- `eval_exchange_rate.py` — standalone eval script (simplified from original codebase)
- `StateParameter.py` — `state_parameter_set()` computes state/sum(state)
- `PredictiorForPoint.py` — `ESE_predictor_system_ar()` with `select_order()` (unused in eval)
- `Cointegration.py` — `cointegration()` function
- `EquilibriumIndex.py` — `equilibrium_index_TED()`, etc.
- `LongRunTraining.py` — `long_run_equilibrium_l()` (NOT used in eval; eval uses simplified equilibrium)

## Config / Parameters
- Input window: 100 steps (hardcoded in eval_exchange_rate.py)
- AR order: fixed at AR(1) (hardcoded in eval_exchange_rate.py ese_predict())
- Horizon: 1 step
- Train/test split: 90:10
- Equilibrium: uses current state proportions directly (no iterative refinement)

## Metric Parser
- Parsed from stdout: lines matching Metric/Reproduced/Paper/Status table
- RMSE, MAE averaged across 16 currencies
- RMSE*, MAE* normalized by currency means * 100

## Safe Modification Targets
1. `ese_predict()` in eval_exchange_rate.py — AR order, equilibrium computation
2. `state_parameter_set()` in StateParameter.py — normalization approach
3. `compute_equilibrium_state()` in eval_exchange_rate.py — EWMA, ensemble
4. `input_steps` parameter in eval_exchange_rate.py main() — window size
5. `load_data()` — data preprocessing

## Risky Files (DO NOT MODIFY)
- Test data in /repo/datasets/Exchange Rate/
- `compute_metrics()` — metric definitions
- Train/test split logic
- Output format parsing

## Red Lines
- No changes to evaluation protocol, test data, splits, or metric formulas
- No hard-coded predictions or metric values
- All changes must be auditable via git commits
