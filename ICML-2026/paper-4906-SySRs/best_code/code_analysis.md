# Code Analysis — Paper 4906: SySRs (Smart Successive Rejects)

## Evaluation Path
- **Entry**: `python3 eval_sysrs_12pct.py`
- **Flow**: Loads pickle files from `datasets/<name>/model_accuracies_filtered.pkl`, sets `bai_algs.model_accuracies`, calls `smart_successive_rejects_wo_replacement_no_budget_limit(n_items=budget, k=1000)`, reports per-dataset and average accuracy
- **Budget**: `max(n_models, (int(0.12 * n_models * n_tasks) // n_models) * n_models)`
- **Metric**: `float((np.asarray(arms) == 0).mean()) * 100` → percentage of 1000 runs that correctly identify arm 0 as best

## Core Algorithm File
- **`bai_algs.py`** (830 lines): Contains all BAI algorithms
  - Utility functions: `get_results()`, `random_argmax()`, `acc()`, `calculate_budgets_from_percentage()`, `cap_and_adjust_budgets()`
  - Budget reallocation: `_reallocate_budget_across_rounds()` (lines 474–623)
  - Baselines: `naive_baseline()` (lines 140–180), `uniform_pulls()` (lines 183–221)
  - UCB-E: `ucb_e()` (lines 228–353), `smart_ucb_e()` (lines 356–467)
  - SR: `successive_rejects_wo_replacement_no_budget_limit()` (lines 630–734)
  - **Smart SR** (target): `smart_successive_rejects_wo_replacement_no_budget_limit()` (lines 737–829)

## Key Code Locations for Optimization
| Location | Issue | Idea |
|----------|-------|------|
| Line 800-808 | Per-run Python for-loop over k runs | IDEA-08: Vectorize with batched numpy indexing |
| Line 94 | `budget = (raw_budget // total_models) * total_models` discards up to K-1 pulls | IDEA-09: Round up instead of down |
| Lines 519-582 | Iterative alpha-adjustment loop for budget reallocation | IDEA-10: Closed-form solution |
| Lines 762-763 | Fixed seed(42) | IDEA-11: Multi-seed ensemble |
| Line 812-813 | Raw empirical means for elimination | IDEA-02: Shrinkage estimator; IDEA-06: LCB |
| Line 782 | Random task sampling | IDEA-03: Variance-ordered task sampling |
| Lines 793-816 | K-1 phase elimination loop | IDEA-01: log2(K) halving; IDEA-04: Adaptive phases; IDEA-05: Multi-elimination |

## Data Format
- 15 datasets in `datasets/<name>/model_accuracies_filtered.pkl`
- Most are tuples `(model_accs_3d, oracle_3d)` — the eval script uses data directly
- Dataset `mmlu_20` is a plain (31, 14042) ndarray
- Model accuracies are float64, pre-computed per-task binary correctness

## Reusable Paper Data
- None mounted at `/paper_data`
- All pickle files included in repo under `datasets/`

## Metric Parser
- Parse from stdout: regex `SySRs Avg Identification Accuracy @ 12% budget: ([\d.]+)%`
- Per-dataset accuracies printed line-by-line

## Safe Modification Targets
- `bai_algs.py`: Add new algorithm functions, modify existing ones
- `eval_sysrs_12pct.py`: Change algorithm called, budget, k, seeds
- Do NOT modify: dataset pickle files, metric computation, test data

## Known Levers
1. Budget percentage (12% → any)
2. Algorithm choice (SySRs vs SR vs UCB-E vs SyUCB-E)
3. Number of runs k (1000 → higher for stability)
4. Algorithm is hyperparameter-free within SySRs itself
5. Dataset subsets
