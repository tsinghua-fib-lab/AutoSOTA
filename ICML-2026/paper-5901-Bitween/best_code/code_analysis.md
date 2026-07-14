# Code Analysis for Paper 5901: Learning Randomized Reductions

## Evaluation Path
- **Command**: `bash run_eval.sh results/<dir> 1800`
- **Script**: `/repo/run_eval.sh` → runs two Python modules:
  1. `bitween.evaluation.evaluation_rsr_bench_paper` (40 main functions)
  2. `bitween.evaluation.evaluation_rsr_bench_paper_extended` (40 extended functions)
- **Entry point**: `evaluate()` in `src/bitween/evaluation/evaluation_rsr_bench_paper.py`
- **Core pipeline**: `infer_property_with_timeout()` → `infer_property()` → `bitween()` in `src/bitween/main.py`

## Config Path
- **Config file**: `/repo/bitween.ini`
- **Config class**: `src/bitween/config.py` → singleton `Config` reads from `bitween.ini`
- **Key settings**: degree=2, epsilon=0.001, precision=2, cross_validation=3, regression_score=r2

## Metric Parser
- **Output format**: Per-function `.txt` files in results directory
- **Parsing regexes** (from `run_eval.sh`):
  - `Verified (N):` → Verified Count
  - `Unverified (N):` → Unverified Count
  - `Faulty (N):` → Faulty Count
  - `Took time: X.XXs` → timing
- **Aggregate metrics**: RSR Count = sum(Verified), Verified Count = sum(Verified), Unverified Count = sum(Unverified), RSR Coverage = count(functions with Verified>0), Average Time = total_time / 80

## Train/Inference Path
1. Sample generation: `infer_property()` samples function values from domain/distribution
2. Trace processing: `process_trace()` generates monomials up to `degree`
3. Model fitting: `multiple_regression_heuristics()` → GridSearchCV over Linear/Ridge/Lasso
4. Equation inference: `infer_equations()` → converts coefficients to symbolic equations
5. Regression refinement: 3 iterations of re-fitting on selected terms
6. Reduction: `Reducer.merge_equations()` → Gröbner basis + union-find dedup
7. Verification: `verify_with_timeout()` → symbolic verification

## Reusable Resources
- No `/paper_data` mount
- Cache mounts: `/autosota_cache`, `/datasets`, `/models`

## Safe Modification Targets
1. `main.py:530-536` — `model_params` dict (add ElasticNet)
2. `bitween.ini:19` — `cross_validation` value
3. `main.py:748-754` — coefficient cutoff logic
4. `bitween.ini:5` — `precision` value
5. `main.py:563` — `random_state` in train_test_split
6. `bitween.ini:20` — `regression_score` value
7. `main.py:844,852,866` — `find_model()` → `find_model_w_lassocv()`

## Risky Files (DO NOT MODIFY)
- `src/bitween/evaluation/evaluation_rsr_bench_paper.py` — test definitions (80 functions)
- `src/bitween/evaluation/evaluation_rsr_bench_paper_extended.py` — extended tests
- `src/bitween/config.py` — config schema (additive changes OK)
- `src/bitween/verifier.py` — verification logic
- `src/bitween/checker.py` — fuzzing logic
- `run_eval.sh` — evaluation harness and metric parsing
