# Code Analysis — Paper 2796 SOTA Optimization

## Evaluation Path
- **Entry**: `Rscript /autosota_cache/eval_reproduce.R`
- **Source**: `/repo/sims/Test_Linear_helperFxns_misc.R` (helper functions)
- **Method**: strategize R package (installed, patched)
- **Key config**: kFactors=5, nObs=10000, nMonteCarlo=10, L2 penalty, GLM outcome model
- **Flow**: Data gen → Lambda calibration → strategize() call → Q estimation

## Key Parameters in eval_reproduce.R
| Line | Parameter | Current Value | Effect |
|------|-----------|---------------|--------|
| 10 | kFactors | 5 | Number of treatment factors |
| 11 | nObs | 10000 | Observations per MC iteration |
| 12 | nMonteCarlo | 10 | Monte Carlo repetitions |
| 59 | TARGETQ selection | which.min(abs(Q-1)) | Lambda calibrated for Q≈1, not max Q |
| 88 | penalty_type | "L2" | L2 trust-region penalty |
| 89 | use_regularization | FALSE | GLM without glinternet screening |
| 90 | use_optax | FALSE | Adagrad-style SGD, no optax |
| 90 | nSGD | 1000 | SGD iterations |

## Key Code Paths in strategize
- **two_step_master.R**: Entry point, handles use_regularization logic
  - Line 793: K>1 forces use_regularization=TRUE (but K=1 in eval, so FALSE sticks)
  - Line 474-475: Loop control — use_regularization=FALSE exits after first GLM fit
- **two_step_model_outcome_glm.R**: Outcome model fitting
  - Lines 479-558: glinternet block (ONLY reached when use_regularization=TRUE)
  - Line 301-475: Loop controlling regularization vs. non-regularization path
- **two_step_optimize_gd_seq.R**: pi* optimization
  - Lines 173-195: compute_penalty_value (L2, L1, KL, LInfinity supported)
  - Lines 1442-1447: optax code path (disabled by use_optax=FALSE)

## Metric Parser
- Parse from stdout: `Mean Q (Optimal SI): <value>`
- Also: `Max-AMCE Q: <value>`
- Format: `===== RESULT =====` block followed by metric lines

## Safe Modification Targets
1. `eval_reproduce.R` line 89: use_regularization=TRUE (enable glinternet)
2. `eval_reproduce.R` line 59: which.max(impliedQ_vec) (maximize Q, not target Q=1)
3. `eval_reproduce.R` line 88: penalty_type="KL" or "elasticnet"
4. `eval_reproduce.R` line 90: use_optax=TRUE, nSGD=3000
5. `eval_reproduce.R` line 12: nMonteCarlo=25-50

## Verified: glinternet works
- glinternet 1.0.12 loaded successfully
- Code path at two_step_model_outcome_glm.R:479-558 is complete
- use_regularization=TRUE + K=1 + ok_counter=1 triggers glinternet path
- nFolds_glm defaults to 3, sufficient for nObs=10000
