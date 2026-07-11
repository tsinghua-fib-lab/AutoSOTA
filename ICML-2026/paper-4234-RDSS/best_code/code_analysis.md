# Code Analysis for Paper 4234: LpSq-QuasiNorm

## Evaluation Path
- **Command**: `octave eval_salinasA.m`
- **Script**: `/repo/eval_salinasA.m`
- **Timeout**: 60 minutes
- **Output format**: Last line matching `EVAL_RESULT PSNR=<float> SSIM=<float>`

## Core Algorithm Flow
1. `eval_salinasA.m` - loads SalinasA_corrected_83x86x204.mat, normalizes, adds noise
2. For each trial: creates random mask (SR=10%), adds Gaussian noise (NSR=0.05)
3. Calls `f_ntc_LpSq_ADMM_dct(obs, opts, memoLpSq)` - the ADMM solver
4. ADMM solver does t-SVD in DCT domain, applies `f_prox_t_LpSq_dct` proximal operator
5. Proximal operator: `f_tsvd_dct` -> `f_prox_p_over_q_inexact` (singular value thresholding) -> `f_prox_q_by_l1_2` (Lq thresholding)
6. Returns reconstructed tensor `memoLpSq.T_hat`
7. Computes PSNR via `h_Psnr(L, T_hat)` and SSIM via `compute_ssim_3d(L, T_hat)`

## Key Parameters (in eval_salinasA.m)
- `obsRatio = 0.10` (10% sampling rate)
- `NSR = 0.05` (noise coefficient)
- `p = 0.80, q = 0.81` (S3 setting)
- `lambda = 0.11` (regularization)
- `rho = 1e-5, nu = 1.1` (ADMM penalty parameters)
- `MAX_ITER_OUT = 250` (max iterations)
- `MAX_EPS = 2e-5` (convergence tolerance)
- `n_trials = 10`

## Metric Parser
- PSNR: `h_Psnr(L, memoLpSq.T_hat)` - 10*log10(peak^2/MSE)
- SSIM: `compute_ssim_3d(L, memoLpSq.T_hat)` - mean SSIM over all bands using standard formula
- Final line: `fprintf('\nEVAL_RESULT PSNR=%.4f SSIM=%.4f\n', mean(psnr_vals), mean(ssim_vals))`

## Reusable Resources
- Dataset: `SalinasA_corrected_83x86x204.mat` (included in repo, 83x86x204 tensor)
- No external downloads required
- No GPU needed (Octave CPU-only)

## Safe Modification Targets
| File | Purpose | Safety |
|------|---------|--------|
| `f_ntc_LpSq_ADMM_dct.m` | ADMM optimizer core | Safe - internal optimization logic |
| `HelperFunctions/f_prox_t_LpSq_dct.m` | Proximal operator dispatch | Safe - internal thresholding |
| `HelperFunctions/f_prox_q_by_l1_2.m` | Lq proximal operator | Safe - internal computation |
| `HelperFunctions/f_prox_p_over_q_inexact.m` | Group sparse proximal | Safe - internal computation |
| `eval_salinasA.m` | Evaluation runner | Only change opts parameters, not metric computation |

## Risky Files (DO NOT MODIFY)
| File | Reason |
|------|--------|
| `HelperFunctions/h_Psnr.m` | Metric definition - must not change |
| `HelperFunctions/compute_ssim_3d.m` | Metric definition - must not change |
| `SalinasA_corrected_83x86x204.mat` | Test data - must not modify |

## Known Issue
PSNR peaks at ~31.67 dB during ADMM iterations (iteration ~80-100) but declines to 30.99 dB
at convergence (iteration ~150-200). The ADMM stopping criterion (eps < MAX_EPS) measures
optimization convergence, not reconstruction quality. PSNR-based early stopping (CODE-1)
should capture the peak.

## Container
- Name: `autosota_repro_paper_4234`
- Image: `autosota/paper-4234:reproduced`
- Tool: `record_score.sh` at `/tools/record_score.sh`
- Scores file: `/autosota_artifacts/paper-4234/sota/scores.jsonl`
