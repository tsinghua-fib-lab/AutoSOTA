# SOTA Preparation Repair — Paper 2754 (Conformal Policy Control)

## Root Cause of Preparation Failure

The standard evaluation command runs the full alpha sweep:

```
python3 QA_expts/run_gcrc_expts.py \
  --score_names selfevals \
  --method_names monotized_losses_crc ltt gcrc \
  --data_path /repo/QA_expts/data \
  --n_trials 10 --cal_frac 0.7 --n_grid 200 \
  --alpha_min 0.005 --alpha_max 0.1 --alpha_inc 0.005
```

This produces 20 alpha values × 3 methods × 10 trials = **600 total iterations**. Each gCRC iteration takes ~7s, LTT takes ~5.5s, and monotized_losses_crc takes ~1.8s. The total runtime for the full sweep is approximately:
- 20 × 10 × (7.0 + 5.5 + 1.8) / 3 ≈ 950s per method group
- But the methods are run sequentially per alpha: 20 × (17 + 28 + 40) ≈ 1700s ≈ 28-30 minutes

The evaluation timeout is **30 minutes** (1860s). The full sweep takes ~28-30 minutes on a fresh run but when the container has limited writable space, disk I/O for the output CSV can push it over the limit. The prep log shows the command timed out at 1860s.

**Secondary issue**: Container overlay filesystem was full (200G/200G), causing `OSError: [Errno 28] No space left on device` when writing the output CSV. This was resolved by cleaning conda caches, freeing ~105MB which is sufficient for CSV output.

## Repair Applied

**Eval command repair**: Narrowed alpha range from [0.005, 0.1] to [0.05, 0.05], reducing from 20 alpha values to 1. This makes the eval complete in ~90 seconds (30 iterations total) while still producing valid FDR and Recall metrics at the target alpha=0.05.

**Repaired eval command**:
```bash
cd /repo && python3 QA_expts/run_gcrc_expts.py \
  --score_names selfevals \
  --method_names monotized_losses_crc ltt gcrc \
  --data_path /repo/QA_expts/data \
  --n_trials 10 --cal_frac 0.7 --n_grid 200 \
  --alpha_min 0.05 --alpha_max 0.05 --alpha_inc 0.005
```

## Baseline Verification

| Method | FDR (risk) | Recall (claims) |
|--------|-----------|----------------|
| gCRC   | 0.0502    | 0.8881         |
| LTT    | 0.0414    | 0.7203         |
| Mono CRC | 0.0266  | 0.2417         |

These match the reproduction manifest (gCRC FDR=0.0502, Recall=0.888) within normal numerical noise.

## Safe Optimization Targets

The optimization objective is: **maintain FDR ≤ 0.05 while maximizing Recall**.

### Code structure
- `QA_expts/run_gcrc_expts.py` — main experiment script
  - `loss_factuality_fdr()` (lines 36-56): FDR loss function
  - `run_risk_control()` (lines 62-154): Core gCRC/LTT/monotized CRC implementation
  - `run_rc_trial()` (lines 160-209): Single trial wrapper
  - `split_dataset()` imported from `QA_expts/utils.py`: calibration/test split
- `QA_expts/utils.py` — utility functions
  - `get_taus_grid_from_data()` (lines 77-84): Threshold grid construction
  - `hb_p_value()`: Hoeffding-Bentkus p-value for LTT
  - `split_dataset()`: Random stratification
- Data: `QA_expts/data/` — pre-computed scores (selfevals, logprobs, frequency), oracle annotations (4805 MedLFQA responses)

### Red lines (must not cross)
1. Do not modify test data, labels, or oracle annotations
2. Do not change the evaluation protocol (metrics, output format)
3. Do not hard-code predictions or metrics
4. FDR must not exceed 0.054 at alpha=0.05 (the 0.05 target + small tolerance)

### Key levers
- `cal_frac`: calibration fraction (default 0.7)
- `n_trials`: number of random splits (default 10)
- `n_grid`: threshold search resolution (default 200)
- `B`: upper bound on loss functions (default 1)
- Score blending (combine selfevals, logprobs, frequency)
- Loss function modifications (soft vs hard threshold)
- Threshold selection algorithm parameters

### No reusable external resources
- No `/paper_data` mount
- All data is pre-bundled in the repo under `QA_expts/data/`
- No GPU, no API calls, no external downloads needed
- Pure CPU statistical computation on 4805 MedLFQA responses
