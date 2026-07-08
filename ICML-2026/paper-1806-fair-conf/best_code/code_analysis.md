# Code Analysis for Paper 1806 SOTA Optimization

## Evaluation Path
- `run_conformal_direct.py` → `run_conformal()` → generates bios.csv + prediction sets
- `run_llm_direct.py` → reads bios.csv → LLM evaluation → generates max_ror_marginalized.csv, accuracy_statistics.csv, size_statistics.csv
- Full eval: `cd /repo && python3 run_conformal_direct.py && python3 run_llm_direct.py`

## Key Issue: Pipeline Synchronization
`run_llm_direct.py` hardcodes `conformal_result_dataset = "Jul07_09-20-57"`. When `run_conformal_direct.py` generates a new output directory, the LLM script must read from the correct directory.

## Config Path
- Base: `src/substantive/faircp/conf/config.yaml`
- Dataset: `src/internal/conf/dataset/bios.yaml`
- Custom overrides: `custom_config.yaml`
- Merge order: base → dataset → custom

## Metric Parsing (Label-Clustered CP)
- maxROR: From `logs/<timestamp>/max_ror_marginalized.csv` row `ConformalMethod.CLUSTERED_LABEL`
- Accuracy: From `logs/<timestamp>/accuracy_statistics.csv` — weighted average of Female/Male for "Label-Clustered"
- Average Set Size: From `logs/<timestamp>/size_statistics.csv` — weighted average of Female/Male for "Label-Clustered"

## Key Source Files
- SAPS scoring: `src/substantive/faircp/conformity/saps.py`
- Clustered CP: `src/substantive/faircp/conformal/clustered.py`
- Cluster config/algorithm: `src/substantive/faircp/conformal/clustered_cp.py`
- HPO: `src/substantive/faircp/conformity/hpo.py`
- Calibration: `src/substantive/faircp/calibration/calibration_methods.py`
- Run conformal: `src/internal/process/run_conformal.py`
- LLM evaluation: `src/internal/validation/run_llm_in_loop.py`

## Safe Modification Targets
- `custom_config.yaml` — hyperparameter changes (M_label, gamma_label, hpo_iterations, embedding_mode)
- `src/substantive/faircp/conformal/clustered_cp.py` — calibration split stratification, distance-weighted fallback
- `src/substantive/faircp/conformity/saps.py` — entropy reweighting, deterministic seeding
- `src/substantive/faircp/conformity/hpo.py` — temperature optimization logic
- `src/substantive/faircp/conformal/clustered.py` — config adaptation, HPO integration

## Risky Files (do not modify)
- Any dataset files under `src/internal/dataset/`
- `src/internal/process/run_conformal.py` (core pipeline — only modify if adding features)
- Test data: `data/BiosBias/test_all.pickle`

## Pre-trained Model
- Checkpoint: `logs/Jul07_09-05-52/checkpoints/model.pt` (BERT linear classifier)
- Retraining is possible but slow (~10 min); prefer skipping for fast CP-only iterations

## Known Levers
- SAPS T and lambda (conformity score hyperparameters)
- k (top-k for Cvg@k)
- alpha (conformal error rate)
- M_label (clusters for Label-Clustered CP)
- gamma_label (clustering data fraction)
- embedding_mode (upper_percentiles vs cdf_grid)
- summary_bins (for cdf_grid mode)
