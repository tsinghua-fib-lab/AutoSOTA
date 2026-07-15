# Code Analysis: Paper 6058 SOTA Preparation Repair

## Original Preparation Failure

The preparation failed because:
1. **git not installed**: The reproduced Docker image (`autosota/paper-6058:reproduced`) has Python/torch but no git. The apt proxy (172.17.0.1:17890) returned 502 errors, blocking `apt-get install git`.
2. **Fix**: Run apt without proxy (`unset HTTP_PROXY...`) — apt can reach archive.ubuntu.com directly. Git 2.25.1 installed successfully.
3. **`/tools` missing**: Created `/tools` and copied `record_score.sh` from host.
4. **`.env` present but env vars not exported**: The `.env` file at `/repo/.env` sets `MODEL_ROOT=/models`, `SAE_ROOT=/models`, `DATASET_ROOT=/datasets`. `run_pipeline.py` calls `load_dotenv()` which loads them correctly.

## Verified Baseline

- **Command**: `cd /repo && python3 run_pipeline.py`
- **Result**: F1=85.33% (precision_safe=0.90, recall_safe=0.74, precision_unsafe=0.79, recall_unsafe=0.92)
- **Matches reproduction**: Manifest baseline 85.3%
- **Recorded**: iteration 0, commit bfa70d4b, status=success

## Available Optimization Levers (without regenerating activations)

The pipeline reads pre-computed SAE activations from `.pt` files. These were generated with layer 18, trainer_2.

### Easily Tunable Parameters (in run_pipeline.py):
1. **TOP_K** (line ~68): Number of top features selected. Default 32. Paper shows K affects precision/recall trade-off.
2. **Feature selection method** (line ~73): Uses `top_diff_ids` (standardized mean difference). Alternatives in GlobalMetricsResult: `top_f1_ids`, `top_precision_ids`, `top_recall_ids`, `pareto_front_ids`.
3. **Random seed** (line ~128): For calibration/test split. Default 42.
4. **Calibration split ratio** (line ~126): Default 80/20. 
5. **Threshold optimization**: Default `precision_recall_curve`.

### NOT Available:
- SAE layer variation: Only layer 18 checkpoint exists in `/models/adamkarvonen/qwen3-8b-saes/`
- SAE trainer variation: Only one variant available
- Different base models: Only Qwen3Guard-Gen-8B available

## Optimization Strategy

Since NExT-Guard is training-free, improvements come from hyperparameter tuning:
1. Grid search over TOP_K (16, 32, 48, 64, 96, 128)
2. Try different feature selection methods (top F1 vs top diff vs pareto)
3. Multi-seed evaluation (3+ seeds) for robust measurement
4. Tune calibration split ratio
5. Combine best parameters and report with std deviation

## Reusable Resources

- Models: `/models/Qwen/Qwen3Guard-Gen-8B` (15.5GB)
- SAE: `/models/adamkarvonen/qwen3-8b-saes/saes_Qwen_Qwen3-8B_batch_top_k/` (layer 18)
- Dataset: `/datasets/Aegis-AI-Content-Safety-Dataset-2.0/` (Aegis2.0)
- Pre-computed activations: `/repo/results/Guard_Qwen3-8B_20260206_1005/Guard_Qwen3-8B_20260714_1533/predictions/Aegis2.0.pt`
