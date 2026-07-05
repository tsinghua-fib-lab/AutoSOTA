# Code Analysis — Paper 385: Semantic Isotropy

## Evaluation Path
- Entry point: `scripts/reproduce_isotropy.py`
- Loads data from `/datasets/paper385/fsbio_phi35_segscore.jsonl` (182 entities, 10 responses each)
- Loads embedding model (Nomic V1) from HuggingFace cache at `/models`
- Computes isotropy per entity via `embedding_density()` → `get_vn()` (von Neumann entropy)
- Computes factuality as mean fraction of "True" statements per entity
- Runs OLS: `factuality ~ isotropy` with bootstrap confidence intervals
- Saves results to `/repo/outputs/isotropy_results_nomic_v1.{pkl,json}`

## Key Files
| File | Purpose | Risk |
|------|---------|------|
| `scripts/reproduce_isotropy.py` | Main evaluation script | SAFE to modify config section |
| `lib/python/semantic_isotropy/metrics/isotropy.py` | Isotropy computation (vNE, eigendecomposition) | SAFE to add functions, modify with care |
| `lib/python/semantic_isotropy/llm/embed.py` | API embedding wrappers | SAFE to read, risky to modify |
| `scripts/segscore/gen_metric.py` | SegmentScore generation | READ ONLY (reference) |

## Config Section (safe targets)
- `MODEL_NAME`: Nomic V1 / V1.5 (line ~29)
- `FP16`: toggle to FP32 (line ~34)
- `N_SAMPLES`: number of responses per entity (line ~33)
- `pooling_method`: passed to `embedding_density()` (line ~97)
- `N_BOOTSTRAP`: bootstrap iterations (line ~32)

## Metric Parser
- R² parsed from stdout: `R2  = VALUE +/- STD`
- Also available from output JSON: `r2_bootstrap_mean`
- Bottleneck: vNE only captures eigenvalue entropy, misses other spectral properties

## Red-Line Boundaries
- Data loading must remain unchanged (same 182 entities, same labels)
- OLS regression evaluation protocol preserved (bootstrap, R² computation)
- No hard-coded metric values
- No changes to factuality label computation

## Safe Modification Targets
1. Isotropy measure functions (add Frobenius, LogDet, InverseTrace, Gini)
2. Model config (FP16→FP32, Nomic V1→V1.5)
3. Feature engineering (polynomial expansion, multiple isotropy measures)
4. Pooling method, response truncation, n_samples
