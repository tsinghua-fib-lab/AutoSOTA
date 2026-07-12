# Code Analysis for Paper 4004 SOTA Optimization

## Key Files
- **Evaluation path**: `/repo/LLaVA/llava_chair.py::main()` — loads model, runs generation, evaluates CHAIR metrics
- **Core methods**: `/repo/LLaVA/llava_methods.py` — CARD extraction, Bayesian Gating, hooks
- **Configuration**: `/repo/config.py` — model paths, hyperparameters, dataset paths

## Evaluation Command
```
cd /repo/LLaVA && python3 llava_chair.py --limit 500 --run_beta --decoding greedy --max_new_tokens 512
```
- Loads LLaVA-1.5-7B from `/models/llava-v1.5-7b-hf`
- Uses Hybrid CHAIR Evaluator (segments + captions GT)
- Output format: `Result (Beta): CHAIRs=X.XXXX, CHAIRi=X.XXXX`
- Results cached in `results/chair_llava/` — skip if metrics file exists

## Config Path
- `/repo/config.py` — imported by `llava_chair.py`
- Key params: INJECTION_LAYERS=[30], BETA_ALPHA_MAX=[20], BETA_K=[5.0], BETA_C=[1.0]

## Safe Modification Targets
1. `llava_methods.py::BayesianGatingHookMaskedDynamic.__call__` — gate computation, RMS matching
2. `llava_methods.py::compute_card_vector_batch` — CARD extraction, pooling method, token pruning
3. `llava_chair.py::run_once_beta` — hook setup, CARD pre-computation
4. `config.py` — hyperparameter values (injection layer, alpha_max, k, c)

## Risky Files (DO NOT MODIFY)
1. `llava_chair.py::HybridCHAIREvaluator` — metric computation (evaluator)
2. `llava_chair.py::CHAIRImageDataset` — data loading/splitting
3. Any scoring script or output format parser

## Known Bottlenecks
1. Duplicate gate computation in BayesianGatingHookMaskedDynamic (lines 250-276 vs 278-306)
2. RMS matching code exists but never enabled (rms_match=False always)
3. Single-layer CARD extraction (layer 30 only)
4. Fixed hyperparameters — no per-token adaptation
5. Per-token norm cap exists in commented-out SimpleAddHook but not in Beta gate
