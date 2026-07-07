# Code Analysis: K-BTS (Paper 1650)

## Evaluation Path
- **Script**: /repo/eval_metrics.py
- **Command**: PYTHONPATH=/repo python3 eval_metrics.py results/<dir> 100
- **Input**: Per-target dirs with init_score.csv and <seed>_result.csv
- **Output**: stdout + JSON at results/<dir>/reproduction_metrics.json
- **Metrics**: Top1/5/10/20 docking Avg/Med, QED Avg/Med, SA Avg/Med, Diversity Avg/Med

## Train/Inference Path
- **Main**: /repo/k_bts_rand.py (rand init)
- **Alt**: /repo/k_bts_diff.py (docking-based init)
- **Flow**: init molecules -> SeedSelector -> KnowledgeManager -> DMTA loop -> LLM (DeepSeek) -> smina docking -> TS update

## Config
- No external config; params hardcoded in k_bts_rand.py
- Key: TARGET_COUNT=100, MAX_ITERS=500, tau=2.0, sim_threshold=0.3, min_delta=0.5, temperature=0.7

## Pre-computed Results Available
- results/rand/ (baseline, 100 targets x 100 mols)
- results/rand_wo_knowledge/, rand_wo_lower/, rand_wo_upper/, rand_wo_warmstart/
- results/diff/ and diff_wo_* variants

## Risky Files (DO NOT MODIFY)
- eval_metrics.py (red-line)
- utils/rdkit_tools.py (internal_diversity)
- datasets/crossdocked/structure-files-test/ (test data)

## Safe Modification Targets
- New result directories with molecule selections from existing runs
- k_bts_rand.py params, core/selector.py, knowledge/manager.py
- utils/llm_tools.py, utils/prompt_tools.py

## No API Key Available
DeepSeek API key not configured. Full pipeline re-run not possible.
Optimization via: cross-variant ensemble, smart molecule selection, algorithm modifications.

## Container Eval Command
cd /repo && PYTHONPATH=/repo python3 eval_metrics.py results/<dir> 100
