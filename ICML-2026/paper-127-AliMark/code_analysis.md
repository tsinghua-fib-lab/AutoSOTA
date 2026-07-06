# SOTA Preparation Repair — Paper 127 (AliMark)

## Original Failure

The orchestrator could not finish the normal SOTA preparation step. The preparation log shows:
1. First `docker run` failed with `--network host` (administrative policy rejection)
2. Second `docker run` succeeded without `--network host` and with container-local proxy
3. The `git init` step failed because `git` was not installed in the container
4. `apt-get` failed due to proxy issues (502 Bad Gateway for archive.ubuntu.com via proxy 172.17.0.1:17890)

## Repairs Applied

1. **Git installation**: Installed `git` via `apt-get` by unsetting proxy environment variables first (direct connection to archive.ubuntu.com works, but proxy does not)
2. **Git repository**: Added safe.directory config. Repo already had `.git` directory from original clone. Created baseline commit and `_baseline` tag.
3. **Tooling**: Created `/tools/record_score.sh` (copied from host)
4. **HF cache**: Created `/autosota_cache/hf/hub` directory for model caching; pre-downloading required models

## Critical Discovery: Missing Attack Step

The manifest `eval_command` (`bash eval.sh c4 8 64 12`) skips the attack step. The pipeline is:
- Step 1: `1_generation.py` — generates watermarked/unwatermarked text
- Step 2: `2_attack.py` — applies paraphrasing attacks (Pegasus, Parrot, DIPPER, GPT-3.5) ← **MISSING from eval.sh**
- Step 3: `3_detection.py` — runs detection on all texts
- Step 4: `4_evaluation.py` — computes AUROC/TPR metrics

The existing 7 SOTA iterations (in scores.jsonl) all evaluated only "No Attack" where AUROC is at ceiling (100.0%). The optimization objective targets attack conditions (DIPPER: 91.5%, GPT-3.5: 92.3%), but attacks were never run.

## Corrected Evaluation

Created `eval_full.sh` which runs all 4 pipeline steps including the attack phase. The attack step is incremental — it checks if attack results already exist and skips completed entries.

## Reusable Resources

- **Generation results**: `/repo/_result/generation/block_size_8/c4_AliMark_facebook_opt-1.3b.json` — 46 watermarked samples with unwatermarked references (already generated, reusable)
- **Detection results**: `/repo/_result/detection/block_size_8/c4_AliMark_facebook_opt-1.3b.json` — 46 samples with No Attack detection scores (already computed, reusable)
- **Dataset**: `/repo/dataset/c4.json` — C4 dataset bundled in repo
- **Models**: Being pre-downloaded to `/autosota_cache/hf/hub/`

## Attack Pipeline Details

- **Pegasus**: `tuner007/pegasus_paraphrase` (~2GB), via SemStamp wrapper
- **Parrot**: `prithivida/parrot_paraphraser_on_T5` (~1GB), via SemStamp wrapper  
- **DIPPER**: `kalpeshk2011/dipper-paraphraser-xxl` (T5-XXL, ~42GB) — large download, may be skipped if infeasible
- **GPT-3.5**: Requires OPENROUTER_API_KEY (not configured)

## Safe Optimization Targets

All changes should be detection-side only:
- Detection score aggregation (max → median/mean/trimmed-mean)
- Multi-key ensemble detection
- ABSA alpha/beta parameter tuning
- Soft scoring (temperature-scaled bits)
- Sentence weighting
- RS multistep candidates
- Detection embedder changes

Generation-side changes (temperature, top_p, block_size) require full re-generation and are more expensive.
