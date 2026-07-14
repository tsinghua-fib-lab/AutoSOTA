# Code Analysis — Paper 5507 SOTA Preparation Repair

## Preparation Failure

**Root cause:** The manifest `eval_command` uses `${oc.env:DATA_PATH}` in `config/9_multiret_m3.yaml`, but the fresh container did not have `DATA_PATH` set. The preparation also failed to pass the environment variable through `docker exec`.

**Fix applied:**
1. Set `export DATA_PATH=/autosota_cache/data` before running the eval command
2. Created symlink: `/autosota_cache/data/neulab/ted_multi/encodings -> /autosota_cache/data/ted_multi/encodings` to bridge the `dataset_name: neulab/ted_multi` path construction in `load_space()`

**Corrected in-container evaluation command:**
```bash
cd /repo && unset HF_ENDPOINT && export DATA_PATH=/autosota_cache/data && uv run python -m scripts.exps.9_multiret --config-name 9_multiret_m3
```

Note: `HF_ENDPOINT` must be unset (hf-mirror.com SSL issues with huggingface_hub 0.34.4).

## Baseline Verification

| Metric | Reproduction | This Run | Match |
|--------|-------------|----------|-------|
| GCPA Avg | 0.7050 | 0.7049 | ✓ (fp-noise) |
| GCPA Worst | 0.5930 | 0.5928 | ✓ (fp-noise) |
| GCCA Avg | 0.6950 | 0.6954 | ✓ (fp-noise) |

## Per-Pair GCPA Metrics (Baseline)

| Pair | Hits@1 |
|------|--------|
| ROBER→SPBER (EN→ES) | 0.7160 |
| ROBER→CAMEM (EN→FR) | 0.7774 |
| SPBER→ROBER (ES→EN) | 0.7468 |
| SPBER→CAMEM (ES→FR) | 0.6115 |
| CAMEM→ROBER (FR→EN) | 0.7852 |
| CAMEM→SPBER (FR→ES) | 0.5928 |

**Bottleneck pair:** CAMEM→SPBER (FR→ES) at 0.5928 — worst of all 6 directed pairs.

## Safe Optimization Targets

1. `gc_tau` and `gc_lam` in `config/alignment.yaml` — primary levers
2. `_GCCorrector` in `src/cycloreps/translator/gpa.py` — MLP architecture
3. GC training loss in `_fit_gc` — currently cosine similarity
4. GPA initialization in `align()` — currently random
5. GC training hyperparams: lr, epochs, batch size

## Red Lines
- Do NOT change evaluation protocol, test split, metric computation
- Do NOT modify pretrained encoder weights
- All changes must be in the alignment code path only

## Reusable Resources
- Pre-computed embeddings at `/autosota_cache/data/ted_multi/encodings/`
- Cached HuggingFace models at `/autosota_cache/hf/hub/`
- Results stored at `/repo/results/9_multiret_m3.json`
