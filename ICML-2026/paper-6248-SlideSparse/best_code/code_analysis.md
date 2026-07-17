# Code Analysis — Paper 6248 SOTA Preparation Repair

## Original Preparation Failure

The SOTA preparation step failed because:
1. **Model missing**: `/models/Qwen2.5-7B/` did not exist in the `autosota_sota_paper_6248` container. The cache mount (`/home/.../cache/models:/models`) was empty — the model was downloaded to the reproduction container's own overlay filesystem, not the shared cache.
2. **Evaluation command error**: `transformers 4.46.0` rejected `/models/Qwen2.5-7B` as a valid local path, interpreting it as a HF repo ID and trying to download it.

## Repairs Applied

### 1. Model Download (CODE-03 pre-requisite)
- Downloaded Qwen2.5-7B from hf-mirror.com to `/models/Qwen2.5-7B/` (15GB, 14 files)
- Used direct `snapshot_download` with `HF_ENDPOINT` unset (proxy blocked HF API calls through hf-mirror.com)

### 2. BFloat16 Fix (CODE-03)
- Changed `main.py` line 17: `torch.float16` → `torch.bfloat16`
- Qwen2.5-7B requires bfloat16; float16 causes NaN activation overflow around layers 26-28
- This alone contributed +0.49pp over the reproduction baseline (71.07% vs 70.58%)

### 3. Git Repository Cleanup
- The git repository had a 9.5GB pack file containing committed pruned model safetensors
- Rebuilt .git with proper .gitignore, reducing git size from 9.7GB to 141MB
- Established clean `_baseline` tag

### 4. Disk Space Management
- Overlay filesystem only has ~13GB writable space (200GB total including image layers)
- Saved all pruned models to NFS-mounted `/autosota_cache/pruned_models/` instead of `/repo/pruned_models/`

### 5. Calibration Sample Count Fix
- `prepare_calibration_input` in `prune.py` had hardcoded `torch.zeros((128, ...))` — fixed to `torch.zeros((len(dataloader), ...))`

### 6. Calibration Data Flexibility
- Added `--calibration_data` argument to `main.py` (choices: wikitext2, c4, mixed)
- Added `get_mixed()` function to `data.py` for wikitext2+c4 interleaved calibration
- Fixed C4 loader config name: `'allenai--c4'` → `'en'`
- Note: C4 download blocked by proxy; mixed calibration not testable in this environment

## Corrected Evaluation Command

```bash
cd /repo/accuracy_eval/wanda && CUDA_VISIBLE_DEVICES=0 python main.py \
  --model /models/Qwen2.5-7B \
  --prune_method wanda \
  --sparsity_type 6:8 --prune_n 2 --prune_m 8 --seed 0 \
  --save_model /autosota_cache/pruned_models/<name>/model && \
cd /repo && CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
  --model_args pretrained=/autosota_cache/pruned_models/<name>/model,dtype=bfloat16,trust_remote_code=True \
  --tasks mmlu --num_fewshot 5 --batch_size auto --output_path /repo/eval_results/mmlu_<name>
```

## Summarized Optimization Progress

| Iter | Idea | Method | Config | MMLU Acc | Status |
|------|------|--------|--------|----------|--------|
| 0 | BASELINE | Wanda | ns=128, s=0 | 71.07% | success |
| 1 | ALGO-02 | SparseGPT | ns=128, s=0 | 70.69% | success |
| 2 | CODE-01 | Wanda+C4 | ns=128, s=0 | FAILED | failed |
| 3 | PARAM-01 | Wanda | ns=256, s=0 | **71.29%** | success ★ |
| 4 | PARAM-01 | Wanda | ns=256, s=42 | 71.25% | success |
| 5 | PARAM-01 | SparseGPT | ns=256, s=0 | 70.10% | success |

**Best result**: Iter 3 — Wanda 6:8, 256 calibration samples, seed=0 — **71.29%** MMLU (+0.22pp over baseline, +0.71pp over reproduction).

## Key Findings

1. **BFloat16 is critical**: Loading Qwen2.5-7B in bfloat16 vs float16 gives ~0.5pp improvement
2. **Calibration sample count matters**: 256 samples > 128 samples for Wanda (+0.22pp)
3. **SparseGPT underperforms Wanda** on this specific task: despite 0.12-0.17 better wikitext2 PPL, MMLU accuracy is consistently lower (70.10-70.69% vs 71.07-71.29%)
4. **Wikitext2 perplexity ≠ MMLU accuracy**: The PPL-accuracy correlation is weak for N:M structured sparsity at 25% rate
5. **Seed sensitivity is modest**: Seed 0 vs 42 at 256 samples differed by only 0.04pp

## Remaining Unexplored Ideas

- ALGO-01 (Per-layer adaptive sparsity via outlier ratio)
- ALGO-03 (Layer-type-aware pruning for SwiGLU)
- ALGO-04 (Symmetric importance scoring)
- ALGO-05 (Post-pruning weight reconstruction / DSnoT-lite)
- ALGO-06 (Gradient-guided importance)
- ALGO-07 (Iterative hard-example calibration mining)
- CODE-02 (Validation perplexity as quality gate)
- CODE-04 (Multi-seed importance score averaging)
- C4/mixed calibration (blocked by network)
- Further nsamples sweep beyond 256
