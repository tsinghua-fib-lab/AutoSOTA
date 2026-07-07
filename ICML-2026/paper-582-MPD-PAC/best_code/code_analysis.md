# Code Analysis: Paper 582 — MPD-PAC SOTA Preparation Repair

## Original Failure

The orchestrator ran the evaluation command from the reproduction manifest without `--max-samples`:
```
cd /repo/LaViDa && HF_HOME=/models HF_ENDPOINT=https://hf-mirror.com python3 eval_refcocog.py \
  --prior 0.3 --rope 0.01 --k 3 --slope 12.0 --center 0.6 \
  --max-new-tokens 8 --step-per-block 4 --output outputs/refcocog_ours.json
```

The full RefCOCOg validation set has **7,573 samples**. At ~1.2s per sample, the evaluation requires approximately **2.5 hours**. The 60-minute timeout caused the evaluation to terminate at ~37% completion (2,800/7,573 samples).

## Root Cause

The orchestrator did not add `--max-samples N` to the eval command. The eval script (`eval_refcocog.py`) supports this flag and computes metrics on the specified subset. **Critically, the manifest baseline metrics (CIDEr 41.72, Bleu_4 2.84, METEOR 10.45) were computed on 500 samples**, not the full 7,573. The 500-sample evaluation matches these values exactly.

## Corrected Evaluation Command

```bash
cd /repo/LaViDa && HF_HOME=/models HF_ENDPOINT=https://hf-mirror.com \
  python3 eval_refcocog.py \
  --prior 0.3 --rope 0.01 --k 3 --slope 12.0 --center 0.6 \
  --max-new-tokens 8 --step-per-block 4 \
  --max-samples 500 \
  --output outputs/refcocog_ours.json
```

- **Evaluation time**: ~10.5 minutes (500 samples at ~1.25s/sample)
- **Well within** the 60-minute timeout
- **Allows** ~5-6 full optimization iterations per hour

## Baseline Verification

| Metric  | Manifest Baseline | Verified Baseline (500 samples) |
|---------|-------------------|--------------------------------|
| CIDEr   | 41.72             | 41.72 ✓                        |
| Bleu_4  | 2.84              | 2.84 ✓                         |
| METEOR  | 10.45             | 10.45 ✓                        |

The verified baseline matches the manifest exactly. The 500-sample subset is stable enough for optimization (metrics are computed on 500 generated captions vs references).

## Container State

- **Container**: `autosota_sota_paper_582` from `autosota/paper-582:reproduced`
- **GPUs**: 2× A100-SXM4-80GB (container indices 0, 1)
- **Model**: `/models/lavida-llada-v1.0-instruct` + `/models/siglip-so400m-patch14-384`
- **Dataset**: RefCOCOg (loaded via Hugging Face datasets from `/models` cache)
- **Git**: Clean repo at `/repo`, baseline tagged `_baseline`
- **Artifacts**: `/autosota_artifacts/paper-582/sota/` (scores.jsonl, final_report.md)
- **Tools**: `/tools/record_score.sh` available and executable

## Safe Optimization Targets

### Key Source Files
- `repo/LaViDa/llava/model/language_model/llada_ours/generate.py` (425 lines)
  - `generate()`: Main generation loop with block scheduling
  - `init_prior_lastlayer_subspace_from_32layers()`: MPS prior computation
  - Mask prior suppression applied at lines ~318-349
- `repo/LaViDa/llava/model/language_model/llada_ours/modeling_llada.py` (1780 lines)
  - RoPE implementation with monotonic frequency mask
  - LLaDABlock attention mechanism
- `repo/LaViDa/eval_refcocog.py`: Evaluation script with `--max-samples` support

### CLI Parameters Available for Sweeping
- `--prior` (MPS lambda, default 0.3): Mask prior suppression strength
- `--rope` (MRS beta, default 0.01): Monotonic RoPE scaling
- `--slope` (MRS eta, default 12.0): RoPE scaling slope
- `--center` (MRS tau_0, default 0.6): RoPE scaling center
- `--k` (PCA rank, default 3): Prior subspace dimensionality
- `--step-per-block` (default 4): Denoising steps per block
- `--max-new-tokens` (default 8): Generation length

### Algorithmic Changes (require code edits)
1. EMA smoothing of mask prior direction across steps (ALGO-03)
2. Per-layer adaptive RoPE scaling (ALGO-06)
3. KL-stability adaptive unmasking (ALGO-01)
4. Cross-attention visual token reweighting (ALGO-05)

## Constraints
- No changes to metric definitions, dataset splits, or benchmark outputs
- All code changes inside container under `/repo`
- Use `/tools/record_score.sh` for every score record
- Git commit each successful implementation
