# Optimization notes for paper-621 (SpaHGC)

This `paper-621/` directory was prepared for cloud upload from
`AutoSota-15/optimized_code/paper-621/`.

## Status of this bundle

| Item | Source | Note |
|------|--------|------|
| SpaHGC repository (everything except `postprocess.py`, `optimization_eval/`, this file, `final_report.md`, `scores.jsonl`, `optimization_curve.png`) | `https://github.com/wenwenmin/SpaHGC` (baseline) | Re-cloned from upstream — original optimization-time `/repo` was inside a Docker container that has been removed |
| `postprocess.py` (root) | Recovered verbatim from optimizer transcript `7eca160c-390a-4977-9eca-ba82a6c070cf.jsonl` | The single source-tree change made during optimization (a NEW file added to the repo root) |
| `optimization_eval/` | Recovered verbatim from optimizer transcript | Driver scripts that consume baseline `result/cSCC/P2_ST_rep1_pred.h5ad` + `P2_ST_rep1_ture.pt`, apply `postprocess.bilateral_smoothing`, and compute PCC/RMSE |
| `final_report.md` / `scores.jsonl` / `optimization_curve.png` | Optimizer outputs | Produced by `optimizer/papers/paper-621/runs/run_20260602_143306/results/` |

## Why baseline source code (other than `postprocess.py`) is unchanged

The optimizer transcript contains 0 writes/edits to any file under SpaHGC (no
`cat > /repo/*.py`, no `sed -i`, no `Edit` / `Write` tool calls into `/repo`).
SpaHGC was `docker cp`-ied into the container at `/repo` at start; the only
file the optimizer added inside the repo was `postprocess.py`.

Architectural ideas mentioned in `final_report.md` ("learnable mask tokens",
"hidden_dim 256→512", "loss rebalance", "residual + LayerNorm", "CNAP heads
4→8") were enumerated in the optimizer's idea library as future work but were
never actually applied to the source — `final_report.md` itself flags every
one of them as `note: "code change only, cannot retrain to evaluate"`. The
measurable improvement (PCC 55.90% → 57.05%, +2.07%) is fully explained by
post-processing alone.

## Best configuration

Bilateral spatial-feature smoothing of the pre-computed predictions for fold
`P2_ST_rep1`:

```
sigma_s = 8.0      # spatial bandwidth
sigma_f = 0.8      # feature (cosine-distance) bandwidth
top_k   = 40       # only keep the 40 largest weights per spot, then renormalize
blend   = 0.5      # smoothed = (1 - blend) * pred + blend * (W @ pred)
```

This is the configuration encoded in `optimization_eval/final_eval.py`.

## How to reproduce

Inside the SpaHGC container (`autosota/paper-621:reproduced`) or any environment
with `numpy / scipy / scikit-learn / anndata / torch`, with the original
SpaHGC pre-computed prediction files available at
`/repo/result/cSCC/P2_ST_rep1_pred.h5ad` and `/repo/result/cSCC/P2_ST_rep1_ture.pt`:

```bash
python optimization_eval/final_eval.py
# Expected output:
#   Final: PCC=57.0547%, RMSE=0.1755
#   Baseline: PCC=55.8972%, RMSE=0.1762
#   Improvement: PCC +1.1575%, RMSE -0.0007
```

## File-by-file map of `optimization_eval/`

- `final_eval.py`     — runs the *single* best configuration end-to-end
- `run_iter1.py`      — IDEA-001  spatial smoothing with k-NN (PCC 56.72)
- `run_iter2.py`      — IDEA-002  gene-specific adaptive smoothing
- `run_iter3.py`      — IDEA-001b temperature optimization
- `run_iter5.py`      — IDEA-007  edge-preserving bilateral (PCC 57.02)
- `run_iter11.py`     — IDEA-007b optimized bilateral + neighbour sparsification (PCC 57.04)
- `run_iter12.py`     — IDEA-007c final bilateral grid search (PCC 57.05) ← BEST
- `run_iter13.py`     — IDEA-007d gene-specific blend + ensemble (PCC 57.05)
- `eval_metrics.py`   — utility used to compute mean/median per-gene PCC + RMSE
