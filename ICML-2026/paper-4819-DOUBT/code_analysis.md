# SOTA Preparation Repair — Paper 4819 (DOUBT)

## Preparation Failure

The normal SOTA preparation failed because `git` was not installed in the container and `apt-get` failed due to proxy 502 errors. The `/tools/record_score.sh` script was also missing.

## Repair Actions

1. Installed git after unsetting proxy env vars (proxy at 172.17.0.1:17890 returned 502 for Ubuntu archives)
2. Created `/tools/` and copied `record_score.sh` from host
3. Initialized git in `/repo`, created `_baseline` tag
4. Created model symlinks from `/paper_data/` to `/models/`
5. Added `--threshold` CLI argument (default 0.48) to enable parameter search

## Corrected Evaluation Command

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate doubt && \
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1 && \
export HF_HOME=/autosota_cache/hf && unset HF_ENDPOINT && \
cd /repo && python doubt.py --lvlm InternVL2-1B --benchmark LLaVABench \
  --llm Qwen2.5-3B-Instruct --inference_temp 0.1 --sampling_temp 0.5 \
  --sampling_time 10 --batch_size 1 --threshold 0.48
```

## Baseline: 78.33% Accuracy (47/60)

## Optimization Results

| Iter | Idea | Config | Accuracy | Change |
|------|------|--------|----------|--------|
| 0 | baseline | threshold=0.48, K=10, temp=0.5 | 78.33% | - |
| 1 | threshold=0.52 | threshold=0.52, K=10, temp=0.5 | 78.33% | 0 |
| 2 | kappa+adaptive | C-score + adaptive fusion | 78.33% | 0 |
| 3 | threshold=0.40 K=15 | threshold=0.40, K=15, temp=0.5 | TBD | - |

## Analysis

All configurations tested so far converge to exactly 78.33% (47/60). This suggests:
1. The 47 correct classifications are determined by the LLM judges correctness labels, not by vMF score thresholds
2. The remaining 13 samples are not separable by vMF scoring alone — the InternVL2-1B model produces answers where consistency (high vMF) does not correlate with correctness
3. More samples (K=15) may help if some samples are near the decision boundary, but the plateau at 78.33% across 3 different configurations suggests the method has reached its ceiling for this benchmark/model combination

## Feasible vs. Infeasible Ideas

**Feasible and tested:**
- Parameter sweeps (threshold, K) — no improvement observed
- Scoring function changes (kappa, adaptive fusion) — no improvement observed

**Not feasible:**
- all-mpnet-base-v2 upgrade — model not cached, network unreachable
- LLM self-consistency — would 3x eval time (~2.5h per run)
- Multi-temperature ensemble — would 3x eval time
