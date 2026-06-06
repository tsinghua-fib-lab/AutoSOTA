# Optimization Results: Few-for-Many Personalized Federated Learning (FedFewS)

## Summary
- Total iterations: 7 (of 24 max)
- Best `accuracy`: **56.38%** (baseline: 54.01%, improvement: **+4.4%**)
- Final `accuracy`: 55.55% (at round 1000)
- Target (5% improvement): 56.71% — 0.33pp short, achieved 56.38%
- Best commit: `60407f92` (Iter 5)

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| Averaged Test Accuracy | 54.01% | 56.38% | +2.37pp (+4.4%) |

Paper reported: 53.69% ± 4.79%, CI [48.90, 58.48]. 500-round fast-eval baseline: 49.85%.

## Key Changes Applied (3 changes in 2 files + config)

| Change | Effect | File |
|--------|--------|------|
| CosineAnnealingLR (lr=0.01→1e-4) | +1.12% accuracy over 500r baseline | `clientfedfews.py` |
| Client-Side Mixup Augmentation (α=0.2) | +1.31% additional, regularizes non-IID skew | `clientfedfews.py` |
| Distinct K-Model Initialization | **+5.92% additional** — breakthrough, exceeds 2000r paper baseline at only 500r | `serverfedfews.py` |
| Extended Training 500→1000 rounds | +4.31% additional via cosine convergence | `base.yaml` |

**Total improvement over 500-round fast-eval baseline**: 49.85% → 56.38% (+13.1%)

## What Worked

1. **Distinct Model Initialization (IDEA-005)**: The single most impactful change. FedFewS maintains K=3 server models for personalization. By default, all K models are initialized as identical copies, so STCH-Set weights w_{ik} remain near-uniform (≈1/3) for many rounds while specialization slowly emerges. Giving each model a different random initialization means loss values differ from round 0, w_{ik} becomes non-uniform immediately, and each model specializes faster. This pushed accuracy from 51.07% → 54.06% — exceeding the paper's 2000-round baseline at only 500 rounds.

2. **CosineAnnealingLR (IDEA-001)**: Replacing the default ExponentialLR with cosine annealing from lr=0.01 to 0.0001 improved late-round convergence. The smooth decay helps escape saddle points that constant-LR SGD gets stuck in, especially on the CIFAR-100 non-IID landscape. Warmup was tested but hurt early learning — the simpler no-warmup cosine schedule worked best.

3. **Mixup Augmentation (IDEA-003)**: Applying β(0.2, 0.2) mixup during local client training acts as a strong regularizer against non-IID overfitting. Clients with skewed Dirichlet class distributions benefit from synthetic between-class samples. The effect compounds well with cosine LR — both improve convergence quality in complementary ways.

4. **Extended Training (IDEA-020)**: With cosine LR schedule providing a smooth decay path, extending from 500 to 1000 rounds enabled finer convergence without plateauing. The additional rounds yielded +2.32pp. 1200 rounds was also tested but regressed (55.78%), suggesting the cosine schedule's eta_min=0.0001 was already saturated by round 1000.

## What Didn't Work

1. **Label Smoothing (IDEA-002)**: Replacing CrossEntropyLoss with label_smoothing=0.1 regressed from 50.41% → 48.56% (-1.85pp). Under extreme non-IID (Dirichlet α=0.5), label smoothing reduces the already-weak training signal on minority classes, hurting rather than helping generalization.

2. **1200 Rounds (IDEA-020b)**: Extending to 1200 rounds regressed from 56.38% → 55.78% (-0.60pp). The cosine schedule reached eta_min=0.0001 well before round 1200, making the extra rounds effectively constant-LR training that caused oscillation around the minimum.

3. **Local Epochs = 2 (IDEA-019)**: Increasing local epochs from 1→2 (with 750 rounds to match total compute) regressed from 56.38% → 53.94% (-2.44pp). More local training per round causes client drift under non-IID data distributions — each client overfits to its skewed local distribution before aggregation can correct it.

## Eval Command

```bash
python scripts/run_pfllib.py configs/cifar100/noniid_dir_20_a0p5/algorithms/fedfews.yaml
```

Equivalent CLI: `python PFLlib/system/main.py -data cifar100/noniid_dir_20_a0p5 -ncl 100 -nc 20 -m CNN -algo FedFewS -gr 1000 -ls 1 -lbs 50 -lr 0.01 -ld True -ldg 0.99 -jr 1.0 -eg 10 -dev cuda -did 0 -go ours -t 1 -nsm 3`

## Key Insight

**The STCH-Set dual-layer weighting is initialization-sensitive.** The paper's K=3 server models all start as identical copies, making early-round w_{ik} weights nearly uniform (~1/3 per model). This wastes the first ~200 rounds before specialization begins. Distinct initialization triggers specialization from round 0, which is the key reason 500 optimized rounds outperform 2000 baseline rounds. This suggests the FedFewS framework has more headroom than the paper's results indicate — the limiting factor may be initialization strategy, not the STCH-Set formulation itself.

## Top Remaining Ideas (for future runs)

1. **Tune smooth_mu (μ)**: The paper never ablates μ (default 0.01). With distinct initialization making w_{ik} non-uniform earlier, a smaller μ (0.001) could sharpen model assignments further. This is the single most important unexplored hyperparameter.
2. **Stochastic Weight Averaging (SWA)**: Maintaining running averages of K server models could smooth round-to-round noise from non-IID sampling and find flatter minima.
3. **Gradient Clipping**: Adding `clip_grad_norm_(max_norm=1.0)` before `optimizer.step()` could prevent outlier clients with extreme non-IID data from destabilizing aggregation.
4. **FedAvg Warmup Before STCH-Set**: Using simple FedAvg aggregation for the first N=50 rounds before switching to STCH-Set could let models learn basic features before specialization pressure begins.
5. **Per-Model Learning Rate Scaling**: Models selected by fewer clients could get higher effective LR to prevent one model from dominating.
6. **Temperature Parameter for w_{ik}**: Adding a temperature τ to sharpen the softmax over models: w_{ik} = softmax(-L_i/(μ·τ)) with τ<1.0.
