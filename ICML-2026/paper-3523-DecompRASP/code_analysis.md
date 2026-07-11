# Code Analysis — Paper 3523 SOTA Preparation Repair

## Original Preparation Failure

The preparation script attempted to install `git` via `apt-get` inside the container but failed because:
1. The proxy settings (`HTTP_PROXY=http://127.0.0.1:7890`) caused 502 Bad Gateway errors from `archive.ubuntu.com`
2. Without git, the baseline commit and tag could not be created

## Repair

**Fix:** Install git with proxy unset:
```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
apt-get update -qq && apt-get install -y -qq git
```

**Additional setup:**
- Created `/tools/` directory and copied `record_score.sh` from host
- Created `/autosota_artifacts/paper-3523/sota/` directory
- Initialized git repo at `/repo` with baseline commit and `_baseline` tag

## Corrected Evaluation Command

```bash
cd /repo/crasp/scripts/patching
python3 eval_unique_copy.py --model_path /repo/share/saved_models/unique_copy-@2l1h64d3lr01drop
```

Runs inside container `autosota_sota_paper_3523`. No Docker exec or host paths needed.

## Baseline Verification

| Metric | Manifest | Reproduced |
|--------|----------|------------|
| Task Acc in < 50 | 100.0% | 100.0% ✓ |
| Task Acc in [51-100] | 100.0% | 100.0% ✓ |
| Task Acc in [101-150] | 99.7% | 99.7% ✓ |

All three metrics match the reproduction manifest exactly.

## Optimization Strategy

The model is a tiny GPT-2 decoder-only transformer (2 layers, 1 head, 64 dim, 129K params) that trains in ~47 seconds. This enabled rapid iteration with 9 non-baseline experiments.

### Key Finding: Training Schedule is the Critical Lever

The single most impactful change was improving the training schedule:
- **warmup_steps=500** (from 0)
- **lr_scheduler_type="cosine"** (from linear decay)
- **patience_steps=1500** after in-distribution saturation (from immediate early-stop)
- **max_steps=6000** (from 30000, but early-stop at 4500 with patience)

This improved OOD [101-150] from 99.7% → **100.0%**, with zero regression on guardrail metrics.

### Multi-Seed Stability

Training with seeds {0, 1, 2, 42, 123} showed:
- 4/5 seeds achieve 100.0% OOD
- Seed 1 achieves 99.7% (equal to baseline)
- Mean OOD: 99.94%, Std: 0.13%

### What Did NOT Help

1. **Position offset randomization** (Idea-02): Catastrophic OOD collapse (0.2%) — the wider offset range confused the induction head
2. **Multi-head architecture** (Idea-05): 99.5% OOD — worse than 1-head, consistent with paper findings
3. **Position coupling** (Idea-04): Severe train-eval mismatch, collapses to 2.9% under standard eval

### Robustness Confirmed

- **Weight decay:** All values [0.0, 0.001, 0.01, 0.05, 0.1] achieve 100% with improved schedule
- **Learning rate:** 1e-3 and 2e-3 both achieve 100%; 5e-4 gets 99.9%; 2e-4 fails to converge
- **Curriculum lengths:** Training on 1→80 (vs 1→50) also achieves 100%, confirming schedule improvement is the key
- **Length sampling:** Triangular distribution (oversampling long sequences) achieves 100%

## Safe Optimization Targets

The training hyperparameters in `train_unique_copy_eval.py` are the safest and most impactful targets:
- `warmup_steps`, `lr_scheduler_type`, `max_steps`, patience in `EvalCallback`
- `weight_decay`, `lr`, `dropout`
- Architecture: `n_layer`, `n_head`, `d_model`

Data generation changes in `patching_data.py` (offset ranges, position encoding) are high-risk and likely to degrade OOD performance.

## Remaining Risks

1. The 100% OOD result was achieved on the baseline model checkpoint — need to confirm it generalizes to new random seeds consistently
2. The evaluation uses 2000 test samples per range; with 100% accuracy, there may be edge cases at the boundary
3. The model is deterministic given the same seed; further diversity testing may reveal fragile cases
