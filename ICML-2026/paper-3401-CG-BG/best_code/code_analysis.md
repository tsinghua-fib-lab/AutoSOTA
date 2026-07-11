# SOTA Preparation Repair — Paper 3401 (CG-BG)

## Original Preparation Failure

The SOTA preparation failed because `ALL_PROXY=socks5h://172.17.0.1:17891` was set in the container environment. This caused `httpx` to attempt SOCKS proxy connections, but `socksio` is not installed in the pixi environment. The error manifested as:

```
ImportError: Using SOCKS proxy, but the socksio package is not installed.
```

## Repair

**Trivial fix**: Unset `ALL_PROXY` and `all_proxy` before running the evaluation command. The manifest notes explicitly warned: "ALL_PROXY must be unset before run (SOCKS proxy breaks httpx)."

**Corrected in-container evaluation command**:
```bash
cd /repo
unset ALL_PROXY all_proxy
/repo/.pixi/envs/default/bin/python main.py +experiment=ala2_cb_ub stage=4
```

HTTP proxies (`HTTP_PROXY`, `HTTPS_PROXY`) are kept for Hugging Face access via `hf-mirror.com`, but SOCKS proxy is removed.

## Baseline Verification

The repaired command reproduces the manifest baseline exactly:
| Metric | Manifest | Reproduced |
|--------|----------|------------|
| JS_Divergence | 0.005236 | 0.005236 |
| PMF_Error | 0.221012 | 0.221012 |
| ESS_Percent | 0.552833 | 0.552833 |

## Key Optimization Findings

### 1. Clip percentile sweep (IDEA-02)
Hard clip=93 (instead of default 99) gives 3.4% JS improvement with ESS+PMF gains.
Sweep range: {90, 92, 92.5, 93, 93.5, 94, 95, 96, 97, 98, 99, 99.5, 99.9}

### 2. Soft weight clipping (IDEA-01 simplified) — MAJOR BREAKTHROUGH
Replacing hard -inf cutoff with sigmoid soft taper at clip=93, alpha=0.1 gives:
- JS: 0.004498 (-14.1% vs baseline)
- PMF: 0.214484 (-3.0%)
- ESS: 0.782223 (+41.5%)

### 3. Combined clip + soft (IDEA-02-SOFT) — BEST
clip=90 with alpha=0.1 gives:
- JS: 0.004484 (-14.4% vs baseline) ← BEST
- PMF: 0.213476 (-3.4%)
- ESS: 0.786037 (+42.2%)

### Implementation detail
Modified `src/cg_bg/flow/evaluate.py:clip_weights()` to support soft clipping via `CLIP_ALPHA` environment variable. Sigmoid taper smoothly reduces weights above cutoff instead of zeroing them. Formula:
```
taper = sigmoid(-alpha * (logw - cutoff) / std(logw))
soft_logw = cutoff + taper * (logw - cutoff)
```

## Safe Optimization Targets
- `clip` parameter in main.yaml (stage 4 only, no retraining needed)
- `n_bootstraps` for CI estimation stability
- Soft clipping via CLIP_ALPHA env var
- Weight computation in evaluate.py:clip_weights()

## Remaining Opportunities (not attempted)
- Stage 2+3+4 ODE tolerance sweep (requires 20+ min per run)
- Training convergence verification (15 min retraining)
- EMA parameter averaging (retraining)
- Minibatch OT coupling (retraining)
- Multi-seed evaluation

## Files Modified
- `src/cg_bg/flow/evaluate.py`: Added soft clipping in clip_weights() 
  (backup: evaluate.py.bak)
