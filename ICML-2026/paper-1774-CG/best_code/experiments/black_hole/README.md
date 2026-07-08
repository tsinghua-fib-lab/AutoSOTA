# Black-hole imaging (paper §6.2)

Calibrated Bayesian Guidance for radio-interferometric black-hole imaging on the
[InverseBench](https://github.com/devzhk/InverseBench) task — a new
state-of-the-art PSNR (Table 2). The estimator draws K candidate reconstructions
from `p(x0|xt)` each denoising step and re-weights them by the gradient-free CBG
estimator (Eq. 20); no likelihood gradient is computed.

Two samplers for `p(x0|xt)`:

* **`--posterior meanflow`** (fast) — a one-step mean-flow "renoise" sampler
  (paper §5.3). Needs a mean-flow checkpoint.
* **`--posterior ddpm`** (slow, original) — a DDPM/EDM inner loop. Needs only the
  diffusion prior. This is the `+ DDPM` row of Table 2.

## Setup

```bash
pip install -e ".[blackhole]"        # ehtim, hydra, omegaconf, piq, ...
bash experiments/black_hole/download.sh   # diffusion prior + test data
```

The engine (forward operator, scheduler, dataset, evaluator) is the vendored
`third_party/InverseBench`; no extra install needed.

## Run

```bash
# Fast mean-flow renoise (recommended), K=128, 100 outer steps:
python run.py black-hole --posterior meanflow --mf-ckpt <ckpt.pkl> \
    --reinf-k 128 --num-steps 100 --id-list 0-99

# Slow DDPM inner loop (original), K=512:
python run.py black-hole --posterior ddpm --reinf-k 512 \
    --num-steps 100 --inner-loop-steps 25 --id-list 0-99

# Smoke test (1 image, tiny K):
python run.py black-hole --posterior ddpm --reinf-k 4 --num-steps 20 --limit 1
```

Key flags: `--reinf-k` (candidates K per step — the main quality/compute knob,
paper sweeps 4 … 65536), `--num-steps` (outer SDE steps), `--guidance-scale`
(default `3e-3`), `--estimator {reinforce,reparam}` (gradient-free default).
`python run.py black-hole --help` lists all.

**Defaults reproduce Table 2.** The defaults match the paper's best config:
`--stop-at-sigma 0.05` (truncate the noisy SDE once σ≤0.05) + `last_hop` (one
final deterministic guided step) + no clamping. These denoise the final sample
and are what reaches the reported PSNR — omitting them costs ~0.6 dB. Pass
`--stop-at-sigma 0` / `--no-last-hop` / `--clamp` to disable. Reproduced with the
non-EMA mean-flow checkpoint: K=128 ≈ 25.4 dB (paper 25.36).

## Output & metrics

Per-instance results are saved to `--output-dir` and metrics (PSNR, blurred PSNR
at f=10/15/20, closure-phase χ², log-closure-amplitude χ²) are logged to the
unified W&B project (`calibrated-guidance`, group `black_hole`) and printed.

## Provenance

The estimators (`algo/reinforce.py`, `algo/meanflow_posterior.py`, the slow
`algo/dps.py` REINFORCE path) are our additions on top of public InverseBench;
this clean driver reuses the engine and the shared `calibrated_guidance` library
without modifying any vendored file. See
[`third_party/InverseBench/UPSTREAM_DIFF.md`](../../third_party/InverseBench/UPSTREAM_DIFF.md).
