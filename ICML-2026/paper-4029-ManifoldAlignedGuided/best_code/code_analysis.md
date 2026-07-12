# Code Analysis — MA-GIG (Paper 4029) SOTA Optimization

## Architecture Overview

The evaluation pipeline flows:
1. `scripts/diffid.py` — Main entry point, Hydra config
2. `LatentGIGExplainer.get_attributions()` → `LatentGuidedPathGenerator.get_paths()` → VAE latent space guided path generation
3. `compute_ig()` in `cleanig/explainer/ig.py` — Riemann sum over path for attributions
4. `compute_diffid_score()` in `cleanig/metric/diffid.py` — Deletion/Insertion curves → DiffID

## Key Files

| File | Role | Safe to Modify? |
|------|------|-----------------|
| `configs/latent_gig.yaml` | Default config (Hydra) | Yes — params only |
| `scripts/diffid.py` | Eval entry point, metric parsing | Yes — structure only, NOT metric computation |
| `cleanig/explainer/ig.py` | `compute_ig()`, `get_grads()` | Yes — numerical integration logic |
| `cleanig/explainer/path_utils.py` | `LatentGuidedPathGenerator.get_paths()` | Yes — path construction logic |
| `cleanig/explainer/latent_gig.py` | `LatentGIGExplainer` wrapper | Yes — orchestrator |
| `cleanig/metric/diffid.py` | DiffID metric computation | NO — metric definition |
| `cleanig/vae_wrapper.py` | VAE loading/encoding/decoding | Safer to avoid |
| `cleanig/dataset/` | Dataset loading | NO |

## Evaluation Command
```
python3 scripts/diffid.py --config-name=latent_gig dataset=imagenet model_name=inception vae_type=mar fraction=0.05 use_slerp=false max_eval_samples=500
```

## Metric Parsing
- stdout "BENCHMARK RESULTS" section: `DiffID: X.XXXX`, `Insertion AUC: X.XXXX`, `Deletion AUC: X.XXXX`
- Also saved to `{save_dir}/metrics.json` with keys `diffid`, `insertion_auc`, `deletion_auc`

## Baseline Metrics
- DiffID: 0.356, Ins: 0.4409, Del: 0.0849

## Red-Line Boundaries
- **DO NOT MODIFY**: `cleanig/metric/diffid.py`, dataset loading logic, test splits, labels, scoring scripts
- **DO NOT**: Hard-code metrics, predictions, or benchmark outputs
- **DO NOT**: Change evaluation protocol

## Critical Code Paths

### Riemann Sum (ig.py:38)
```python
deltas = paths[:, 1:] - paths[:, :-1]
grads_for_deltas = grads[:, :-1]
attributions = (deltas * grads_for_deltas).sum(dim=1)
```
This is the LEFT-POINT Riemann sum. Change to trapezoidal: `(deltas) * (grads[:,:-1] + grads[:,1:]) / 2.0`

### Endpoint Anchoring (path_utils.py:385-393)
Endpoints use raw pixel values, not VAE-decoded. Creates discontinuity.

### Fraction Threshold (path_utils.py:438-444)
```python
threshold = torch.quantile(abs_grad.reshape(-1), self.fraction, interpolation="lower")
s = (torch.abs(grad) <= threshold) & (grad != float("inf"))
```

## Pre-downloaded Resources
- ImageNet val set: `/datasets/imagenet2012/val/` (ImageFolder format, zero-padded class dirs)
- MAR VAE checkpoint: from HuggingFace `jadechoghari/mar`, converted to `.ckpt`
- Imagenet classifier: torchvision pretrained inception_v3

## Manifest Recovery Notes
- Host `eval_command` passes through container; works directly as `cd /repo && python3 scripts/diffid.py ...`
- Container uses `/workspace` as default CWD; explicitly `cd /repo` before all commands
