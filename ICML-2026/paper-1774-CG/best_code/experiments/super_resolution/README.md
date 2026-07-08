# 4x ImageNet super-resolution (paper §6.3a)

Calibrated Bayesian Guidance for 4x super-resolution (256x256 from 64x64) on
ImageNet-val with a pixel mean-flow (pMF) prior. Each flow-matching step draws
`K` candidate reconstructions from the one-step pMF posterior and re-weights
them by the gradient-free CBG estimator against the Gaussian SR likelihood
(Eq. 20); no likelihood gradient is computed.

The SR core (bicubic 4x downsampler, Gaussian likelihood, pMF adapter, and the
single-image guided sampler) is vendored verbatim under
`pixel_space_inverse_problems/`; this driver reuses it and the shared
`calibrated_guidance` library, and logs to the unified W&B project.

## Setup

```bash
pip install -e ".[superres]"                  # fire, piq, Pillow, wandb, ...
bash experiments/super_resolution/download.sh # clones pMF repo + pMF-B-16.pt
```

`download.sh` clones the public pMF inference repo
([Lyy-iiis/pmf](https://github.com/Lyy-iiis/pmf)) into
`pixel_space_inverse_problems/external/pMF/` (where the adapter expects it) and
fetches the `pMF-B-16.pt` checkpoint from HuggingFace
([Lyy0725/pMF](https://huggingface.co/Lyy0725/pMF)) into `checkpoints/`.

ImageNet-val-256 is **user-provided**: pass `--val-root <dir>` pointing at an
ImageNet validation set laid out for `StratifiedValSubsample` (a `raw/` dir of
`ILSVRC2012_val_0000XXXX.JPEG` files and/or a `val_synsets.txt` manifest, or an
indexed `--val-tar-path`).

## Run

```bash
# Full stratified 1-per-class run (1000 images), K=256, 32 outer steps:
python run.py super-resolution --val-root <imagenet-val-256> \
    --num-particles 256 --num-outer-steps 32 --subsample-size 1000

# Smoke test (1 image):
python run.py super-resolution --val-root <imagenet-val-256> --limit 1

# Shard a range across jobs:
python run.py super-resolution --val-root <imagenet-val-256> \
    --start-index 0 --end-index 250
```

`python run.py super-resolution --help` lists all flags.

Key flags:

- `--num-particles` (K, candidate count per step — the main quality/compute
  knob; default 256).
- `--num-outer-steps` (flow-matching steps; default 32).
- `--noise-std` (Gaussian SR likelihood std; default 0.05).
- `--cfg-omega` (pMF classifier-free guidance; default 7.5).
- `--estimator {reinforce,reparam}` (gradient-free default).
- `--guidance-scale` (likelihood temperature; default `None` -> adaptive
  per-step scaling).
- `--start-index` / `--end-index` / `--num-images` / `--limit` (image range and
  smoke-test caps).
- `--subsample-size` (stratified 1-per-class subset size; default 1000).

## Output & metrics

Per image, the driver saves `gt/`, `lr/`, and `sr/` PNGs and appends a row to
`<output-dir>/per_image_metrics.csv`. Per-image PSNR / LPIPS / seconds are
logged to the unified W&B project (`calibrated-guidance`, group
`super_resolution`) under `image/{psnr,lpips,seconds}`, and the run-level means
under `final/{psnr,lpips}`. LPIPS and PSNR use [`piq`](https://github.com/photosynthesis-team/piq).

## Attribution

The pixel mean-flow prior is from the public pMF project:

- Code: [github.com/Lyy-iiis/pmf](https://github.com/Lyy-iiis/pmf)
- Checkpoints: [huggingface.co/Lyy0725/pMF](https://huggingface.co/Lyy0725/pMF)
- Paper: arXiv:2601.22158
