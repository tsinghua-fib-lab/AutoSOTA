# Paper 9002 — reproduction record (round-2 baseline)

**Status: REPRODUCTION SUCCEEDED** — bit-identical to the registered round-1 best state.

## Exact evaluation command (run from /repo)

```
python pipeline/run_pipeline.py --out-dir outputs && python /paper_data/eval/score_oof.py outputs/oof_predictions.csv
```

Cold run (after `rm -rf outputs`) → 50.6 s wall, CPU only, exit 0.

## Reproduced metrics (official scorer, `F1=` last line)

| metric | measured | registered | delta |
|---|---|---|---|
| F1 (fold-mean, weighted) | **0.7831** | 0.7831 | 0.0000 |
| F1_SE | 0.0106 | 0.0106 | 0.0000 |
| F1_overall (pooled) | 0.7778 | 0.7778 | 0.0000 |
| LGB_tabular_F1 (guardrail) | 0.7623 | 0.7623 | 0.0000 |
| CNN_ensemble_F1 (guardrail) | 0.7254 | 0.7254 | 0.0000 |
| per-fold | [0.7838, 0.7833, 0.8204, 0.7727, 0.7554] | same | exact |

Rubric acceptance band was [0.7725, 0.7937] → **inside**, above the lower bound.

Integrity: `sha256(ensemble_outputs.npz)=36291c39f2c3040e…` (recorded every run in
`outputs/metrics.json` as `cnn_integrity.npz_sha256`); `sha256(cnn_oof.f32)=004a2251c689838f…`.

Coverage/format: `outputs/oof_predictions.csv` = 6256 rows (`x,y,pred`);
`outputs/submission.csv` = 98 rows, passes `validate_submission.py`.
Submission class counts (ALG 26 / FMAT 18 / NVB 39 / SGAM 9 / SGZ 6) differ slightly from the
registered lock candidate (ALG 25 / NVB 40). That field is informational: the F1 and both
component metrics reproduce exactly, and the paper documents that the rare/tail counts swing
between candidate recipes.

## Determinism

Two independent default runs produced **byte-identical** `oof_predictions.csv` and
`submission.csv` (`diff -q` clean). LightGBM thread count does not change the fitted model.

## Escape hatches verified live (independent cross-checks)

```
python pipeline/run_pipeline.py --out-dir /tmp/run_ref --min-child-samples 20 --tab-sharpen 1
```

→ LGB_tabular 0.7519 ± 0.0238, blend **0.7688 ± 0.0133**, per-fold
[0.7841, 0.7642, 0.8078, 0.7606, 0.7274] — matches the published reference recipe and the
rubric `Baseline_Value` exactly. The repo therefore reproduces both ends of the round-1 range.

## Environment (container autosota_repro_paper_9002)

Base image `docker.1ms.run/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` (Python 3.10.13,
Ubuntu 20.04, 4× A100-80GB visible, 128 nproc, ~337 GB free on `/`).

- Pipeline deps (pinned by repo `requirements.txt`, Tsinghua pip mirror): pandas 2.1.4,
  scikit-learn 1.3.2, lightgbm 4.6.0, numpy 1.26.0.
- Added for downstream (SOTA) work, torch deliberately left untouched at 2.1.0 /
  torchvision 0.16.0: `git` 2.25.1 + `curl` (apt, Aliyun mirror; original sources.list kept
  at `/etc/apt/sources.list.orig.bak`), `timm==0.9.16`, `rasterio==1.3.9`, and
  `huggingface_hub==0.24.6`.
- **The hf_hub pin is load-bearing**: pip resolves `huggingface_hub` to 2.x by default, which
  removed `repo_type_and_id_from_hf_id`; `timm` 0.9.16 imports that symbol, so a silent
  ImportError makes timm believe HF hub is absent and every `pretrained=True` call fails.
- CNN Path A is enabled and smoke-tested: `timm.create_model(name, pretrained=True,
  num_classes=5)` loads for resnet18, efficientnet_b0 and convnext_tiny on 96×96×3 input with
  `HF_ENDPOINT=https://huggingface.co` (weights land under `/autosota_cache/hf/hub`).
  The first HEAD request is intermittently flaky behind the corporate proxy — retry once.
  Both `https://huggingface.co` and `https://hf-mirror.com` resolve from this container.
- **Known-bad download**: `torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")`
  fetches `efficientnet_b0_rwightman-3dd342df.pth` from download.pytorch.org and fails the
  built-in hash check twice in a row (expected `3dd342df`, got `7f5810bc…`). Download
  integrity through this proxy is broken for that URL; take the B0 backbone from `timm`
  (verified working) or use `weights=None`.

## Container hygiene note (relevant to future papers)

`/repo` as handed over contained untracked leftovers from **paper 6078** (`OLR-*.ipynb`,
`Wine-Data-Experiment.ipynb`, `Drone-Experiment.ipynb`, `Figures/`, `LICENSE`, `work_6078/`).
The host checkout `/home/dataset-assist-0/lxy/autosota-v2.6/workdir/paper-9002/repo` is clean,
so the carrier is the **base image tag**: the `docker.1ms.run/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
image (08f930a16ddf, built 4 days ago) already carried the previous paper's `/repo` snapshot, and
this run copied paper-9002 files over it without deleting. The leftovers were moved (not deleted)
to `/autosota_cache/quarantine/paper6078-leftovers/`; `git status` in `/repo` is now clean at
commit `00ce4d1590c05b17e37ba304fe897e03b63c2b1e`.

## Levers available to the SOTA phase (all offline; CPU-cheap unless noted)

- The pipeline already dumps every intermediate probability matrix into `outputs/`
  (`tab_oof.npy`, `cnn_oof.npy`, `blend_oof.npy`, `tab_test_proba.npy`,
  `cnn_test_proba.npy`, `blend_test_proba.npy`, `blend_test_proba_adj.npy`), so fusion-operator
  and decision-bias searches cost milliseconds per candidate and need no retraining.
- The only live-trained branch is LightGBM (87 features: 86 v07 + `acr_sapa_w21`); the CNN is
  the frozen vendored npz. Any feature-side change is therefore measurable in ~50 s per candidate.
- Fold-honest (LOO) selection is mandatory for anything tuned on OOF (paper §6): round 1 measured
  +0.0037 honest versus +0.0143 naive for the fusion temperature.
- Runtimes: full eval command 50 s; a single LightGBM 5-fold + full refit is most of that.
