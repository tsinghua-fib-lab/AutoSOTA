#!/usr/bin/env python
"""Publish the low-resource (en/zh binary) CLD convex heads to the HF Hub.

These are the whisper-small low-resource experiments: the same convex ReLU-MLP
language-detection head trained on frozen whisper-small encoder embeddings for
the binary en/zh task at four training-set sizes (100 / 500 / 1000 / 10000
samples per class).

All four are uploaded into a single combined repo, one subfolder per config,
with a shared model card. The card uses the actual numbers from our runs.

    python scripts/push_lr_models_to_hf.py            # push
    python scripts/push_lr_models_to_hf.py --dry-run  # write card locally only
"""
import argparse
import os

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMESPACE = "williamhtan"
REPO = "cld-whisper-small-enzh"
BASE = "openai/whisper-small"
SRC_PKL = "model.pkl"  # name on disk
PKL_NAME = "model.pkl"  # name on the Hub
LANGUAGES = ["en", "zh"]

# Actual values from our runs.
#   acc/wer/cer/test_n   benchmark_cld.py on the held-out test split
#   val_peak             best test-split accuracy during CRONOS training (cronos_results.txt)
#   data_seed/train_time/tflops   from cronos_results.txt
CONFIGS = [
    # n,    acc,   wer,    cer,    val_peak, data_seed, test_n, train_time, tflops
    (100,   1.0000,  42.11,  38.17,  1.000,    3,    20,    36.1,   "8,340"),
    (500,   0.9900,  28.03,  36.30,  1.000,    1,    100,   50.3,   "11,623"),
    (1000,  0.9900,  31.68,  31.22,  0.985,    6,    200,   64.4,   "14,871"),
    (10000, 0.9887,  27.67,  28.47,  0.989,    6,    1860,  310.1,  "71,642"),
]

FIG_IMG = "assets/fig_1_2.png"


def src_pkl(n):
    return os.path.join(ROOT, f"data/cld-whisper-small-enzh-{n}/{BASE}/{SRC_PKL}")


def build_card():
    rows = []
    for n, acc, wer, cer, vp, seed, tn, tt, tflops in CONFIGS:
        rows.append(
            f"| {n} | {acc:.2f} (n={tn}) | {wer:.2f} | {cer:.2f} | {vp:.3f} | {seed} | {tt:.1f} | {tflops} |"
        )
    results_table = "\n".join(rows)

    fm = (
        "---\n"
        "license: mit\n"
        "language:\n- en\n- zh\n"
        "pipeline_tag: audio-classification\n"
        "library_name: jaxcld\n"
        f"base_model: {BASE}\n"
        "tags:\n"
        "- language-identification\n- language-detection\n"
        "- spoken-language-identification\n- speech\n"
        "- automatic-speech-recognition\n- whisper\n- convex-optimization\n"
        "- jax\n- admm\n- cronos\n- low-resource\n- accent-robust\n"
        "metrics:\n- accuracy\n- wer\n- cer\n"
        "---\n"
    )

    body = f"""\
# CLD — Low-Resource Convex Language-Detection Heads (whisper-small, en/zh)

**Convex Low-resource Accent-Robust Language Detection (CLD)** heads for the
**binary English/Chinese** spoken language detection task, trained on frozen
[`{BASE}`](https://huggingface.co/{BASE}) encoder embeddings. This repo contains the
**low-resource sweep**: the same convex ReLU-MLP head trained at four training-set
sizes — **100, 500, 1000, 10000** samples per class — to study data efficiency.

[![paper](https://img.shields.io/badge/paper-ICML%202026-blue.svg)](https://arxiv.org/abs/2605.23235)
[![code](https://img.shields.io/badge/code-GitHub-181717.svg?logo=github)](https://github.com/pilancilab/CLD)
[![pypi](https://img.shields.io/badge/pip-jaxcld-3775A9.svg?logo=pypi&logoColor=white)](https://pypi.org/project/jaxcld/)

## Model description

Each artifact is a **two-layer convex ReLU MLP** spoken language detection head trained
on **mean-pooled frozen `{BASE}` encoder embeddings** (dim 768). It is trained by solving
a convex reformulation with **CRONOS** (ADMM in JAX) rather than standard NN training.
At inference it predicts the spoken language (en/zh) and selects the matching Whisper
language token before decoding. Each is a `CVX_ReLU_MLP` with `theta1 (2, 768, 128)`,
`theta2 (2, 128)`, `n_classes = 2`. Label order is the sorted ISO-639-1 codes:
`0`→`en`, `1`→`zh`.

## How to use

Loading the head requires JAX (the weights are JAX arrays):

```bash
pip install jaxcld jax
```

```python
import numpy as np
from huggingface_hub import hf_hub_download
from jaxcld import ASRModel, CVXNNLangDetectHead

languages = ["en", "zh"]
config = "1000"   # one of: 100, 500, 1000, 10000

asr = ASRModel.from_pretrained("{BASE}", config={{"languages": languages}})
head_path = hf_hub_download("{NAMESPACE}/{REPO}", f"{{config}}/{PKL_NAME}")
head = CVXNNLangDetectHead.load(head_path, asr)

asr.set_lang_detect_head(head)
audio_16k_mono: np.ndarray = ...   # shape (T,), 16 kHz mono
pred_langs, pred_texts = asr.predict(audio_16k_mono)
print(pred_langs[0], pred_texts[0])
```

## Contents

One convex head per training-set size, under a per-config subfolder:

```
{NAMESPACE}/{REPO}
├── 100/{PKL_NAME}
├── 500/{PKL_NAME}
├── 1000/{PKL_NAME}
└── 10000/{PKL_NAME}
```

## Results

Detection accuracy / WER / CER from `benchmark_cld.py` on the held-out test split for
**these exact artifacts**. Validation peak = best test-split accuracy during CRONOS
training. Training time and TFLOPs from the training run.

| Samples/class | Det. acc | WER (↓) | CER (↓) | Val peak | data_seed | Train time (s) | TFLOPs |
|---|---|---|---|---|---|---|---|
{results_table}

![Low-resource WER and detection accuracy vs. training-set size](fig_1_2.png)

The convex head stays at ~0.99–1.00 detection accuracy across *all* sample sizes,
including the 100-sample regime. The 100-sample test split is tiny (n=20), so its
WER/CER are higher-variance.

## Training

Trained with `train_cvxnn.py` (CRONOS / ADMM in JAX). Shared hyperparameters:
`rank=20, neuron=64, beta=0.001, rho=0.1, gamma_ratio=1, admm_iters=6, pcg_iters=32,
opt_seed=1024` (per-config `data_seed` in the table above). Inputs are mean-pooled
frozen `{BASE}` encoder embeddings (dim 768).

## Citation

```bibtex
@inproceedings{{feng2026cld,
  title     = {{Convex Low-resource Accent-Robust Language Detection in Speech Recognition}},
  author    = {{Feng, Miria and Tan, William and Pilanci, Mert}},
  booktitle = {{Proceedings of the 43rd International Conference on Machine Learning}},
  year      = {{2026}},
  series    = {{Proceedings of Machine Learning Research}},
  publisher = {{PMLR}},
  url       = {{https://arxiv.org/abs/2605.23235}}
}}
```

## License

MIT — see the [CLD repository](https://github.com/pilancilab/CLD).
"""
    return fm + "\n" + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    card = build_card()
    stage = os.path.join(ROOT, "data", "hf_upload", REPO)
    os.makedirs(stage, exist_ok=True)
    card_path = os.path.join(stage, "README.md")
    with open(card_path, "w") as f:
        f.write(card)
    print(f"[card] wrote {card_path} ({len(card)} chars)")

    for n, *_ in CONFIGS:
        assert os.path.isfile(src_pkl(n)), f"missing pkl: {src_pkl(n)}"

    if args.dry_run:
        print(f"[dry-run] would push {NAMESPACE}/{REPO} with configs "
              + ", ".join(str(c[0]) for c in CONFIGS) + " + README.md + fig_1_2.png")
        return

    api = HfApi()
    repo_id = f"{NAMESPACE}/{REPO}"
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md",
                    repo_id=repo_id, repo_type="model")
    fig = os.path.join(ROOT, FIG_IMG)
    if os.path.isfile(fig):
        api.upload_file(path_or_fileobj=fig, path_in_repo="fig_1_2.png",
                        repo_id=repo_id, repo_type="model")
    for n, *_ in CONFIGS:
        api.upload_file(path_or_fileobj=src_pkl(n),
                        path_in_repo=f"{n}/{PKL_NAME}",
                        repo_id=repo_id, repo_type="model")
        print(f"  uploaded {n}/{PKL_NAME}")
    print(f"[done] https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
