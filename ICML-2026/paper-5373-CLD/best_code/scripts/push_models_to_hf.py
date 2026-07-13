#!/usr/bin/env python
"""Publish the three 5-language CLD convex language-detection heads to the HF Hub.

Creates (or updates) one public model repo per backbone under the authenticated
user, uploads the trained head as ``model.pkl`` and a generated model card.

Auth: uses the ambient token from ``hf auth login`` (no token is hardcoded).
Idempotent: safe to re-run (``exist_ok=True``).

    python scripts/push_models_to_hf.py            # push all three
    python scripts/push_models_to_hf.py --dry-run  # write cards locally, skip upload

All metrics below are the actual values measured in our runs (training time and
TFLOPs from cronos_results.txt; detection accuracy / WER / CER from benchmark_cld.py).
"""
import argparse
import os
import textwrap

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMESPACE = "williamhtan"
PKL_NAME = "model.pkl"
LANGUAGES = ["en", "hi", "id", "ms", "zh"]
LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "id": "Indonesian",
    "ms": "Malay",
    "zh": "Mandarin Chinese",
}

# Per-backbone facts from our runs.
#   src               trained pkl on disk
#   det_acc/wer/cer   benchmark_cld.py on the held-out 5-lang test split (n=1546)
#   val_peak          best test-split accuracy during CRONOS training (cronos_results.txt)
#   data_seed/train_time/tflops   from cronos_results.txt
MODELS = [
    {
        "repo": "cld-whisper-small-5lang",
        "base": "openai/whisper-small",
        "base_tag": "whisper",
        "src": "data/cld-whisper-small-5lang/openai/whisper-small/model.pkl",
        "loader": "CVXNNLangDetectHead",
        "dim": 768,
        "theta1": "(5, 768, 128)",
        "data_seed": 6,
        "val_peak": "0.9799",
        "det_acc": "0.98",
        "wer": "48.23",
        "cer": "27.47",
        "train_time": "257.8",
        "tflops": "59,542",
    },
    {
        "repo": "cld-whisper-large-v3-5lang",
        "base": "openai/whisper-large-v3",
        "base_tag": "whisper",
        "src": "data/cld-whisper-large-v3-5lang/openai/whisper-large-v3/model.pkl",
        "loader": "CVXNNLangDetectHead",
        "dim": 1280,
        "theta1": "(5, 1280, 128)",
        "data_seed": 2,
        "val_peak": "0.9845",
        "det_acc": "0.98",
        "wer": "31.11",
        "cer": "19.81",
        "train_time": "454.2",
        "tflops": "104,931",
    },
    {
        "repo": "cld-mms-1b-5lang",
        "base": "facebook/mms-1b-all",
        "base_tag": "mms",
        "src": "data/cld-mms-1b-5lang/facebook/mms-1b-all/model.pkl",
        "loader": "CVXNNLangDetectHead",
        "dim": 1280,
        "theta1": "(5, 1280, 128)",
        "data_seed": 8,
        "val_peak": "0.9825",
        "det_acc": "0.96",
        "wer": "48.10",
        "cer": "23.47",
        "train_time": "739.2",
        "tflops": "170,744",
    },
]


def build_card(m):
    fm = textwrap.dedent(f"""\
    ---
    license: mit
    language:
    - en
    - hi
    - id
    - ms
    - zh
    pipeline_tag: audio-classification
    library_name: jaxcld
    base_model: {m['base']}
    tags:
    - language-identification
    - language-detection
    - spoken-language-identification
    - speech
    - automatic-speech-recognition
    - {m['base_tag']}
    - convex-optimization
    - jax
    - admm
    - cronos
    - low-resource
    - accent-robust
    metrics:
    - accuracy
    - wer
    - cer
    ---
    """)

    langs_list = "\n".join(f"  - `{c}` — {LANG_NAMES[c]}" for c in LANGUAGES)
    idx_map = ", ".join(f"`{i}`→`{c}`" for i, c in enumerate(LANGUAGES))
    adapter = "Whisper language token" if m["base_tag"] == "whisper" else "MMS language adapter"

    body = f"""\
# CLD — Convex Language-Detection Head for `{m['base']}` (5 languages)

**Convex Low-resource Accent-Robust Language Detection (CLD)** head for multilingual
speech recognition. A lightweight **convex ReLU-MLP** spoken-language classifier trained
on **frozen pooled encoder embeddings** from
[`{m['base']}`](https://huggingface.co/{m['base']}). At inference it performs
spoken language detection over 5 languages and selects the matching {adapter}
before decoding, improving downstream transcription accuracy.

[![paper](https://img.shields.io/badge/paper-ICML%202026-blue.svg)](https://arxiv.org/abs/2605.23235)
[![code](https://img.shields.io/badge/code-GitHub-181717.svg?logo=github)](https://github.com/pilancilab/CLD)
[![pypi](https://img.shields.io/badge/pip-jaxcld-3775A9.svg?logo=pypi&logoColor=white)](https://pypi.org/project/jaxcld/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/pilancilab/CLD)

## Model description

CLD attaches a small **spoken language detection (5-way)** head to the (frozen) encoder
of a pre-trained ASR model and uses it to pick the language before decoding. Instead of a
standard neural network, the head is a **two-layer convex ReLU MLP** trained by solving a
convex reformulation with **CRONOS** — an ADMM solver (with preconditioned conjugate
gradient / Nyström preconditioning) implemented in **JAX**. This yields a large training
speedup over a standard NN head while matching or exceeding its accuracy, and is
especially strong in **low-resource** and **accent-diverse** regimes.

- **Backbone (frozen):** `{m['base']}`
- **Task:** spoken language detection (5-way) → ASR language selection
- **Languages (5):**
{langs_list}
- **Class index → language:** {idx_map} (labels are the sorted ISO-639-1 codes)

## How to use

The head is loaded and run through the [`jaxcld`](https://pypi.org/project/jaxcld/)
package. Loading the artifact requires JAX (the weights are JAX arrays):

```bash
pip install jaxcld jax
```

```python
import numpy as np
from huggingface_hub import hf_hub_download
from jaxcld import ASRModel, CVXNNLangDetectHead

languages = ["en", "hi", "id", "ms", "zh"]

# 1) Load the frozen base ASR model
asr = ASRModel.from_pretrained("{m['base']}", config={{"languages": languages}})

# 2) Download + load this convex language-detection head
head_path = hf_hub_download("{NAMESPACE}/{m['repo']}", "{PKL_NAME}")
head = {m['loader']}.load(head_path, asr)

# 3) Attach and run
asr.set_lang_detect_head(head)
audio_16k_mono: np.ndarray = ...   # shape (T,), 16 kHz mono
pred_langs, pred_texts = asr.predict(audio_16k_mono)
print(pred_langs[0], pred_texts[0])
```

Pair the head with the **matching frozen base encoder** — embeddings are
backbone-specific, so a head is not transferable across backbones or to other languages.

## Architecture

The head consumes **mean-pooled** encoder hidden states `X ∈ ℝ^(B×{m['dim']})`
(pooled over time) and computes, per class, `logits = relu(X @ W1) @ W2`, then takes the
`argmax`. The pickle stores a `CVX_ReLU_MLP` object whose key tensors are:

| Tensor | Shape | Role |
|---|---|---|
| `theta1` | `{m['theta1']}` | first-layer (ReLU) weights, per class |
| `theta2` | `(5, 128)` | output-layer weights, per class |

Configuration: `n_classes = 5`, `P_S = 64` hyperplane samples (→ 128 ReLU units),
input dim `{m['dim']}`.

## Training

Trained with `train_cvxnn.py` (CRONOS / ADMM in JAX) on mean-pooled frozen
`{m['base']}` encoder embeddings from **Mozilla Common Voice**, accent-stratified across
the five languages (~12,368 train / ~1,546 validation pooled embeddings).

| rank | neurons | beta | rho | gamma_ratio | admm_iters | pcg_iters | opt_seed | data_seed |
|---|---|---|---|---|---|---|---|---|
| 20 | 64 | 0.001 | 0.1 | 1 | 6 | 32 | 1024 | {m['data_seed']} |

Training time **{m['train_time']} s**, estimated **{m['tflops']} TFLOPs**.

## Evaluation

Measured on the held-out 5-language test split (n=1546):

| Metric | Value |
|---|---|
| Detection accuracy | {m['det_acc']} |
| WER (↓) | {m['wer']} |
| CER (↓) | {m['cer']} |

Best validation accuracy during training: **{m['val_peak']}**.

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
    ap.add_argument("--dry-run", action="store_true",
                    help="Write cards to data/hf_upload/ but do not create/upload to the Hub.")
    args = ap.parse_args()

    api = HfApi()
    stage = os.path.join(ROOT, "data", "hf_upload")
    os.makedirs(stage, exist_ok=True)

    for m in MODELS:
        repo_id = f"{NAMESPACE}/{m['repo']}"
        card = build_card(m)
        card_dir = os.path.join(stage, m["repo"])
        os.makedirs(card_dir, exist_ok=True)
        card_path = os.path.join(card_dir, "README.md")
        with open(card_path, "w") as f:
            f.write(card)
        print(f"[card] wrote {card_path} ({len(card)} chars)")

        pkl_path = os.path.join(ROOT, m["src"])
        assert os.path.isfile(pkl_path), f"missing pkl: {pkl_path}"

        if args.dry_run:
            print(f"[dry-run] would push {repo_id}: {PKL_NAME} + README.md")
            continue

        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md",
                        repo_id=repo_id, repo_type="model")
        api.upload_file(path_or_fileobj=pkl_path, path_in_repo=PKL_NAME,
                        repo_id=repo_id, repo_type="model")
        print(f"[done] https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
