"""Push the CLD training datasets to the Hugging Face Hub (with dataset cards).

Mirrors the local on-disk datasets to public Hub dataset repos so that tests and
users can pull them with `datasets.load_dataset` instead of needing the (multi-GB)
local copies:

  * ``data/final``                          -> ``williamhtan/cld-multi-dataset``
        5-language (en/hi/id/ms/zh) speech, splits: train/valid/test.

  * ``data/lr_exp/{N}_config/dataset``      -> ``williamhtan/cld-enzh-dataset``
        en/zh binary speech at four sample budgets, one Hub *config* per N
        (``100``/``500``/``1000``/``10000``), each with train/valid/test.

After uploading the data, the matching dataset card (README.md) is generated and
pushed, preserving the auto-generated config/split metadata. Reference copies of
the cards are written under ``data/hf_upload/``.

Usage::

    python scripts/push_datasets_to_hf.py            # push data + cards
    python scripts/push_datasets_to_hf.py --dry-run  # print plan + write local cards
    python scripts/push_datasets_to_hf.py --only multi
    python scripts/push_datasets_to_hf.py --only enzh
    python scripts/push_datasets_to_hf.py --cards-only   # (re)push cards only
"""
from __future__ import annotations

import argparse
import os

from datasets import load_from_disk
from huggingface_hub import DatasetCard, DatasetCardData

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMESPACE = "williamhtan"
LICENSE = "cc-by-4.0"

MULTI_SRC = os.path.join(ROOT, "data", "final")
MULTI_REPO = f"{NAMESPACE}/cld-multi-dataset"

ENZH_REPO = f"{NAMESPACE}/cld-enzh-dataset"
ENZH_SIZES = [100, 500, 1000, 10000]

HF_UPLOAD_DIR = os.path.join(ROOT, "data", "hf_upload")


def _enzh_src(n: int) -> str:
    return os.path.join(ROOT, "data", "lr_exp", f"{n}_config", "dataset")


# --------------------------------------------------------------------------- #
# Dataset cards
# --------------------------------------------------------------------------- #

_PROVENANCE = """\
## Dataset description

We curate a dataset of multilingual voice transcriptions across high-resource
languages and their low-resource sub-dialects. As a primary source of
transcription data we use the **Common Voice (v23)** dataset (Ardila et al., 2020).
We supplement this with several additional dialect datasets for regional speech
variance:

- **Singaporean English** from the **National Speech Corpus (NSC)** — the first
  Singapore English corpus — provided through the Info-communications and Media
  Development Authority (IMDA) of Singapore. Singlish is selected because studies
  show it incurs particularly high error rates during voice transcription
  (Fong et al., 2002).
- The **Lahaja** dataset (Sanket et al., 2024), a benchmark comprising 12.5 hours
  of Hindi from 132 speakers across 83 Indian districts.

We normalize and augment all audio files via time stretching, volume gain, pitch
shift, and augmented background noise with **MUSAN** (Snyder et al., 2015).

## Schema

Each split is a [`datasets`](https://huggingface.co/docs/datasets) `Dataset` with
columns:

| column   | type                              | description                          |
|----------|-----------------------------------|--------------------------------------|
| `audio`  | `Audio(sampling_rate=16000)` mono | the speech clip, 16 kHz mono         |
| `text`   | `string`                          | reference transcription              |
| `lang`   | `string`                          | ISO-639-1 language code              |
| `accent` | `string`                          | accent / dialect label               |
"""


def _multi_card_body() -> str:
    return f"""\
# CLD — Multilingual (5-language) Speech Dataset

Speech dataset for **Convex Low-resource Accent-Robust Language Detection (CLD)**,
covering 5 languages chosen for a deliberately challenging classification
boundary. This is the **multiclass** division of the CLD data.

[![paper](https://img.shields.io/badge/paper-ICML%202026-blue.svg)](https://arxiv.org/abs/2605.23235)
[![code](https://img.shields.io/badge/code-GitHub-181717.svg?logo=github)](https://github.com/pilancilab/CLD)
[![pypi](https://img.shields.io/badge/pip-jaxcld-3775A9.svg?logo=pypi&logoColor=white)](https://pypi.org/project/jaxcld/)

{_PROVENANCE}

## Multiclass setup

For the multiclass classification task we select 5 languages: **English, Chinese,
Indonesian, Malay, Hindi**. This selection establishes a challenging classification
boundary, since these languages share linguistic and geographical proximity — such
regional influences often cause misidentification (e.g. Singaporean English is
frequently confused with Malay or Indonesian). To maintain a low-resource setting
we curate ~16,000 training samples across these 5 languages, incorporating 24 unique
accents (~3,200 samples per language, ~666 per accent), with an 80-10-10
train/test/validation split.

- **Languages (5):** `en` (English), `zh` (Chinese), `id` (Indonesian), `ms` (Malay), `hi` (Hindi)
- **Splits:** `train` / `valid` / `test`

## How to use

```python
from datasets import load_dataset

ds = load_dataset("{MULTI_REPO}")
print(ds)
sample = ds["test"][0]
print(sample["lang"], sample["text"])
audio = sample["audio"]          # {{"array": np.ndarray, "sampling_rate": 16000}}
```

## Citation

If you use this dataset, please cite the CLD paper (ICML 2026) and the underlying
corpora: Common Voice (Ardila et al., 2020), the National Speech Corpus (IMDA),
Lahaja (Sanket et al., 2024), and MUSAN (Snyder et al., 2015).
"""


def _enzh_card_body() -> str:
    cfgs = ", ".join(f"`{n}`" for n in ENZH_SIZES)
    return f"""\
# CLD — English/Chinese Binary Speech Dataset (sample-size ablation)

Speech dataset for **Convex Low-resource Accent-Robust Language Detection (CLD)**,
covering the **binary** English vs. Chinese division. Provided at four training
sample budgets as separate Hub **configs** for low-resource ablation studies.

[![paper](https://img.shields.io/badge/paper-ICML%202026-blue.svg)](https://arxiv.org/abs/2605.23235)
[![code](https://img.shields.io/badge/code-GitHub-181717.svg?logo=github)](https://github.com/pilancilab/CLD)
[![pypi](https://img.shields.io/badge/pip-jaxcld-3775A9.svg?logo=pypi&logoColor=white)](https://pypi.org/project/jaxcld/)

{_PROVENANCE}

## Binary setup

English and Mandarin are the two highest-resource languages in existing speech
datasets, yet still display some of the lowest accuracy in language prediction for
accented speech, due to the high variance of dialects and accents present in these
two languages. For example, Whisper-Small achieves 100% accuracy on Midwestern
English, drops to 91.8% on Wales-accented English, yet only 61.4% on Malaysian-
accented English. We select 5 regional dialects per language and perform ablation
studies on training sample sizes spanning 100 to 10,000 samples per language,
splitting training samples equally across all accents.

- **Languages (2):** `en` (English), `zh` (Chinese)
- **Configs (samples/language):** {cfgs}
- **Splits per config:** `train` / `valid` / `test`

## How to use

```python
from datasets import load_dataset

# pick a sample-size config: "100", "500", "1000", or "10000"
ds = load_dataset("{ENZH_REPO}", "1000")
print(ds)
sample = ds["test"][0]
print(sample["lang"], sample["text"])
```

## Citation

If you use this dataset, please cite the CLD paper (ICML 2026) and the underlying
corpora: Common Voice (Ardila et al., 2020), the National Speech Corpus (IMDA),
Lahaja (Sanket et al., 2024), and MUSAN (Snyder et al., 2015).
"""


def _card(repo: str, languages: list[str], body: str) -> DatasetCard:
    data = DatasetCardData(
        license=LICENSE,
        language=languages,
        pretty_name=repo.split("/")[-1],
        task_categories=["audio-classification", "automatic-speech-recognition"],
        tags=[
            "language-identification",
            "spoken-language-identification",
            "speech",
            "low-resource",
            "accent-robust",
        ],
    )
    return DatasetCard(f"---\n{data.to_yaml()}\n---\n\n{body}")


def _write_local_card(repo: str, card: DatasetCard) -> None:
    out_dir = os.path.join(HF_UPLOAD_DIR, repo.split("/")[-1])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "README.md")
    with open(path, "w") as f:
        f.write(str(card))
    print(f"[card] wrote {path}")


def _push_card(repo: str, languages: list[str], body: str, dry_run: bool) -> None:
    """Build the dataset card, save a local copy, and (unless dry-run) push it.

    Loads the live card first so the auto-generated config/split metadata that
    ``push_to_hub`` wrote is preserved; only the body text + a few metadata keys
    are (re)set.
    """
    card = _card(repo, languages, body)
    _write_local_card(repo, card)
    if dry_run:
        print(f"[dry-run] would push card -> {repo}")
        return
    try:
        live = DatasetCard.load(repo, repo_type="dataset")
        live.text = card.text
        # Merge our metadata onto the auto-generated (configs/dataset_info) block.
        for key, value in card.data.to_dict().items():
            setattr(live.data, key, value)
        card = live
    except Exception as e:
        print(f"[card] no existing card to merge for {repo} ({e}); pushing fresh card")
    card.push_to_hub(repo, repo_type="dataset")
    print(f"[done] card -> https://huggingface.co/datasets/{repo}")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def push_multi(dry_run: bool, cards_only: bool) -> None:
    print(f"[multi] {MULTI_SRC} -> {MULTI_REPO}")
    if not cards_only:
        ds = load_from_disk(MULTI_SRC)
        print(f"  {ds}")
        if not dry_run:
            ds.push_to_hub(MULTI_REPO, private=False)
            print(f"[done] https://huggingface.co/datasets/{MULTI_REPO}")
    _push_card(MULTI_REPO, ["en", "zh", "id", "ms", "hi"], _multi_card_body(), dry_run)


def push_enzh(dry_run: bool, cards_only: bool) -> None:
    if not cards_only:
        for n in ENZH_SIZES:
            src = _enzh_src(n)
            config = str(n)
            print(f"[enzh] {src} -> {ENZH_REPO} (config={config})")
            ds = load_from_disk(src)
            print(f"  {ds}")
            if not dry_run:
                ds.push_to_hub(ENZH_REPO, config_name=config, private=False)
                print(f"[done] https://huggingface.co/datasets/{ENZH_REPO} (config={config})")
    _push_card(ENZH_REPO, ["en", "zh"], _enzh_card_body(), dry_run)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Print plan + write local cards, no upload.")
    ap.add_argument("--only", choices=["multi", "enzh"], help="Push just one dataset.")
    ap.add_argument("--cards-only", action="store_true", help="Skip data; (re)push dataset cards only.")
    args = ap.parse_args()

    if args.only in (None, "multi"):
        push_multi(args.dry_run, args.cards_only)
    if args.only in (None, "enzh"):
        push_enzh(args.dry_run, args.cards_only)


if __name__ == "__main__":
    main()
