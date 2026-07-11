# MalTree Data Card

This document describes the MalTree data release: its contents, schema, and how
to obtain and verify it.

## Dataset

MalTree analyzes **103,883 malware samples** spanning **538 families**, with
VirusTotal first-submission timestamps from 2010 to 2023. Each sample is
identified by the **SHA256 of its binary**. `data/manifest.csv` is the sample
index, with one row per sample listing its family, timestamp, and per-modality
coverage.

## What is released

| Location | Contents |
|----------|----------|
| **This repository** | All code, the sample manifest, family labels, timestamps, and a small example subset. |
| **Zenodo** | The four multi-modal embedding files (~14 GB; too large for git). |
| **Not distributed** | Raw malware binaries and memory dumps (under NDA). |

Raw samples are not shared, but because every sample is keyed by its SHA256, a
researcher with VirusTotal (or similar) access may be able to retrieve the
original binaries from those hashes. As stated in the paper, MalTree releases
*embeddings*, not samples.

## Released files (Zenodo)

The release is archived at **[doi.org/10.5281/zenodo.20261117](https://doi.org/10.5281/zenodo.20261117)**.
Each file is a JSON object keyed by the sample's **SHA256**. Download them with
`python scripts/download_data.py` and verify against `CHECKSUMS.sha256`.

| File | Samples | Schema |
|------|---------|--------|
| `embeddings_fused.json` | 102,322 | `{sha256: {embedding: float[1000], family: str}}` |
| `embeddings_static.json` | 102,322 | `{sha256: {embedding: float[3512], family: str}}` |
| `embeddings_dynamic.json` | 102,322 | `{sha256: {embedding: float[1000], family: str}}` |
| `embeddings_image.json` | 102,322 | `{sha256: {embedding: float[2048], family: str}}` |

`embeddings_fused.json` is the 1000-d representation used for tree construction
and is the entry point for the reproduction pipeline (see `README.md`).

## In-repo data

| File | Description |
|------|-------------|
| `data/manifest.csv` | Sample index (103,883 rows): `id`, `id_type`, `sha256`, `family`, `first_submission`, per-modality flags. |
| `data/family_labels.json` | `{sha256: family}`. |
| `data/timestamps.json` | `{sha256: {first_submission: date}}` (VirusTotal first-submission dates). |
| `data/example/` | A 96-sample, 8-family subset (fused embeddings, labels, timestamps) for running the pipeline without the full download. |

## Download and verify

```bash
# Full release, fetched from Zenodo (doi.org/10.5281/zenodo.20261117)
python scripts/download_data.py
python scripts/download_data.py --verify-only

# Or use the bundled example subset, no download required
python scripts/download_data.py --example
```

`scripts/dataset/` contains the scripts that build the released files from the
source data: `build_manifest.py` (the sample manifest), `prepare_release.py`
(the SHA256-keyed embedding files), and `make_release_artifacts.py` (family
labels, timestamps, and the example subset).

## Scope

Feature extraction operates on raw artifacts that are **not distributed**:

- **Pseudo-static features** are extracted from **process memory dumps** of the
  samples. Reproducing this stage requires the dumps.
- **Dynamic features** come from sandbox (ANY.RUN / VirusTotal) behavioral
  traces.
- **Image features** are extracted from the sample binaries.

The memory dumps and raw binaries are under NDA and are not shared. The
extraction code is included for completeness; reproduction starts from the
released embeddings. `phylogenetic_tree/embedding_fusion.py` provides the
concatenation and L2 normalization used to form the fused representation.