# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2026-05-19

### Added

- Type annotations across all `tedbench/` Python files (`data`, `layer`,
  `model`, `utils`, `lr_schedulers`). Uses plain `torch.Tensor` style
  throughout; `py.typed` marker enables PEP 561 downstream type-checking.
- README badges: CI status, PyPI version, Python versions, license.
- Project logo.

---

## [0.1.0] — 2026-05-13

### Added

- Initial public release accompanying the ICML 2026 paper
  *Protein Fold Classification at Scale: Benchmarking and Pretraining*.
- **TEDBench dataset**: 462,175 AlphaFold structures across 965 CATH topology
  classes, with train / val / test splits and an external CATH 4.4 experimental
  test set (27,638 structures). Available on HuggingFace Hub
  (`TEDBench/ted`) and via auto-download from MPCDF.
- **MiAE** (Masked Invariant Autoencoders): SE(3)-invariant masked autoencoder
  for protein backbone frames in three sizes (S / B / L).  Pretrained
  checkpoints, fine-tuned fold classifiers, and from-scratch baselines are all
  published on HuggingFace Hub.
- Top-level convenience API: `tedbench.load_model(name)` and
  `tedbench.list_models()`.
- `LightningStructureDataset` data module supporting both HuggingFace Hub and
  local auto-downloading backends.
- Baselines: ESM2, SaProt, ProteinMPNN (scripts in `baselines/`; requires
  `pip install TEDBench[baselines]` for ESM2/SaProt).
