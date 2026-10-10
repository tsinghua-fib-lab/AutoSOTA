# MeisenMeister

MeisenMeister is a framework for breast cancer classification on DCE-MRI. It is designed to help develop reproducible multi-stage pipelines, from dataset fingerprinting and experiment planning to ROI preprocessing, fold-safe training, benchmarking, and ROI-level inference.

> 🏆 Winning solution of the **MICCAI 2025 ODELIA Breast MRI Challenge** on Grand Challenge: https://odelia2025.grand-challenge.org/

## Installation

```bash
conda create -n meisenmeister python=3.12 -y
conda activate meisenmeister
pip install -e .
```

## Pretrained Weights

Pretrained MeisenMeister weights for bilateral breast MRI classification into `healthy`, `benign`, and `malignant` are available on Hugging Face:

https://huggingface.co/Bubenpo/MeisenMeister

Download them with:

```bash
hf download Bubenpo/MeisenMeister --local-dir ./MeisenMeister
```

Then run inference from the downloaded weights with `mm_predict_from_modelfolder`.

## Environment Configuration

MeisenMeister resolves datasets and outputs via three required storage roots:

- `MM_RAW`
- `MM_PREPROCESSED`
- `MM_RESULTS`

The code also accepts lowercase variants (`mm_raw`, `mm_preprocessed`, `mm_results`).

```bash
export MM_RAW=/path/to/mm_raw
export MM_PREPROCESSED=/path/to/mm_preprocessed
export MM_RESULTS=/path/to/mm_results
```

## Workflow At A Glance

1. Raw data is discovered under `MM_RAW` (`Dataset_<id>_*` naming).
2. Fingerprinting and planning define preprocessing geometry and output format.
3. Preprocessing writes ROI tensors and metadata to `MM_PREPROCESSED`.
4. Split generation writes leakage-safe five-fold splits.
5. Training and benchmarking write experiment artifacts to `MM_RESULTS`.
6. Inference runs either from local dataset context or from downloaded model weights via `mm_predict_from_modelfolder`.

## Quickstart

```bash
# 1) Optional: create breast masks if masksTr is missing
mm_create_breast_segmentations -d 1

# 2) Fingerprint + plan + preprocess
mm_extract_dataset_fingerprint -d 1 --num-workers 8
mm_plan_and_preprocess -d 1 --num-workers 8

# 3) Build 5-fold split file
mm_create_5fold -d 1

# 4) Train fold 0
mm_train -d 1 -f 0

# 5) Predict with fold ensemble
mm_predict -d 1 -i /path/to/images -o /path/to/preds -f 0 1 2 3 4
```

## Installed CLI Commands

| Command | Purpose |
|---|---|
| `mm_extract_dataset_fingerprint` | Compute dataset fingerprint statistics used by planning. |
| `mm_create_breast_segmentations` | Create breast segmentations in `masksTr`. |
| `mm_homogenize` | Resample channels into `_0000` space and overwrite raw NIfTI files. |
| `mm_plan_experiment` | Build `mmPlans.json` from fingerprint data. |
| `mm_preprocess` | Generate preprocessed ROI data from `mmPlans.json`. |
| `mm_plan_and_preprocess` | Run planning and preprocessing in one step. |
| `mm_create_5fold` | Create case-aware leakage-safe `splits.json`. |
| `mm_train` | Run fold-based training with registered trainer classes. |
| `mm_benchmark_train` | Benchmark train/validation throughput with warmup controls. |
| `mm_predict` | Run ROI-level inference using local dataset + results roots. |
| `mm_predict_from_modelfolder` | Run inference from a portable experiment folder. |

## Storage Layout

- `MM_RAW`: source datasets (`Dataset_001_*`, `imagesTr`, `masksTr`, `dataset.json`)
- `MM_PREPROCESSED`: `dataset_fingerprint.json`, `mmPlans.json`, `splits.json`, ROI outputs
- `MM_RESULTS`: experiment folders, checkpoints, logs, plots, evaluation outputs

## Documentation

- [High-Level Overview](docs/overview.md)
- [Setup and Paths](docs/setup.md)
- [CLI Guide](docs/cli.md)
- [Training and Experiments](docs/training.md)
- [Data and Pipeline Notes](docs/pipeline.md)

## License

The repository source code is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).

Model weights are licensed under **CC BY-NC-SA 4.0** due to downstream dataset licensing constraints from the data used for training.

## Citation

If you use MeisenMeister in research, please cite:

Hamm, B., Kirchhoff, Y., Rokuss, M., and Maier-Hein, K., *MeisenMeister: A Simple Two Stage Pipeline for Breast Cancer Classification on MRI*, arXiv:2510.27326 [cs.CV], 2025.

Paper: https://arxiv.org/pdf/2510.27326

```bibtex
@article{hamm2025meisenmeister,
  title={MeisenMeister: A Simple Two Stage Pipeline for Breast Cancer Classification on MRI},
  author={Hamm, Benjamin and Kirchhoff, Yannick and Rokuss, Maximilian and Maier-Hein, Klaus},
  journal={arXiv preprint arXiv:2510.27326},
  year={2025}
}
```

This model also relies on the BreastDivider dataset and segmentation work:

Dataset: https://huggingface.co/datasets/Bubenpo/BreastDividerDataset

```bibtex
@article{rokuss2025breastdivider,
  title     = {Divide and Conquer: A Large-Scale Dataset and Model for Left-Right Breast MRI Segmentation},
  author    = {Rokuss, Maximilian and Hamm, Benjamin and Kirchhoff, Yannick and Maier-Hein, Klaus},
  journal   = {arXiv preprint arXiv:2507.13830},
  year      = {2025}
}
```
