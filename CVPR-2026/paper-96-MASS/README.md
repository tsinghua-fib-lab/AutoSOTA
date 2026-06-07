<div align="center">

# MASS: Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision

**CVPR 2026**

Yunhe Gao, Yabin Zhang, Chong Wang, Jiaming Liu, Maya Varma,
Jean-Benoit Delbrouck, Akshay Chaudhari, Curtis Langlotz

**Stanford University**

[![arXiv](https://img.shields.io/badge/arXiv-2603.13660-b31b1b.svg)](https://arxiv.org/abs/2603.13660)
[![GitHub](https://img.shields.io/badge/GitHub-Stanford--AIMI%2FMASS-black)](https://github.com/Stanford-AIMI/MASS)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://yhygao.github.io/MASS_page/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/StanfordAIMI/MASS)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

<div align="center">
  <img src="figs/framework.jpg" width="90%">
</div>


MASS is a mask-guided self-supervised learning framework for 3D medical images.
It learns strong representations from unlabeled CT, MRI, and PET volumes without
using ground-truth annotations during pretraining.

The key idea is to turn automatically generated class-agnostic masks into a
large collection of in-context segmentation tasks. Given a reference image-mask
pair, the model learns to segment the corresponding region in a query image.
Across many masks, modalities, anatomies, and spatial scales, MASS learns
generalizable medical image representations from diverse dense prediction tasks.

This release implements MASS with the Iris in-context segmentation architecture.
The MASS pretraining objective is architecture-agnostic in principle and can be
adapted to other in-context segmentation models.

## News

- **2026/02** MASS was accepted to **CVPR 2026**.
- **2026/03** The paper is available on [arXiv](https://arxiv.org/abs/2603.13660).
- **2026/05** This repository includes preprocessing, pretraining, in-context
  evaluation, raw NIfTI inference, and downstream examples.
- **2026/05** We release [`mass_base.pth`](https://huggingface.co/StanfordAIMI/MASS),
  a MASS-pretrained checkpoint trained only with auto-generated masks and no
  expert-labeled annotations.

## Highlights

| Feature | Why it matters |
|:--|:--|
| Annotation-free self-supervision | MASS learns from unlabeled CT, MRI, and PET volumes using auto-generated masks, without ground-truth annotations during pretraining. |
| Diverse in-context tasks | Every auto mask becomes a dense segmentation task, so generalization comes from broad task diversity across anatomy, pathology, scale, and modality. |
| Segmentation-native representation learning | Unlike reconstruction or contrastive SSL, MASS directly trains the model to perform spatially precise, anatomy-aware dense prediction. |
| Fast and broadly transferable | MASS converges efficiently and transfers to training-free in-context segmentation, low-label supervised finetuning, and frozen-encoder classification. |
| End-to-end open pipeline | This release includes SAM2-based preprocessing, mask-guided pretraining, in-context evaluation, raw NIfTI inference, and downstream examples. |

## Why Segmentation-Native SSL?

Most medical SSL methods pretrain with reconstruction or contrastive objectives.
These objectives are useful, but they are indirect proxies for the tasks we care
about in medical imaging.
Medical image understanding usually requires both semantic recognition and
spatial localization: a model must know what an anatomical structure or
pathology is, and also where it is, how its boundary is shaped, and how it sits
relative to surrounding anatomy. Reconstruction and contrastive objectives only
weakly enforce this joint "what and where" understanding.

Reconstruction-based SSL teaches a model to recover missing intensities. This
often emphasizes low-level texture and appearance, and usually requires large
amounts of data and long training schedules before useful semantic features
emerge. MASS is more supervision-efficient: each auto mask defines a dense
in-context segmentation task, so every volume provides hundreds to thousands of
spatially grounded pretraining tasks. In our experiments, this leads to faster
convergence and substantially stronger downstream performance than
reconstruction-based SSL.

Contrastive SSL, including image contrastive learning and CLIP-style image-text
contrastive learning, learns powerful global representations by aligning
augmented views or image-report pairs. However, these objectives are usually
weakly localized: they encourage image-level or patch-level invariance and global
semantic alignment, but do not directly require the model to identify where an
anatomical structure is, what its boundary looks like, or how local shape relates
to surrounding anatomy. This mismatch is especially important in 3D medical
imaging, where downstream tasks often require dense spatial reasoning over
organs, vessels, lesions, and fine anatomical subregions.

MASS directly pretrains with dense in-context segmentation tasks. Each auto mask
defines a localized task, forcing the model to learn anatomy, morphology,
boundaries, and spatial context through prediction in 3D image space. As a
result, MASS is well aligned with dense medical image understanding while still
producing an encoder that transfers to global classification.

In short, MASS is designed to be both effective and efficient: it learns from
unlabeled images, converges quickly, performs strongly on dense segmentation
tasks, and provides a generalizable 3D medical image encoder for downstream
classification and finetuning.

## Installation

The training, evaluation, and raw NIfTI inference code use the root environment:

```bash
conda create -n mass python=3.10
conda activate mass
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if the default wheel is
not suitable for your machine.

SAM2 preprocessing uses a separate environment because SAM2 and
TotalSegmentator dependencies are heavier and more version-sensitive:

```bash
conda create -n mass-preprocess python=3.10
conda activate mass-preprocess
pip install -r preprocessing/requirements.txt
```

See [`preprocessing/README.md`](preprocessing/README.md) for preprocessing
setup, SAM2 checkpoint setup, data layout, resume behavior, and dataset scripts.

## Pretrained Checkpoint

We release `mass_base.pth` on Hugging Face at
[`StanfordAIMI/MASS`](https://huggingface.co/StanfordAIMI/MASS). This is a MASS
pretraining checkpoint trained with the Iris in-context architecture using only
automatically generated class-agnostic masks. It has not seen expert-labeled
ground-truth annotations during pretraining.

`mass_base.pth` can be used directly for training-free in-context segmentation:
given reference image-mask examples, it segments corresponding structures in
query images without task-specific finetuning. For stronger task-specific
performance, use it as initialization and finetune with GT labels.

Download with:

```bash
hf download StanfordAIMI/MASS mass_base.pth --local-dir checkpoints
```

Use EMA weights when available:

```bash
--use-ema
```

or in downstream configs:

```yaml
use_ema_checkpoint: true
```

## Raw NIfTI In-Context Inference

`inference.py` runs in-context segmentation directly from raw `.nii` /
`.nii.gz` files. Inputs are testing image(s), reference image(s), and matching
reference segmentation mask(s).

```bash
python inference.py \
  --checkpoint /path/to/mass_checkpoint.pth \
  --test-image /path/to/test_image.nii.gz \
  --reference-image /path/to/reference_image.nii.gz \
  --reference-mask /path/to/reference_mask.nii.gz \
  --output outputs/test_image_seg.nii.gz \
  --gpu 0 \
  --use-ema \
  --modality ct \
  --orientation RAS \
  --target-spacing 1.5 1.5 1.5 \
  --window-size 128 128 128 \
  --overlap 0.5
```

The script reorients and resamples images, crops the testing image, encodes
reference masks into task embeddings, runs sliding-window inference, and writes
the segmentation back in the original image space.

Please make sure the input NIfTI metadata is complete and reliable, especially
orientation and spacing. Missing or incorrect metadata can break reorientation.
`mass_base.pth` was trained after standardizing images to RAS orientation, so
using `--orientation RAS` is recommended for stable inference.

Multi-class inference is supported. If the reference mask contains multiple
nonzero labels, the script encodes each label separately, concatenates task
embeddings, and predicts the multi-class output in one forward pass when memory
allows. Multiple reference image/mask pairs are averaged into a task embedding
ensemble for each label.

Useful options:

- `--reference-label`: segment only selected labels from the reference masks.
- `--output-label`: remap the output label when segmenting one class.
- `--max-classes-per-forward`: split labels into chunks if GPU memory is limited.
- `--save-probability`: save one probability map per reference label.
- `--no-largest-component`: disable largest-connected-component postprocessing.
- `--output-dir`: process multiple testing images and write one output per image.

## Segmentation Finetuning

The finetuning example initializes MASS/Iris from a pretrained checkpoint and
trains it as a task-specific segmentation model.

```bash
python train.py \
  --config config/downstream/segmentation_finetune_example.yaml \
  --gpu 0 \
  --name segmentation_finetune_example \
  --override \
    finetuning.pretrained_checkpoint=/path/to/mass_checkpoint.pth \
    data.train.data_root=/path/to/mass_h5 \
    data.val.data_root=/path/to/mass_h5 \
    data.train.datasets='[example_segmentation]' \
    data.val.datasets='[example_segmentation]'
```

The example uses `FineTuningDataset`. The number of foreground classes is
inferred from the dataset metadata in `data/split.py`; for a custom class subset,
set `data.train.foreground_classes` and `data.val.foreground_classes`.

You can also launch the example script:

```bash
CHECKPOINT=/path/to/mass_checkpoint.pth GPUS=0 \
  bash scripts/segmentation_finetune_example.sh
```

## Classification Linear Probing

The classification example loads the MASS/Iris encoder, freezes it by default,
and trains a lightweight classification head.

```bash
python train.py \
  --config config/downstream/classification_linear_probe_example.yaml \
  --gpu 0 \
  --name classification_linear_probe_example \
  --override \
    classification.encoder.pretrained_checkpoint=/path/to/mass_checkpoint.pth \
    classification.num_classes=2 \
    data.train.data_root=/path/to/classification_data \
    data.val.data_root=/path/to/classification_data \
    data.train.datasets='[example_classification]' \
    data.val.datasets='[example_classification]'
```

`labels.csv` should contain a `filename` column, label columns, and optionally a
`split` column. For single-label classification, labels should be zero-based
class indices, such as `0/1` for binary tasks or `0/1/2` for three-class tasks.
Make sure the resolved `num_classes` matches the dataset labels. See
`data/dataset_classification.py` for accepted CSV formats.

You can also launch the example script:

```bash
CHECKPOINT=/path/to/mass_checkpoint.pth GPUS=0 \
  bash scripts/classification_linear_probe_example.sh
```

## Preprocessing and Mask-Guided Pretraining

The released `mass_base.pth` checkpoint was trained with the data used in our
paper and the Iris in-context segmentation architecture. It uses only
auto-generated masks during pretraining and does not use expert ground-truth
annotations.

If you want to train MASS on your own imaging data, or adapt the MASS
mask-guided objective to another in-context model architecture, this repository
also provides the full preprocessing and pretraining framework. The default
preprocessing pipeline uses SAM2 to generate class-agnostic mask proposals, then
resamples images and masks into the format used by the pretraining dataset.

For image-only unlabeled pretraining data, start from:

```bash
python preprocessing/scripts/unlabeled.py
```

For labeled public segmentation datasets, use the dataset-specific scripts as
templates:

```bash
python preprocessing/scripts/bcv.py
python preprocessing/scripts/amos.py
python preprocessing/scripts/autopet.py
python preprocessing/scripts/totalsegmentator.py
```

The detailed training data format, SAM2 mask generation, enhancement profiles,
physical slice interval, resume behavior, and known dataset issues are documented
in [`preprocessing/README.md`](preprocessing/README.md).

MASS can train with any reasonable mask proposals that cover meaningful image
objects. SAM2 is our default mask generator, but other auto masks can be used,
and auto masks can also be mixed with expert masks during pretraining.

After preprocessing, run mask-guided self-supervised pretraining with:

```bash
python train.py \
  --config config/pretrain/mask_guided_self_supervised.yaml \
  --gpu 0 \
  --name mass_pretrain
```

or with the launcher:

```bash
GPUS=0 bash scripts/train_pretrain.sh
```

For multi-GPU training on one node, override the processed data root and dataset
list:

```bash
python train.py \
  --config config/pretrain/mask_guided_self_supervised.yaml \
  --gpu 0,1,2,3 \
  --name mass_pretrain \
  --override data.train.data_root=/path/to/mass_h5 data.train.datasets='[bcv,amos_ct,amos_mr]'
```

The pretraining config controls:

- `data.train.data_root`: root folder containing processed dataset folders.
- `data.train.datasets`: list of dataset folders to sample from.
- `target_spacing`: processed voxel spacing in `[z, y, x]` order.
- `training_size`: 3D crop size used by the model.
- `augmentation`: weak and strong augmentation settings for reference/query
  views.
- `model`, `optimizer`, `scheduler`, `amp`, `ema`: model and optimization
  settings.

Outputs are written under `run.output_dir` from the config, with the run name
from `--name`:

```text
runs/pretrain/mass_pretrain/
  train.log
  logs/
  metrics/
  checkpoints/
    latest.pth
    best.pth
    epoch_*.pth
```

Resume training with:

```bash
python train.py \
  --config config/pretrain/mask_guided_self_supervised.yaml \
  --gpu 0,1,2,3 \
  --resume runs/pretrain/mass_pretrain/checkpoints/latest.pth
```

Evaluate a checkpoint on processed datasets with GT masks:

```bash
python evaluate.py \
  --checkpoint runs/pretrain/mass_pretrain/checkpoints/best.pth \
  --dataset bcv \
  --data-root /path/to/mass_h5 \
  --reference-mode fixed \
  --ensemble-size 1 \
  --gpus 0 \
  --use-ema
```

`evaluate.py` uses the processed MASS data format rather than raw NIfTI files.
It reports Dice, ASD, and HD95. Reference selection is controlled by
`--reference-mode` (`random` or `fixed`) and `--ensemble-size`; `fixed`
references are read from `data/split.py`.

Useful evaluation options:

- `--save-predictions`: save predicted segmentation volumes next to the
  checkpoint under `predictions_<dataset>_<timestamp>/`.
- `--skip-surface-metrics`: skip ASD/HD95 for faster evaluation.
- `--disable-amp`: disable mixed precision.

## Configuration

All training entry points use YAML configs plus CLI overrides:

```bash
python train.py --config path/to/config.yaml --override key1.key2=value key3=value
```

Nested keys are addressed with dots. CLI overrides are useful for changing
paths, datasets, checkpoint locations, or small hyperparameters without copying
the full YAML file.

Registered names used by the example configs:

- model: `iris`
- pretraining dataset: `MaskGuidedSelfSupervisedDataset`
- evaluation dataset: `MetaUniversalDataset`
- finetuning dataset: `FineTuningDataset`
- classification dataset: `classification`
- trainers: `mask_guided_self_supervised`, `finetuning`, `classification`

## Notes

- The preprocessing pipeline is intentionally standalone. Use
  `preprocessing/requirements.txt` rather than the root `requirements.txt` for
  SAM2 mask generation.
- Raw NIfTI inference is designed to match training preprocessing as closely as
  possible, but dataset-specific spacing/orientation issues should still be
  inspected visually.
- Public medical datasets often contain metadata issues. The preprocessing
  scripts include dataset-specific fixes for several known cases.

## Citation

If you find MASS useful in your research, please cite:

```bibtex
@article{gao2026learning,
  title={Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision},
  author={Gao, Yunhe and Zhang, Yabin and Wang, Chong and Liu, Jiaming and Varma, Maya and Delbrouck, Jean-Benoit and Chaudhari, Akshay and Langlotz, Curtis},
  journal={arXiv preprint arXiv:2603.13660},
  year={2026}
}

@inproceedings{gao2025show,
  title={Show and segment: Universal medical image segmentation via in-context learning},
  author={Gao, Yunhe and Liu, Di and Li, Zhuowei and Li, Yunsheng and Chen, Dongdong and Zhou, Mu and Metaxas, Dimitris N},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={20830--20840},
  year={2025}
}
```

## License

This project is released under the [MIT License](LICENSE).
