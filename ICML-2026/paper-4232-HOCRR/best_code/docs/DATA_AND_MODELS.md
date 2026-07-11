# Data and Model Setup

This repository does not include raw datasets or pretrained checkpoints. Create local directories such as `data/` and `models/` and pass paths explicitly to scripts.

## MNIST Rotation

The rotated MNIST experiment uses the standard MNIST dataset downloaded through `torchvision`, then creates rotated images with ground-truth angles. The helper code lives in `experiments/mnist_rotation/`.

To train or load the E2CNN rotation regressor, see `docs/MNIST_ROTATION.md`.

## UTKFace Age Estimation

Download UTKFace from the dataset page:

- https://github.com/aicip/UTKFace

The scripts parse labels from filenames of the form:

```text
[age]_[gender]_[race]_[date&time].jpg
```

## MiVOLO-v2

The age-estimation experiment uses MiVOLO-v2 for inference only. Download the Hugging Face checkpoint locally, for example into `models/mivolo_v2_hf/`:

- https://huggingface.co/iitolstykh/mivolo_v2

The scripts accept `--model_dir models/mivolo_v2_hf`.
