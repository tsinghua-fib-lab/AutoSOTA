# g2net-gravitational-wave-detection

This folder contains [datasaurus's](https://www.kaggle.com/anjum48) part of the G2Net Gravitational Wave Detection solution.
# Setup
Edit `src/config.py` to reflect the `INPUT_PATH` and `OUTPUT_PATH` locations on your machine:
* `INPUT_PATH` is where the competition train & test data resides as well as any pseudolabelling files
* `OUTPUT_PATH` is where the outputted model weights will be saved. If you have the model weights, put them in this folder.

# Training
To train a single model using a config listed in `hyperparams.yml` run:
```
python train.py --config <config_name>
```
To run a 5-fold cross validation use the shell script `train.sh`. This script will also run `infer.py` and 
generate out-of-fold (OOF) predictions for stacking models.
```
sh train.sh <config_name>
```
The output directory structure is as follows:
```
├── YYYYMMDD-HHMMSS  # Model timestamp
│   ├── model_name  # e.g. resnet200d
│   │   ├── fold_0
│   │   │   ├── model_checkpoint_for_best_loss.ckpt
│   │   │   ├── model_checkpoint_for_best_auc.ckpt
│   │   │   ├── hparams.yaml
│   │   │   ├── tensorboard_events_file
│   │   ├── fold_1
│   │   │   ├── etc.
│   ├── oofs_YYYYMMDD-HHMMSS_CV-XXXX.csv  # OOF predictions from infer.py
│   ├── sub_YYYYMMDD-HHMMSS_CV-XXXX.csv  # submission file from infer.py

```
Note that the configurations are set for training on a machine with 2x RTX 3090 GPUs. You may need to edit `hyperparams.yml` to reflect your hardware setup for the number of GPUs and batch size depending on available memory.

# Models used in competition
The following models were trained either using the CQT transform from `nnAudio` or CWT.
* ResNet-200d (CQT)
* EfficientNet B5 (CQT)
* EfficientNet V2 M (CQT)
* A basic 1D CNN
* EfficientNet B5 (CWT)
* EfficientNet B5 (CQT) with pseudo labelling
* EfficientNet B3 (CQT) with pseudo labelling
* EfficientNet B3 (CWT) with pseudo labelling

# Reproducing Kaggle models
To reproduce the model weights used in the final submission run:
```
sh reproduce_train.sh 
```
Note that a CSV file with the pseudolabels will need to be in the `INPUT_PATH` for the pesudolabelled models (e.g. `submission_power2_weight.csv`).

To reproduce the submission files (and OOF predictions), copy all the model weights to the `OUTPUT_PATH` (maintaining the structure described above) and run:
```
sh reproduce_infer.sh 
```
