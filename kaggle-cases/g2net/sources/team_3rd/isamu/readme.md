# Part of 2D models for the Kaggle competition [G2Net Gravitational Wave Detection](https://www.kaggle.com/c/g2net-gravitational-wave-detection)

## Model Overview
In this program, whitening is performed on three signals and the CQT image is generated. The image is combined with the output of each signal as a channel to create a single image of size [512, 512, 3]. A CNN model with these images as input is used to learn the classification problem.

## Models
We used  Keras and TPU(Tensor processing unit) to run the following large models for high speed training
 - EfficientNet-B3, B4, B5, B7
 - Inception-V3 

Part of models run in Kaggle TPU during the competition, We made them public when the competition was over, so you can see the code and result in Kaggle. EfficientNet-B5([fold0](https://www.kaggle.com/yamsam/g2net-soft-pl-b5-fold0),  [fold1](https://www.kaggle.com/yamsam/g2net-soft-pl-b5-fold1), [fold2](https://www.kaggle.com/yamsam/g2net-soft-pl-b5-fold2), [fold3](https://www.kaggle.com/yamsam/g2net-soft-pl-b5-fold3), [fold4](https://www.kaggle.com/yamsam/g2net-soft-pl-b5-fold4)) and EfficientNet-B7([fold0](https://www.kaggle.com/yamsam/tpu-whiten-data-baseline-b7-fold0), [fold1](https://www.kaggle.com/yamsam/tpu-whiten-data-baseline-b7-fold1), [fold2](https://www.kaggle.com/yamsam/tpu-whiten-data-baseline-b7-fold2), [fold3](https://www.kaggle.com/yamsam/tpu-whiten-data-baseline-b7-fold3), [fold4](https://www.kaggle.com/yamsam/tpu-whiten-data-baseline-b7-fold4)). The rest of the code is stored in notebooks.

Execution time will vary depending on the model, but will take 1-2 days on TPU.

## Model Settings
  - Image size 512x512
  - Stratified KFold
  - On the fly preprocessing(whitening signal and CQT image generating)
  - CQT params
    - hop_length=32
    - fmin=22
    - fmax=None
    - n_bins=64
    - norm=1
    - window='nuttall'
    - bins_per_octave=21
    - filter_scale=1
  - Optimizer: AdamW
  - Loss function: Binary Cross Entropy
  - Augmentation: swapping channel between two LIGOs when training as augmentation and inference as TTA(Test Time Augmentation)
  - Soft Pseudo Labeling made from prediction of other models
  

## TFRecords Dataset
We created the following TFRecord data set which is useful for handling large data sets on TPU.Those TFRecords datasets are publicly available on Kaggle Dataset.

- [Strafieid KFold TFRecord Dataset for training](https://www.kaggle.com/vincentwang25/g2net-skf) that included original G2Net training data(signal and target values) with strafieid kfold setup.
- TFRecord Dataset for test [(0)](https://www.kaggle.com/hidehisaarai1213/g2net-waveform-tfrecords-test-0-4)[(1)](https://www.kaggle.com/hidehisaarai1213/g2net-waveform-tfrecords-test-5-9) that included original G2Net test data(signal)
- Soft Pseudo Labeling TFRecord Dataset[(0)](https://www.kaggle.com/yamsam/g2net-public-s-01) [(1)](https://www.kaggle.com/yamsam/g2net-public-s-02) using a prediction from [public kaggle kernel](https://www.kaggle.com/hijest/g2net-efficientnetb3-b7-ensemble) (Puiblic LB 0.8779)
- [Soft Pseudo Labeling TFRecord Dataset](https://www.kaggle.com/yamsam/g2net-sp-av) using a prediction from emsemble of other model outputs (Puiblic LB 0.8822)

## Requirements

This notebooks has been tested in the following TPU environments.
  - kaggle TPU kernel
  - colab pro TPU

## Training and Inference on GPU
These models were trained and inferred in the TPU environment, but we have attached code that works in the GPU environment, just in case. The inference code can also be run using the weights of the models trained on TPU.

- Download the tfrecord data for training and save it in the train_tfrecord directory, download the tfrecod data for testing and save it in the test_tfrecord directory, and if you use pseudo label data, download it and save it in pseudo_ tfrecord directory. Store the directory containing all of these in the INPUT_PATH directory of config.py.

- In the DATA_PATH of config.py, specify the directory where the data used for [spectral whitening](https://www.kaggle.com/yamsam/g2net-w-prof) was downloaded. Set the CONFIG_PATH to the yaml file.

- Training: Specify the output path of the trained model in OUTPUT_PATH and run train.sh.

- Inference: Specify the directory where the weights of the models to be used for inference are stored in MODEL_PATH. if you want to use the weights learned in the TPU environment, store them in the following directories for each model: Eff-B3, Eff-B4, Eff-B5, Eff-B7. These names are set in the yaml file. Run infer.sh at the end.
