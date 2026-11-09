#!/usr/bin/env python
# coding: utf-8

# In[1]:


from src.config import OUTPUT_PATH, INPUT_PATH, DATA_PATH
from src.utils import prepare_args


# # Config

# In[2]:


config = prepare_args()
print ('config=', config)


# In[3]:


MODEL_NAME= config.model_name

# CQT params
FMIN=config.fmin
FMAX=config.fmax
WINDOW_TYPE=config.window_type
BINS=config.bins
HOP_LENGTH = config.hop_length
SCALE=config.scale
NORM=config.norm
OCTAVE=config.octave

SMOOTHING=0.00
ST=int(4096 / 16 * 7)
EN=int(4096 / 16 * 15)

NUM_FOLDS = config.nfold
FOLDS=config.folds

LR=config.lr
IMAGE_SIZE = config.image_size
BATCH_SIZE = config.batch_size_infer
EFFICIENTNET_SIZE = config.efficientnet_size
WEIGHTS =  config.weight

MIXED=True # mixed precision does not work with tf models
TFHUB_MODEL=config.tfhub_model

MIXUP_PROB = 0.0
EPOCHS = config.epochs
R_ANGLE = 0
S_SHIFT = 0.0
T_SHIFT = 0.0
LABEL_POSITIVE_SHIFT = 1.0
TTA = config.tta

SEED = config.seed
WHITE=True

# G2Net SKF(Stratified KFold) dataset
# gs-path were generated from https://www.kaggle.com/yamsam/g2net-skf-path
# tf reccords Dataset are stored in https://www.kaggle.com/vincentwang25/g2net-skf

#FILES =['gs://kds-545de03072f7f12c036f4c687111566f0a27586e55b81c3a5ee34eff']
# https://www.kaggle.com/yamsam/g2net-tf-on-the-fly-cqt-tpu-inference-path
FILES =[
    f'{INPUT_PATH}/train_tfrecord'
]

# Pseudo Label dataset
# gs-path were generated from  https://www.kaggle.com/yamsam/g2net-tpu-soft-pseudo-path
# tf reccords Dataset are stored in https://www.kaggle.com/yamsam/g2net-public-s-01, https://www.kaggle.com/yamsam/g2net-public-s-02
#PFILES = ['gs://kds-bc6ce0467b324bf699c0f253c26655b210f3d84d5f73624eecd9f0c9', 
#          'gs://kds-2d2718f40449c5e3ad78362852d1ac9328d70f6cdf5ed5f2c4d6edd1']

PFILES = [ f'{INPUT_PATH}/pseudo_tfrecord']

DEBUG = config.debug

if not config.use_pseudo:
    PFILES=[] # set [], if you do not use pseudo labeling dataset
    
if DEBUG:
#    FOLDS=[0]
    EPOCHS = 1
    IMAGE_SIZE = 32
    if TFHUB_MODEL:
        IMAGE_SIZE = 128 # 32 is not working with V3
    
    BATCH_SIZE = 32
    PFILES=[]


# In[4]:


#get_ipython().system('pip install efficientnet tensorflow_addons > /dev/null')


# In[5]:


import os
import math
import random
import re
import warnings
import gc
from pathlib import Path
from typing import Optional, Tuple

import efficientnet.tfkeras as efn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa

from scipy.signal import get_window
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from tensorflow import keras
from tensorflow.keras import mixed_precision
import tensorflow_hub as hub


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[6]:


print('tf=', tf.__version__)


# In[7]:


SAVEDIR = OUTPUT_PATH / MODEL_NAME 
SAVEDIR.mkdir(exist_ok=True)

OOFDIR = OUTPUT_PATH / MODEL_NAME 
OOFDIR.mkdir(exist_ok=True)

print ('SAVEDIR=', SAVEDIR, 'OOFDIR=', OOFDIR)


# ## Utilities

# In[8]:


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(SEED)


# In[9]:


def auto_select_accelerator():
    TPU_DETECTED = False
    try:
        if MIXED and TFHUB_MODEL is None:
          tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
          tf.config.experimental_connect_to_cluster(tpu)
          tf.tpu.experimental.initialize_tpu_system(tpu)
          strategy = tf.distribute.experimental.TPUStrategy(tpu)
          policy = mixed_precision.Policy('mixed_bfloat16')
          mixed_precision.set_global_policy(policy)
          tf.config.optimizer.set_jit(True)
        else:
          tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
          tf.config.experimental_connect_to_cluster(tpu)
          tf.tpu.experimental.initialize_tpu_system(tpu)
          strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
        TPU_DETECTED = True
    except ValueError:
        strategy = tf.distribute.get_strategy()
        
        
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    
    if MIXED and TFHUB_MODEL is None:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print ('using mixed precisison')
    return strategy, TPU_DETECTED


# In[10]:


strategy, tpu_detected = auto_select_accelerator()
AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync


# ## Data Loading

# In[11]:


gcs_paths = []
for file in FILES:
    gcs_paths.append(file)
    print(file)


# In[12]:


fold_files = []
for path in gcs_paths:
    for foldi in range(5):
        folds = []
        folds.extend(np.sort(np.array(tf.io.gfile.glob(path + f"/tr{foldi}_*.tfrecords")))) # !!!
        fold_files.append(folds)

        print(f"train_files fold{foldi}: ", len(folds))


# In[13]:


p_gcs_paths = []
for file in PFILES:
    p_gcs_paths.append(file)
    print(file)


# In[14]:


pseudo_files = []

for path in p_gcs_paths:
  pseudo_files.extend(np.sort(np.array(tf.io.gfile.glob(path + f"/*.tfrecords")))) # !!!

print(f"pseudo_files: ", len(pseudo_files))


# ## Dataset Preparation
# 
# Here's the main contribution of this notebook - Tensorflow version of on-the-fly CQT computation. Note that some of the operations used in CQT computation are not supported by TPU, therefore the implementation is not a TF layer but a function that runs on CPU.

# In[15]:


def create_cqt_kernels(
    q: float,
    fs: float,
    fmin: float,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    norm: float = 1,
    window: str = "hann",
    fmax: Optional[float] = None,
    topbin_check: bool = True
) -> Tuple[np.ndarray, int, np.ndarray, float]:
    fft_len = 2 ** _nextpow2(np.ceil(q * fs / fmin))
    
    if (fmax is not None) and (n_bins is None):
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
    elif (fmax is None) and (n_bins is not None):
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
    else:
        warnings.warn("If nmax is given, n_bins will be ignored", SyntaxWarning)
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
        
    if np.max(freqs) > fs / 2 and topbin_check:
        raise ValueError(f"The top bin {np.max(freqs)} Hz has exceeded the Nyquist frequency,                            please reduce the `n_bins`")
    
    kernel = np.zeros((int(n_bins), int(fft_len)), dtype=np.complex64)
    
    length = np.ceil(q * fs / freqs)
    for k in range(0, int(n_bins)):
        freq = freqs[k]
        l = np.ceil(q * fs / freq)
        
        if l % 2 == 1:
            start = int(np.ceil(fft_len / 2.0 - l / 2.0)) - 1
        else:
            start = int(np.ceil(fft_len / 2.0 - l / 2.0))

        sig = get_window(window, int(l), fftbins=True) * np.exp(
            np.r_[-l // 2:l // 2] * 1j * 2 * np.pi * freq / fs) / l
        
        if norm:
            kernel[k, start:start + int(l)] = sig / np.linalg.norm(sig, norm)
        else:
            kernel[k, start:start + int(l)] = sig
    return kernel, fft_len, length, freqs


def _nextpow2(a: float) -> int:
    return int(np.ceil(np.log2(a)))

def prepare_cqt_kernel(
    sr=22050,
    hop_length=512,
    fmin=32.70,
    fmax=None,
    n_bins=84,
    bins_per_octave=12,
    norm=1,
    filter_scale=1,
    window="hann"
):
    q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
    print(q)
    return create_cqt_kernels(q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax)


# In[16]:



cqt_kernels, KERNEL_WIDTH, lengths, _ = prepare_cqt_kernel(
    sr=2048,
    hop_length=HOP_LENGTH,
    fmin=FMIN,
    fmax=FMAX,
    n_bins=BINS,
    norm=NORM,
    window=WINDOW_TYPE,
    bins_per_octave=OCTAVE,
    filter_scale=SCALE)
LENGTHS = tf.constant(lengths, dtype=tf.float32)
CQT_KERNELS_REAL = tf.constant(np.swapaxes(cqt_kernels.real[:, np.newaxis, :], 0, 2))
CQT_KERNELS_IMAG = tf.constant(np.swapaxes(cqt_kernels.imag[:, np.newaxis, :], 0, 2))
PADDING = tf.constant([[0, 0],
                        [KERNEL_WIDTH // 2, KERNEL_WIDTH // 2],
                        [0, 0]])


# In[17]:


def create_cqt_image(wave, hop_length=16):
    CQTs = []
    for i in range(3):
        x = wave[i][ST:EN]
        x = tf.expand_dims(tf.expand_dims(x, 0), 2)
        x = tf.pad(x, PADDING, "REFLECT")

        CQT_real = tf.nn.conv1d(x, CQT_KERNELS_REAL, stride=hop_length, padding="VALID")
        CQT_imag = -tf.nn.conv1d(x, CQT_KERNELS_IMAG, stride=hop_length, padding="VALID")
        CQT_real *= tf.math.sqrt(LENGTHS)
        CQT_imag *= tf.math.sqrt(LENGTHS)

        CQT = tf.math.sqrt(tf.pow(CQT_real, 2) + tf.pow(CQT_imag, 2))
        CQTs.append(CQT[0])
    return tf.stack(CQTs, axis=2)


# In[18]:


def read_labeled_tfrecord(example):
    tfrec_format = {
        "wave": tf.io.FixedLenFeature([], tf.string),
        "wave_id": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return prepare_image(example["wave"], IMAGE_SIZE), tf.reshape(tf.cast(example["target"], tf.float32), [1])


def read_unlabeled_tfrecord(example, return_image_id):
    tfrec_format = {
        "wave": tf.io.FixedLenFeature([], tf.string),
        "wave_id": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return prepare_image(example["wave"], IMAGE_SIZE), example["wave_id"] if return_image_id else 0


def count_data_items(filenames): 
    # The number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

def mixup(image, label, probability=0.5, aug_batch=64 * 8):
    imgs = []
    labs = []
    for j in range(aug_batch):
        p = tf.cast(tf.random.uniform([], 0, 1) <= probability, tf.float32)
        k = tf.cast(tf.random.uniform([], 0, aug_batch), tf.int32)
        a = tf.random.uniform([], 0, 1) * p

        img1 = image[j]
        img2 = image[k]
        imgs.append((1 - a) * img1 + a * img2)
        lab1 = label[j]
        lab2 = label[k]
        labs.append((1 - a) * lab1 + a * lab2)
    image2 = tf.reshape(tf.stack(imgs), (aug_batch, IMAGE_SIZE, IMAGE_SIZE, 3))
    label2 = tf.reshape(tf.stack(labs), (aug_batch,))
    return image2, label2


def time_shift(img, shift=T_SHIFT):
    if shift > 0:
        T = IMAGE_SIZE
        P = tf.random.uniform([],0,1)
        SHIFT = tf.cast(T * P, tf.int32)
        return tf.concat([img[-SHIFT:], img[:-SHIFT]], axis=0)
    return img


def rotate(img, angle=R_ANGLE):
    if angle > 0:
        P = tf.random.uniform([],0,1)
        A = tf.cast(angle * P, tf.float32)
        return tfa.image.rotate(img, A)
    return img


def spector_shift(img, shift=S_SHIFT):
    if shift > 0:
        T = IMAGE_SIZE
        P = tf.random.uniform([],0,1)
        SHIFT = tf.cast(T * P, tf.int32)
        return tf.concat([img[:, -SHIFT:], img[:, :-SHIFT]], axis=1)
    return img

def img_aug_f(img):
#    img = time_shift(img)
#    img = spector_shift(img)
    #img = tf.image.random_flip_left_right(img) 
#    img = tf.image.random_brightness(img, 0.2)
#    img = AUGMENTATIONS_TRAIN(image=img)['image']
    # img = rotate(img)
    #print(img.shape)
    img = swap_img(img)
    return img


def swap_img(img):
   p = tf.random.uniform([],0,1)
   if p < 0.2:
     img = tf.stack([img[:,:,1], img[:,:,0], img[:,:,2]],axis=2)
     return  img
   else:
     return img


def imgs_aug_f(imgs, batch_size):
    _imgs = []
    DIM = IMAGE_SIZE
    for j in range(batch_size):
        _imgs.append(img_aug_f(imgs[j]))

    return tf.reshape(tf.stack(_imgs),(batch_size,DIM,DIM,3))


def label_positive_shift(labels):
    return labels * LABEL_POSITIVE_SHIFT


def aug_f(imgs, labels, batch_size):
    #imgs, label = mixup(imgs, labels, MIXUP_PROB, batch_size)
    imgs = imgs_aug_f(imgs, batch_size) 
    return imgs, labels

# used for whitening
window = tf.cast(np.load(DATA_PATH/'window.npy'), tf.float64)
arv_w = tf.cast(np.load(DATA_PATH/'avr_w.npy'), tf.complex64)

def whiten(c):
  #print (c.shape)
  c2 = tf.concat([tf.reverse(-c, axis=[1])[:,4096-2049:-1] + 2 *c[:,:1], c, tf.reverse(-c, axis=[1])[:,1:2049] + 2*c[:,-2:-1]],axis=1)
  #print (c2.shape)
  c3 = tf.math.real(tf.signal.ifft(tf.signal.fft(tf.cast(1e20*c2*window, tf.complex64))/arv_w))[:,2048:-2048]
  #print (c3.shape)
  return c3


def prepare_image(wave, dim=256):
    wave = tf.reshape(tf.io.decode_raw(wave, tf.float64), (3, 4096))
    #wave = tf.cast(wave, tf.float32)
    if WHITE:
      wave = whiten(wave)
    # normalized_waves = []
    # for i in range(3):
    #     normalized_wave = wave[i] - means[i]
    #     normalized_wave = normalized_wave / stds[i]

    #     normalized_waves.append(normalized_wave)
    # wave = tf.stack(normalized_waves)
    wave = tf.cast(wave, tf.float32)
    image = create_cqt_image(wave, HOP_LENGTH)
    #image = tf.keras.layers.Normalization()(image)
    image = tf.image.resize(image, size=(dim, dim))
    return tf.reshape(image, (dim, dim, 3))


def get_dataset(files, batch_size=16, repeat=False, shuffle=False, aug=True, labeled=True, return_image_ids=True):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO, compression_type="GZIP")
    ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(1024 * 10, reshuffle_each_iteration=True)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    if labeled:
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else:
        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_ids), num_parallel_calls=AUTO)

    ds = ds.batch(batch_size * REPLICAS)
    if aug:
        ds = ds.map(lambda x, y: aug_f(x, y, batch_size * REPLICAS), num_parallel_calls=AUTO)
    ds = ds.prefetch(AUTO)
    return ds


# # soft pseudo labeling dateaset

# In[19]:


def logit(x):
  return -tf.math.log(1./x - 1.)

def read_softlabeled_tfrecord(example):
    tfrec_format = {
        "wave": tf.io.FixedLenFeature([], tf.string),
        "wave_id": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.float32)
    }
    example = tf.io.parse_single_example(example, tfrec_format)

    label = tf.cast(example["target"], tf.float32)
    temperature = 2

#     if HARDEN and label > 0.75: # Only harden confident positives
#       label = tf.math.sigmoid(logit(label) * temperature)

    return prepare_image(example["wave"], IMAGE_SIZE), tf.reshape(label, [1])


def get_soft_dataset(files, batch_size=16, repeat=False, shuffle=False, aug=True, labeled=True, return_image_ids=True):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO, compression_type="GZIP")
    ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(1024 * 10, reshuffle_each_iteration=True)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    ds = ds.map(read_softlabeled_tfrecord, num_parallel_calls=AUTO)
    ds = ds.batch(batch_size * REPLICAS)
    if aug:
        ds = ds.map(lambda x, y: aug_f(x, y, batch_size * REPLICAS), num_parallel_calls=AUTO)
    ds = ds.prefetch(AUTO)
    return ds


# ## Model

# In[20]:


def build_model(size=256, efficientnet_size=0, weights="imagenet", count=0):
    inputs = tf.keras.layers.Input(shape=(size, size, 3))
    
    if TFHUB_MODEL:
      load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
      loaded_model = hub.load(TFHUB_MODEL, options=load_options)
      efn_layer = hub.KerasLayer(loaded_model, trainable=True) 
      x = efn_layer(inputs)
    else:
      efn_string= f"EfficientNetB{efficientnet_size}"
      efn_layer = getattr(efn, efn_string)(input_shape=(size, size, 3), weights=weights, include_top=False)
      x = efn_layer(inputs)
      x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    lr_decayed_fn = tf.keras.experimental.CosineDecay(1e-3, count)
    opt = tfa.optimizers.AdamW(lr_decayed_fn, learning_rate=LR)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=SMOOTHING)
    #loss = tfa.losses.SigmoidFocalCrossEntropy()

    model.compile(optimizer=opt, loss=loss, metrics=["AUC"])
    return model


# In[21]:


def get_lr_callback(batch_size=8, replicas=8):
    lr_start   = 1e-4
    lr_max     = 0.000015 * replicas * batch_size
    lr_min     = 1e-7
    lr_ramp_ep = 3
    lr_sus_ep  = 0
    lr_decay   = 0.7
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
    return lr_callback


# In[22]:


#plot
def display_one_flower(image, title, subplot, red=False):
    plt.subplot(subplot)
    plt.axis('off')
    # for i in range(3):
    #   image[i,:] -= image[i,:].min()
    #   image[i,:] /= image[i,:].max()
#    print (image.shape)
    plt.imshow(image[:,:,0].transpose())
    plt.title(title, fontsize=16, color='red' if red else 'black')
    return subplot+1

def dataset_to_numpy_util(dataset, N):
    dataset = dataset.unbatch().batch(N)
    for images, labels in dataset:
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()
        break;  
    return numpy_images, numpy_labels

def display_9_images_from_dataset(dataset):
    subplot=331
    plt.figure(figsize=(13,13))
    images, labels = dataset_to_numpy_util(dataset, 9)
    for i, image in enumerate(images):
        title = labels[i]
        subplot = display_one_flower(image, f'{title}', subplot)
        if i >= 8:

            break;
              
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()  


# In[23]:


fold_files[0]


# In[24]:


# plot CQT images
ds = get_dataset(fold_files[0], labeled=True, return_image_ids=False, repeat=False, shuffle=True, batch_size=BATCH_SIZE * 2, aug=True)
display_9_images_from_dataset(ds)


# ## Training

# In[25]:


oof_pred = []
oof_target = []
oof_ids = []

files_train_all = np.array(fold_files)


# In[ ]:





# In[ ]:


for fold in FOLDS:
    all_fold = range(NUM_FOLDS)
    files_train = list(files_train_all[(np.delete(all_fold, fold))].reshape(-1)) 
#    pseudo_files
    files_valid = files_train_all[fold]

    print("=" * 120)
    print(f"Fold {fold}")
    print("=" * 120)

    train_image_count = count_data_items(files_train + pseudo_files) # check
    valid_image_count = count_data_items(files_valid)
    
    print ('train files:', train_image_count, 'valid files:', valid_image_count)

#    tf.keras.backend.clear_session()
#    strategy, tpu_detected = auto_select_accelerator()
    with strategy.scope():
        model = build_model(
            size=IMAGE_SIZE, 
            efficientnet_size=EFFICIENTNET_SIZE,
            weights=WEIGHTS, 
            count=train_image_count // BATCH_SIZE // REPLICAS // 4)
    
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        str(SAVEDIR / f"fold{fold}.h5"), monitor="val_auc", verbose=1, save_best_only=True,
        save_weights_only=True, mode="max", save_freq="epoch"
    )

    ds_train = get_dataset(files_train, batch_size=BATCH_SIZE, shuffle=True, repeat=True, aug=True)
    

    w = [(train_image_count - valid_image_count) / train_image_count, valid_image_count / train_image_count ]

    if len(pseudo_files) > 0:
      print ('using pseudo labeling', w)
      ds_pseudo = get_soft_dataset(pseudo_files, batch_size=BATCH_SIZE, shuffle=True, repeat=True, aug=True)  
      ds_train = tf.data.experimental.sample_from_datasets([ds_train, ds_pseudo], w)

    history = model.fit(
        ds_train,
        epochs=EPOCHS,
        callbacks=[model_ckpt, get_lr_callback(BATCH_SIZE, REPLICAS)],
        steps_per_epoch=train_image_count // BATCH_SIZE // REPLICAS // 4,
        validation_data=get_dataset(files_valid, batch_size=BATCH_SIZE * 4, repeat=False, shuffle=False, aug=False),
        verbose=1
    )
    
    del ds_train
    gc.collect()
    
    print("Loading best model...")
    model.load_weights(str(SAVEDIR / f"fold{fold}.h5"))

    ds_valid = get_dataset(files_valid, labeled=False, return_image_ids=False, repeat=True, shuffle=False, batch_size=BATCH_SIZE * 2, aug=False)
    STEPS = valid_image_count / BATCH_SIZE / 2 / REPLICAS
    
    pred = model.predict(ds_valid, steps=STEPS, verbose=0)[:valid_image_count]
    print (pred.shape)
    oof_pred.append(np.mean(pred.reshape((valid_image_count, 1), order="F"), axis=1))
         
    ds_valid = get_dataset(files_valid, batch_size=BATCH_SIZE * 2, repeat=False, labeled=True, return_image_ids=True, aug=False, shuffle=False)
    oof_t = np.array([target.numpy() for _, target in iter(ds_valid.unbatch())])
    oof_target.append(oof_t)

    ds_valid = get_dataset(files_valid, batch_size=BATCH_SIZE * 2, repeat=False, shuffle=False, aug=False, labeled=False, return_image_ids=True)
    file_ids = np.array([target.numpy() for _, target in iter(ds_valid.unbatch())])
    oof_ids.append(file_ids)

    print (pred.shape, oof_t.shape, file_ids.shape)

    plt.figure(figsize=(8, 6))
    sns.distplot(oof_pred[-1])
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.plot(
        np.arange(len(history.history["auc"])),
        history.history["auc"],
        "-o",
        label="Train auc",
        color="#ff7f0e")
    plt.plot(
        np.arange(len(history.history["auc"])),
        history.history["val_auc"],
        "-o",
        label="Val auc",
        color="#1f77b4")
    
    x = np.argmax(history.history["val_auc"])
    y = np.max(history.history["val_auc"])

    xdist = plt.xlim()[1] - plt.xlim()[0]
    ydist = plt.ylim()[1] - plt.ylim()[0]

    plt.scatter(x, y, s=200, color="#1f77b4")
    plt.text(x - 0.03 * xdist, y - 0.13 * ydist, f"max auc\n{y}", size=14)

    plt.ylabel("auc", size=14)
    plt.xlabel("Epoch", size=14)
    plt.legend(loc=2)

    plt2 = plt.gca().twinx()
    plt2.plot(
        np.arange(len(history.history["auc"])),
        history.history["loss"],
        "-o",
        label="Train Loss",
        color="#2ca02c")
    plt2.plot(
        np.arange(len(history.history["auc"])),
        history.history["val_loss"],
        "-o",
        label="Val Loss",
        color="#d62728")
    
    x = np.argmin(history.history["val_loss"])
    y = np.min(history.history["val_loss"])
    
    ydist = plt.ylim()[1] - plt.ylim()[0]

    plt.scatter(x, y, s=200, color="#d62728")
    plt.text(x - 0.03 * xdist, y + 0.05 * ydist, "min loss", size=14)

    plt.ylabel("Loss", size=14)
    plt.title(f"Fold {fold + 1} - Image Size {IMAGE_SIZE}, EfficientNetB{EFFICIENTNET_SIZE}", size=18)

    plt.legend(loc=3)
    plt.savefig(OOFDIR / f"fig{fold}.png")
    plt.show()
    
    del model, ds_valid, history
    keras.backend.clear_session()
    gc.collect()


# ## OOF

# In[ ]:


oof = np.concatenate(oof_pred)
oof_ids = np.concatenate(oof_ids)
true = np.concatenate(oof_target)

auc = roc_auc_score(y_true=true, y_score=oof)
print(f"AUC: {auc:.5f}")


# In[ ]:


df = pd.DataFrame({
    "id": [i.decode("UTF-8") for i in oof_ids],
    "y_true": true.reshape(-1),
    "y_pred": oof.astype(float)
})
df.head()


# In[ ]:


df.to_csv(OOFDIR / f"oof.csv", index=False)


# In[ ]:


df = pd.read_csv(OOFDIR / f"oof.csv")

auc = roc_auc_score(y_true=true, y_score=oof)


# In[ ]:


print ('oof auc=', auc)

