#!/usr/bin/env python
# coding: utf-8

# ## Config

# In[1]:


from src.config import OUTPUT_PATH, INPUT_PATH, DATA_PATH, MODEL_PATH
from src.utils import prepare_args


# In[2]:


config = prepare_args()
print ('config=', config)


# In[3]:


config.debug


# In[4]:


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

# https://www.kaggle.com/yamsam/g2net-tf-on-the-fly-cqt-tpu-inference-path
FILES_TEST =[
    f'{INPUT_PATH}/test_tfrecord'
]

DEBUG=config.debug

if DEBUG:
    TTA=False
#    FOLDS=[0]
    print (FOLDS)
    IMAGE_SIZE = 32
    if TFHUB_MODEL:
        IMAGE_SIZE = 128 # 32 is not working with V3
    
    BATCH_SIZE = 32


# In[5]:


#get_ipython().system('pip install efficientnet tensorflow_addons > /dev/null')


# In[6]:


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
#from kaggle_datasets import KaggleDatasets
from scipy.signal import get_window
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from tensorflow.keras import mixed_precision
import tensorflow_hub as hub

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[7]:


print('tf:',tf.__version__)


# ## Utilities

# In[8]:


if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


# In[9]:


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(SEED)


# In[10]:


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

    return strategy, TPU_DETECTED


# In[11]:


strategy, tpu_detected = auto_select_accelerator()
AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync


# ## Data Loading

# In[12]:


gcs_paths = []
for file in FILES_TEST:
    gcs_paths.append(file)
    print(file)


# In[13]:


all_files = []

for path in gcs_paths:
    all_files.extend(np.sort(np.array(tf.io.gfile.glob(path + f"/*.tfrecords"))))

print('test_files: ', len(all_files))


# ## Dataset Preparation
# 
# Here's the main contribution of this notebook - Tensorflow version of on-the-fly CQT computation. Note that some of the operations used in CQT computation are not supported by TPU, therefore the implementation is not a TF layer but a function that runs on CPU.

# In[14]:


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


# In[15]:



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


# In[16]:


ORDER=[0,0,0]
def create_cqt_image(wave, hop_length=16):
    CQTs = []
    for i in ORDER:
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


# In[17]:


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
        ds = ds.shuffle(1024 * 2)
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


# ## Model

# In[18]:


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


# In[19]:


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


# In[20]:


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


# ## Inference

# In[21]:


files_test_all = np.array(all_files)
all_test_preds = []


# In[22]:


# if DEBUG:
#     files_test_all = [files_test_all[0]]
#     print('debug:only use 1file', files_test_all)


# In[23]:


# def count_data_items(filenames): 
#     # The number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
#     n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
#     return np.sum(n)

# count_data_items_test = count_data_items

# #for arais dataset
# def count_data_items(fileids):
#     return len(fileids) * 5600 # 28000

def count_data_items_test(fileids):
    return len(fileids) * 22600

print (count_data_items_test(files_test_all))


# In[24]:


len(files_test_all)


# In[25]:


with strategy.scope():
    model = build_model(
        size=IMAGE_SIZE,
        efficientnet_size=EFFICIENTNET_SIZE,
        weights=WEIGHTS,
        count=0)


# In[26]:


FOLDS


# In[27]:


ORDER=[0,1,2]

for i in FOLDS:
    print(f"Load weight for Fold {i + 1} model")
    model.load_weights(f"{MODEL_PATH}/{MODEL_NAME}/fold{i}.h5")
    ds_test = get_dataset(files_test_all, batch_size=BATCH_SIZE * 2, repeat=True, shuffle=False, aug=False, labeled=False, return_image_ids=False)
    STEPS = count_data_items_test(files_test_all) / BATCH_SIZE / 2 / REPLICAS
    pred = model.predict(ds_test, verbose=1, steps=STEPS)[:count_data_items_test(files_test_all)]

    all_test_preds.append(pred.reshape(-1))

    gc.collect()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()


# In[28]:


if TTA:
        
    ORDER=[1,0,2]

    for i in FOLDS:
        print(f"Load weight for Fold {i + 1} model")
        model.load_weights(f"{MODEL_PATH}/{MODEL_NAME}/fold{i}.h5")

        ds_test = get_dataset(files_test_all, batch_size=BATCH_SIZE * 2, repeat=True, shuffle=False, aug=False, labeled=False, return_image_ids=False)
        STEPS = count_data_items_test(files_test_all) / BATCH_SIZE / 2 / REPLICAS
        pred = model.predict(ds_test, verbose=1, steps=STEPS)[:count_data_items_test(files_test_all)]

        all_test_preds.append(pred.reshape(-1))

        gc.collect()
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        


# In[29]:


ds_test = get_dataset(files_test_all, batch_size=BATCH_SIZE * 2, repeat=False, shuffle=False, aug=False, labeled=False, return_image_ids=True)
file_ids = np.array([target.numpy() for img, target in iter(ds_test.unbatch())])


# In[30]:


test_pred = np.zeros_like(all_test_preds[0])
for i in range(len(all_test_preds)):
    test_pred += all_test_preds[i] / len(all_test_preds)
    
test_df = pd.DataFrame({
    "id": [i.decode("UTF-8") for i in file_ids],
    "target": test_pred.astype(float)
})

#test_df.head()


# In[31]:


test_df.to_csv(f"{OUTPUT_PATH}/{MODEL_NAME}/submission.csv", index=False)

