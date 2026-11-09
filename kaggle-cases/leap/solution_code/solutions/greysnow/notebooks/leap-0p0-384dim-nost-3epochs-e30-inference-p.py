DEBUG = False

%%capture
import os, gc
os.environ["KERAS_BACKEND"] = "tensorflow"
!pip install keras==2.15.0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import backend as K
import tensorflow as tf
print(tf.__version__)

import numpy as np
from sklearn.model_selection import KFold
import sklearn
import matplotlib.pyplot as plt
import pickle
import math
import shutil

save_folder = '/kaggle/working'
try:
    os.mkdir(save_folder)
except Exception as e:
    print('exception error:')
    print(e)

# Configure Strategy. Assume TPU...if not set default for GPU
tpu = None
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    tpu_connceted = True
    print("on TPU")
    print("REPLICAS: ", strategy.num_replicas_in_sync)
except Exception as e:
    print('exception error:')
    print(e)
    strategy = tf.distribute.get_strategy()

print(strategy)

gcp_data = '/kaggle/input/leap-gcp-path-tfrecs-s'
all_lens_lr_concat = pickle.load(open(f'{gcp_data}/all_lens_lr_concat.p', 'br'))
all_tffiles_lr_concat = pickle.load(open(f'{gcp_data}/all_tffiles_lr_concat.p', 'br'))
all_lens_hr_concat = pickle.load(open(f'{gcp_data}/all_lens_hr_concat.p', 'br'))
all_tffiles_hr_concat = pickle.load(open(f'{gcp_data}/all_tffiles_hr_concat.p', 'br'))

gcp_data = '/kaggle/input/leap-gcp-path-tfrecs-hr'
all_lens_hr_concat_TOTAL = pickle.load(open(f'{gcp_data}/all_lens_hr_concat_TOTAL.p', 'br'))
all_tffiles_hr_concat_TOTAL = pickle.load(open(f'{gcp_data}/all_tffiles_hr_concat_TOTAL.p', 'br'))

%%time

input_folder='/kaggle/working/input'
try:
    os.mkdir(input_folder)
except Exception as e:
    print('exception error:')
    print(e)

stds_path = '/kaggle/input/leap-sample-submission-stds-ds/stds.p'
stds_new_path = '/kaggle/input/leap-sample-submission-stds-ds/stds_new.p'

FEAT_COLS_path = '/kaggle/input/leap-sample-submission-stds-ds/FEAT_COLS.p'
TARGET_COLS_path = '/kaggle/input/leap-sample-submission-stds-ds/TARGET_COLS.p'
data_path_MSM = '/kaggle/input/leap-msm-ds'
data_path_gdata ='/kaggle/input/leap-gdata-ds'

y_max_per_col_all_path = '/kaggle/input/leap-sample-submission-stds-ds/y_max_per_col_all.p'
y_min_per_col_all_path = '/kaggle/input/leap-sample-submission-stds-ds/y_min_per_col_all.p'

[area, hyai, hyam, hybi, hybm, ilev, lat, lev, lon] = pickle.load(open(f'{data_path_gdata}/gdata.p', 'br'))
lat = tf.convert_to_tensor(lat)
lon = tf.convert_to_tensor(lon)
lat_mean = tf.math.reduce_mean(lat)
lat_std = tf.math.reduce_std(lat)
lon_mean = tf.math.reduce_mean(lon)
lon_std = tf.math.reduce_std(lon)

lat = (lat - lat_mean)/lat_std
lon = (lon - lon_mean)/lon_std

lat = tf.cast(lat, tf.float32)
lon = tf.cast(lon, tf.float32)

day_time_sin = tf.convert_to_tensor(np.sin(np.asarray(range(12*31))*2*np.pi/(12*31)), dtype = tf.float32)
day_time_cos = tf.convert_to_tensor(np.cos(np.asarray(range(12*31))*2*np.pi/(12*31)), dtype = tf.float32)
sec_time_sin = tf.convert_to_tensor(np.sin(np.asarray(range(24*60*60))*2*np.pi/(24*60*60)), dtype = tf.float32)
sec_time_cos = tf.convert_to_tensor(np.cos(np.asarray(range(24*60*60))*2*np.pi/(24*60*60)), dtype = tf.float32)

[area_hr, hyai_hr, hyam_hr, hybi_hr, hybm_hr, ilev_hr, lat_hr, lev_hr, lon_hr] = pickle.load(open(f'{data_path_gdata}/data_hr.p', 'br'))
lat_hr = tf.convert_to_tensor(lat_hr)
lon_hr = tf.convert_to_tensor(lon_hr)
lat_mean_hr = tf.math.reduce_mean(lat_hr)
lat_std_hr = tf.math.reduce_std(lat_hr)

lat_hr = (lat_hr - lat_mean)/lat_std
lon_hr = (lon_hr - lon_mean)/lon_std

lat_hr = tf.cast(lat_hr, tf.float32)
lon_hr = tf.cast(lon_hr, tf.float32)

FEAT_COLS = pickle.load(open(FEAT_COLS_path, 'br'))
TARGET_COLS = pickle.load(open(TARGET_COLS_path, 'br'))

X_cols = FEAT_COLS
y_cols = TARGET_COLS

X_cols_series_parents = ['state_t', 'state_q0001', 'state_q0002', 'state_q0003', 'state_u', 'state_v',
                         'pbuf_ozone', 'pbuf_CH4', 'pbuf_N2O']
X_cols_series = [x for x in X_cols if np.mean([x.startswith(y) for y in X_cols_series_parents])>0]
X_cols_series_NOT = [x for x in X_cols if x not in X_cols_series]
print(len(X_cols_series)+len(X_cols_series_NOT) == len(X_cols))
X_cols_series_indices = np.asarray([np.where(np.asarray(X_cols) == x)[0][0] for x in X_cols_series])
X_cols_series_NOT_indices = np.asarray([np.where(np.asarray(X_cols) == x)[0][0] for x in X_cols_series_NOT])

y_cols_series_parents = ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v']
y_cols_series = [x for x in y_cols if np.mean([x.startswith(y) for y in y_cols_series_parents])>0]
y_cols_series_NOT = [x for x in y_cols if x not in y_cols_series]
print(len(y_cols_series)+len(y_cols_series_NOT) == len(y_cols))
y_cols_series_indices = np.asarray([np.where(np.asarray(y_cols) == x)[0][0] for x in y_cols_series])
y_cols_series_NOT_indices = np.asarray([np.where(np.asarray(y_cols) == x)[0][0] for x in y_cols_series_NOT])

stds = pickle.load(open(stds_path, 'br'))
stds_new = pickle.load(open(stds_new_path, 'br'))
stds_new = stds_new.astype(np.float64)

if DEBUG:
    N_EPOCHS = 40
    batch_size = 2
else:
    N_EPOCHS = 40
    batch_size = 2048
val_batch_size = batch_size
N_WARMUP_EPOCHS = 0
LR_MAX = 1e-3
LR_MIN = 8.4e-4
WD_RATIO = 4.0
WARMUP_METHOD = "exp"
lr_factor = 0.7
PAD = 0.0

print(len(all_lens_lr_concat))
print(len(all_lens_hr_concat))
print(len(all_lens_hr_concat_TOTAL))
print(np.sum(all_lens_lr_concat))
print(np.sum(all_lens_hr_concat_TOTAL))

rand_seed = np.random.randint(10000)
print(rand_seed)
rng = np.random.default_rng(rand_seed)
rng.shuffle(all_tffiles_hr_concat_TOTAL[20:])

if DEBUG:
    val_1_tffiles = all_tffiles_lr_concat[:1]
    val_2_tffiles = all_tffiles_lr_concat[:1]
    val_3_tffiles = all_tffiles_hr_concat[:1]
    val_4_tffiles = all_tffiles_lr_concat[:1]
    train_tffiles = all_tffiles_lr_concat[1:3]
    lens_tffiles = np.concatenate([all_lens_lr_concat[1:2]], axis = 0)
else:
    val_1_tffiles = all_tffiles_lr_concat[:20]
    val_2_tffiles = all_tffiles_lr_concat[20:40]
    val_3_tffiles = all_tffiles_hr_concat[0:20]
    val_4_tffiles = all_tffiles_lr_concat[40:60]
    train_tffiles = []
    for i in range(11):
        train_tffiles_temp = all_tffiles_lr_concat[40:].copy()
        train_tffiles_temp_HR = all_tffiles_hr_concat_TOTAL[20+i*8060:20+(i+1)*8060].copy()
        tffiles_temp_concat = np.concatenate([train_tffiles_temp, train_tffiles_temp_HR], axis = 0)
        np.random.shuffle(tffiles_temp_concat)
        train_tffiles.append(tffiles_temp_concat)
    train_tffiles = np.concatenate(train_tffiles)
    lens_tffiles = np.concatenate([all_lens_lr_concat[40:], all_lens_hr_concat_TOTAL[20:20+8060]], axis = 0)

num_val_1 = len(val_1_tffiles)*5000
num_val_2 = len(val_2_tffiles)*5000
num_val_3 = len(val_3_tffiles)*5000
num_val_4 = len(val_4_tffiles)*5000
num_train = np.sum(lens_tffiles)
if DEBUG:
    num_val_1 = 64
    num_val_2 = 64
    num_val_3 = 64
    num_val_4 = 64
    num_train = 64
print(f'num train: {num_train}')

steps_per_epoch = int(num_train/batch_size/10)
if DEBUG:
    steps_per_epoch = 2
    
val_steps_per_epoch = num_val_1//val_batch_size
val_steps_per_epoch_2 = num_val_2//val_batch_size
val_steps_per_epoch_3 = num_val_3//val_batch_size
val_steps_per_epoch_4 = num_val_4//val_batch_size

val_samples_num_1 = val_steps_per_epoch*val_batch_size
val_samples_num_2 = val_steps_per_epoch_2*val_batch_size
val_samples_num_3 = val_steps_per_epoch_3*val_batch_size
val_samples_num_4 = val_steps_per_epoch_4*val_batch_size
print(f'steps_per_epoch: {steps_per_epoch}')
print(val_steps_per_epoch)
print(val_samples_num_1)
print(val_samples_num_2)
print(val_samples_num_3)
print(val_samples_num_4)


epoch_samples_num = steps_per_epoch*batch_size
if DEBUG:
    epoch_samples_num = 10
print(epoch_samples_num)

y_min_per_col_all = pickle.load((open(y_min_per_col_all_path, 'br')))
y_max_per_col_all = pickle.load((open(y_max_per_col_all_path, 'br')))

min_y = pickle.load((open(f'{data_path_MSM}/min_y.p', 'br')))
max_y = pickle.load((open(f'{data_path_MSM}/max_y.p', 'br')))
mean_y = pickle.load((open(f'{data_path_MSM}/mean_y.p', 'br')))
std_y = pickle.load((open(f'{data_path_MSM}/std_y.p', 'br')))

min_col_not = pickle.load((open(f'{data_path_MSM}/min_col_not.p', 'br')))
max_col_not = pickle.load((open(f'{data_path_MSM}/max_col_not.p', 'br')))
mean_col_not = pickle.load((open(f'{data_path_MSM}/mean_col_not.p', 'br')))
std_col_not = pickle.load((open(f'{data_path_MSM}/std_col_not.p', 'br')))

min_col_not_test = pickle.load((open(f'{data_path_MSM}/min_col_not_test.p', 'br')))
max_col_not_test = pickle.load((open(f'{data_path_MSM}/max_col_not_test.p', 'br')))
mean_col_not_test = pickle.load((open(f'{data_path_MSM}/mean_col_not_test.p', 'br')))
std_col_not_test = pickle.load((open(f'{data_path_MSM}/std_col_not_test.p', 'br')))

x_col_min = pickle.load((open(f'{data_path_MSM}/x_col_min.p', 'br')))
x_col_max = pickle.load((open(f'{data_path_MSM}/x_col_max.p', 'br')))
x_col_mean = pickle.load((open(f'{data_path_MSM}/x_col_mean.p', 'br')))
x_col_std = pickle.load((open(f'{data_path_MSM}/x_col_std.p', 'br')))

x_col_min_test = pickle.load((open(f'{data_path_MSM}/x_col_min_test.p', 'br')))
x_col_max_test = pickle.load((open(f'{data_path_MSM}/x_col_max_test.p', 'br')))
x_col_mean_test = pickle.load((open(f'{data_path_MSM}/x_col_mean_test.p', 'br')))
x_col_std_test = pickle.load((open(f'{data_path_MSM}/x_col_std_test.p', 'br')))

X_total_min = pickle.load((open(f'{data_path_MSM}/X_total_min.p', 'br')))
X_total_max = pickle.load((open(f'{data_path_MSM}/X_total_max.p', 'br')))
X_total_mean = pickle.load((open(f'{data_path_MSM}/x_total_mean.p', 'br')))
X_total_std = pickle.load((open(f'{data_path_MSM}/x_total_std.p', 'br')))

X_total_min_test = pickle.load((open(f'{data_path_MSM}/X_total_min_test.p', 'br')))
X_total_max_test = pickle.load((open(f'{data_path_MSM}/X_total_max_test.p', 'br')))
X_total_mean_test = pickle.load((open(f'{data_path_MSM}/x_total_mean_test.p', 'br')))
X_total_std_test = pickle.load((open(f'{data_path_MSM}/x_total_std_test.p', 'br')))

y_total_min = pickle.load((open(f'{data_path_MSM}/y_total_min.p', 'br')))
y_total_max = pickle.load((open(f'{data_path_MSM}/y_total_max.p', 'br')))
y_total_mean = pickle.load((open(f'{data_path_MSM}/y_total_mean.p', 'br')))
y_total_std = pickle.load((open(f'{data_path_MSM}/y_total_std.p', 'br')))

stds_temp = np.where(std_y == 0, 0, 1/np.where(std_y>0, std_y, 1))
stds = stds_temp[0]

mean_col_not = mean_col_not[0]
std_col_not = std_col_not[0]
min_col_not_norm = ((np.where(min_col_not<min_col_not_test, min_col_not , min_col_not_test)-mean_col_not)/std_col_not)[0]

X_total_mean = X_total_mean[0]
X_total_std = X_total_std[0]
X_total_norm_min = ((np.where(X_total_min<X_total_min_test, X_total_min , X_total_min_test)-X_total_mean)/X_total_std)[0]

x_col_mean = x_col_mean[0]
x_col_std = x_col_std[0]
x_col_norm_min = ((np.where(x_col_min<x_col_min_test, x_col_min , x_col_min_test)-x_col_mean)/x_col_std)[0]

mean_y = mean_y[0]

x_col_min = x_col_min[0]

norm_y_min = tf.convert_to_tensor((y_min_per_col_all- mean_y)*stds)
norm_y_max = tf.convert_to_tensor((y_max_per_col_all- mean_y)*stds)

def decode_tfrec(record_bytes):
    schema = {}
    schema["x1"] = tf.io.VarLenFeature(dtype=tf.string)
    schema["x2"] = tf.io.VarLenFeature(dtype=tf.string)
    schema["x3"] = tf.io.VarLenFeature(dtype=tf.string)
    schema["x4"] = tf.io.VarLenFeature(dtype=tf.string)
    schema["time"] = tf.io.VarLenFeature(dtype=tf.int64)
    schema["gidx"] = tf.io.VarLenFeature(dtype=tf.int64)
    schema["idx"] = tf.io.VarLenFeature(dtype=tf.int64)
    schema["batch_index"] = tf.io.VarLenFeature(dtype=tf.int64)
    features = tf.io.parse_single_example(record_bytes, schema)

    data_cols = tf.sparse.to_dense(features["x1"])
    data_cols = tf.io.parse_tensor(data_cols[0], out_type=tf.float64)
    data_cols_not = tf.sparse.to_dense(features["x2"])
    data_cols_not = tf.io.parse_tensor(data_cols_not[0], out_type=tf.float64)
    data_cols_targets = tf.sparse.to_dense(features["x3"])
    data_cols_targets = tf.io.parse_tensor(data_cols_targets[0], out_type=tf.float64)
    data_cols_not_target = tf.sparse.to_dense(features["x4"])
    data_cols_not_target = tf.io.parse_tensor(data_cols_not_target[0], out_type=tf.float64)

    time = tf.sparse.to_dense(features["time"])
    gidx = tf.sparse.to_dense(features["gidx"])
    gidx = gidx[0]
    res = 0
    if gidx<0:
        gidx = -gidx-1
        res = 1
    idx = tf.sparse.to_dense(features["idx"])
    batch_index = tf.sparse.to_dense(features["batch_index"])

    out = {}
    out['X_col']  = data_cols
    out['X_col_not']  = data_cols_not
    out['y']  = tf.concat([data_cols_targets, data_cols_not_target], axis = 0)
    out['time']  = time
    out['gidx']  = gidx
    out['res'] = res
    return out

def get_timespace(x, lat_hr, lon_hr, lat, lon, day_time_sin, day_time_cos, sec_time_sin, sec_time_cos):
    res = x['res']
    gidx = x['gidx']
    time = x['time']
    if res == 0:
        x_lat = lat_hr[gidx]
        x_lon = lon_hr[gidx]
    else:
        x_lat = lat[gidx]
        x_lon = lon[gidx]

    day_sin = day_time_sin[(time[1]-1)*31+time[2]-1]
    day_cos = day_time_cos[(time[1]-1)*31+time[2]-1]
    sec_sin = sec_time_sin[time[3]]
    sec_cos = sec_time_cos[time[3]]

    x.pop('gidx')
    x.pop('time')

    x['lat']  = x_lat
    x['lon']  = x_lon
    x['day_sin']  = day_sin
    x['day_cos']  = day_cos
    x['sec_sin']  = sec_sin
    x['sec_cos']  = sec_cos
    return x

def get_labels(x):
    return x['y']

def get_labels_2(x):
    return x['normed_y']

def get_normed(x, valid = False):
    y_norm =  (x['y'] - mean_y)*stds
    y_norm = tf.cast(y_norm, tf.float32)
    
    x_col_not_norm = (x['X_col_not'] - mean_col_not)/std_col_not

    x_total_norm = tf.reshape(x['X_col'], [-1, 9, 60])
    x_total_norm = (x_total_norm - X_total_mean)/X_total_std
    x_total_norm = tf.reshape(x_total_norm, [60*x_total_norm.shape[1]])


    x_col_norm = (x['X_col'] - x_col_mean)/x_col_std
    x_col_norm_log = tf.where((x_col_norm-x_col_norm_min+1)>=1, tf.math.log(x_col_norm-x_col_norm_min+1),
                                    -tf.math.log(1+1-(x_col_norm-x_col_norm_min+1)))


    wind = tf.reshape(x['X_col'], [-1, 9, 60])
    wind = wind[:, 4:6, :]
    wind = (tf.math.reduce_sum(wind**2, axis = 1))**0.5
    wind = wind[0]
    wind = ((wind - tf.math.reduce_mean((tf.reshape(x_col_mean, [9, 60])[4:6]), axis = 0))/
                    tf.math.reduce_sum((tf.reshape(x_col_std, [9, 60])[4:6]), axis = 0))


    cutoff = 30
    square_cutoff = cutoff**0.5
    x_col_norm = tf.where(x_col_norm>cutoff, x_col_norm**0.5+cutoff-square_cutoff, x_col_norm)
    x_col_norm = tf.where(x_col_norm<-cutoff, -tf.math.abs(x_col_norm)**0.5-cutoff+square_cutoff, x_col_norm)

    wind = tf.where(wind>cutoff, wind**0.5+cutoff-square_cutoff, wind)
    wind = tf.where(wind<-cutoff, -tf.math.abs(wind)**0.5-cutoff+square_cutoff, wind)


    x_col_not_norm = tf.cast(x_col_not_norm, tf.float32)
    x_total_norm = tf.cast(x_total_norm, tf.float32)
    x_col_norm = tf.cast(x_col_norm, tf.float32)
    x_col_norm_log = tf.cast(x_col_norm_log, tf.float32)
    wind = tf.cast(wind, tf.float32)

    cutoff_2 = 86.0
    log_cutoff = tf.math.log(cutoff_2)
    x_col_norm = tf.where(x_col_norm>cutoff_2, tf.math.log(x_col_norm)+cutoff_2-log_cutoff, x_col_norm)
    x_col_norm = tf.where(x_col_norm<-cutoff_2, -tf.math.log(-x_col_norm)-cutoff_2+log_cutoff, x_col_norm)

    x_col_not_norm = tf.where(x_col_not_norm>cutoff_2, tf.math.log(x_col_not_norm)+cutoff_2-log_cutoff, x_col_not_norm)
    x_col_not_norm = tf.where(x_col_not_norm<-cutoff_2, -tf.math.log(-x_col_not_norm)-cutoff_2+log_cutoff, x_col_not_norm)

    x_total_norm = tf.where(x_total_norm>cutoff_2, tf.math.log(x_total_norm)+cutoff_2-log_cutoff, x_total_norm)
    x_total_norm = tf.where(x_total_norm<-cutoff_2, -tf.math.log(-x_total_norm)-cutoff_2+log_cutoff, x_total_norm)

    wind = tf.where(wind>cutoff_2, tf.math.log(wind)+cutoff_2-log_cutoff, wind)
    wind = tf.where(wind<-cutoff_2, -tf.math.log(-wind)-cutoff_2+log_cutoff, wind)

    x.pop('X_col')
    x.pop('X_col_not')
    x.pop('y')

    x['normed_y'] = y_norm
    x['x_col_not_norm'] = x_col_not_norm
    x['x_total_norm'] = x_total_norm
    x['x_col_norm'] = x_col_norm
    x['x_col_norm_log'] = x_col_norm_log
    x['wind'] = wind
    x['epoch'] = 0
    return x

def get_concats(x):
    x['concat_x'] = tf.concat([x['x_total_norm'], x['x_col_norm'], x['x_col_norm_log'],x['wind']], axis = 0)
    x['concat_x_col_not'] = tf.concat([x['x_col_not_norm']], axis = 0)
    x['concat_target'] = tf.concat([x['normed_y'], x['lat'][None], x['lon'][None],
                                    x['day_sin'][None], x['day_cos'][None], x['sec_sin'][None], x['sec_cos'][None]], axis = 0)

    x.pop('x_total_norm')
    x.pop('x_col_norm')
    x.pop('x_col_norm_log')
    x.pop('x_col_not_norm')
    x.pop('lat')
    x.pop('lon')
    x.pop('day_sin')
    x.pop('day_cos')
    x.pop('sec_sin')
    x.pop('sec_cos')
    x.pop('wind')
    return x

def get_cols(x):
    return x['concat_x']

def get_cols_not(x):
    return x['concat_x_col_not']

def get_stateq002(x):
    return x['X_col'][120:120+30]

def get_final_output(x):
    x_concat = tf.concat([x['concat_x'], x['concat_x_col_not']], axis = 0)
    concat_target = x['concat_target']
    return x_concat, concat_target

ds = tf.data.TFRecordDataset(
    val_1_tffiles, num_parallel_reads=tf.data.AUTOTUNE, compression_type = 'GZIP').prefetch(tf.data.AUTOTUNE)
ds = ds.map(decode_tfrec, tf.data.AUTOTUNE)
ds = ds.map(lambda x: get_timespace(x, lat_hr, lon_hr, lat, lon, day_time_sin, day_time_cos, sec_time_sin, sec_time_cos), tf.data.AUTOTUNE)
ds = ds.map(get_normed, tf.data.AUTOTUNE)
ds = ds.map(get_concats, tf.data.AUTOTUNE)

ds_col = ds.map(get_cols, tf.data.AUTOTUNE)
batch = next(iter(ds_col))
col_len = batch.shape[0]
print(col_len)

ds_col_not = ds.map(get_cols_not, tf.data.AUTOTUNE)
batch = next(iter(ds_col_not))
col_not_len = batch.shape[0]
print(col_not_len)

ds = ds.map(get_final_output, tf.data.AUTOTUNE)
batch = next(iter(ds))
print(batch[0].shape, batch[1].shape)
col_num_y = batch[1].shape[0]
print(col_num_y)

def get_dataset(tffiles, batch_size, shuffle, split = 'valid', to_repeat = False, drop_reminder = True, cache = False, valid = False):
    ds = tf.data.TFRecordDataset(
        tffiles, num_parallel_reads=tf.data.AUTOTUNE, compression_type = 'GZIP').prefetch(tf.data.AUTOTUNE)
    ds = ds.map(decode_tfrec, tf.data.AUTOTUNE)
    ds = ds.map(lambda x: get_timespace(x, lat_hr, lon_hr, lat, lon, day_time_sin, day_time_cos, sec_time_sin, sec_time_cos), tf.data.AUTOTUNE)


    if DEBUG:
        ds = ds.take(64)

    if cache:
        ds = ds.cache()
        samples_num = ds.reduce(0, lambda x,_: x+1).numpy()

    if shuffle:
        ds = ds.shuffle(shuffle, reshuffle_each_iteration = True)
    if to_repeat:
        ds = ds.repeat()

    ds_label = ds.map(get_labels, tf.data.AUTOTUNE)
    ds_q002 = ds.map(get_stateq002, tf.data.AUTOTUNE)

    ds = ds.map(lambda x: get_normed(x, valid), tf.data.AUTOTUNE)
    ds = ds.map(get_concats, tf.data.AUTOTUNE)

    ds_label_2 = ds.map(get_labels_2, tf.data.AUTOTUNE)

    ds = ds.map(get_final_output, tf.data.AUTOTUNE)

    ds = ds.padded_batch(
                batch_size, padding_values=(PAD, PAD),
        padded_shapes=([col_len+col_not_len],[col_num_y]), drop_remainder=drop_reminder)

    ds_label = ds_label.batch(batch_size)
    ds_label_2 = ds_label_2.batch(batch_size)
    ds_q002 = ds_q002.batch(batch_size)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, ds_label, ds_label_2, ds_q002

val_1_ds, val_1_labels_ds, val_1_labels_ds_2, val_1_ds_q0002 = get_dataset(
    val_1_tffiles, val_batch_size, shuffle = False, cache = True)

batch = next(iter(val_1_ds))
batch[0].shape ,batch[1].shape

class GLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, x, mask=None):
        x,gate = tf.split(x, 2, axis = -1)
        x = x*tf.keras.activations.swish(gate)
        return x

class GLUMlp(tf.keras.layers.Layer):
    def __init__(self, dim_expand, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim_expand = dim_expand
        self.dim = dim
        self.dense_1 = tf.keras.layers.EinsumDense("abc,cd->abd",output_shape=(None, self.dim_expand), activation = 'linear', bias_axes = 'd')
        self.glu_1 = GLU()
        self.dense_2 = tf.keras.layers.EinsumDense("abc,cd->abd",output_shape=(None, self.dim), activation = 'linear', bias_axes = 'd')
    def call(self, x, training = False):
        x = self.dense_1(x)
        x = self.glu_1(x)
        x = self.dense_2(x)
        return x

class ScaleBias(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.scale_bias = tf.keras.layers.EinsumDense("abc,c->abc",output_shape=(None, input_shape[-1]),activation = 'linear', bias_axes = 'c')
    def call(self, x, mask=None):
        return self.scale_bias(x)

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)
        self.ffn = GLUMlp(feed_forward_dim, embed_dim)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.scale_bias_1 = ScaleBias()
        self.scale_bias_2 = ScaleBias()
    def call(self, x, training = None):
        residual = x
        x = self.att(x, x)
        x = self.scale_bias_1(x)
        x = self.layer_norm_1(x + residual)
        residual = x
        x = self.ffn(x, training = training)
        x = self.scale_bias_2(x)
        x = self.layer_norm_2(x + residual)
        return x


class ECA(tf.keras.layers.Layer):
    # TF implementation from https://www.kaggle.com/code/hoyso48/1st-place-solution-training
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)
    def call(self, inputs):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:,None,:]
        return inputs * nn


class HeadDense(tf.keras.layers.Layer):
    def __init__(self, head_dim, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim
    def build(self, input_shape):
        self.length = input_shape[1]
        self.dim = input_shape[2]
        self.dense = tf.keras.layers.EinsumDense("abc,cd->abd",output_shape=(self.length, self.head_dim), activation = 'swish', bias_axes = 'd')
    def call(self, x):
        x = self.dense(x)
        return x

class Conv1DBlockSqueezeformer(tf.keras.layers.Layer):
    def __init__(self, channel_size, kernel_size, dilation_rate=1,
                 expand_ratio=4, se_ratio=0.25, activation='swish', name=None, **kwargs):
        super().__init__()
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.activation = activation
        self.scale_bias = ScaleBias()
        self.glu_layer = GLU()
        self.ffn = GLUMlp(channel_size*4, channel_size)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.scale_bias_1 = ScaleBias()
        self.scale_bias_2 = ScaleBias()
    def build(self, input_shape):
        self.length = input_shape[1]
        self.channels_in = input_shape[2]
        self.channels_expand = self.channels_in * self.expand_ratio
        self.dwconv = tf.keras.layers.DepthwiseConv1D(self.kernel_size,dilation_rate=self.dilation_rate,padding='same',use_bias=False)
        self.BatchNormalization_layer = tf.keras.layers.BatchNormalization(momentum=0.95)
        self.conv_activation = tf.keras.layers.Activation(self.activation)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ECA_layer = ECA()
        self.expand = tf.keras.layers.EinsumDense("abc,cd->abd",output_shape=(self.length, self.channels_expand), activation = 'linear', bias_axes = 'd')
        self.project =tf.keras.layers.EinsumDense("abc,cd->abd",output_shape=(self.length, self.channel_size), activation = 'linear', bias_axes = 'd')
    def call(self, x, training = None):
        skip = x
        x = self.expand(x)
        x = self.glu_layer(x)
        x = self.dwconv(x)
        x = self.BatchNormalization_layer(x)
        x = self.conv_activation(x)
        x = self.ECA_layer(x)
        x = self.project(x)
        x = self.scale_bias_1(x)

        x = x+skip

        residual = x
        x = self.ffn(x)
        x = self.scale_bias_2(x)
        x = self.layer_norm_2(x + residual)
        return x

class Reshape1(tf.keras.layers.Layer):
    def __init__(self, col_len, **kwargs):
        super().__init__(**kwargs)
        self.col_len = col_len
    def call(self, x):
        x_seq = x[:, :self.col_len]
        x_seq = tf.reshape(x_seq, [-1, tf.cast(self.col_len/60, tf.int32), 60])
        x_seq = tf.transpose(x_seq, [0, 2, 1])
        x_seq_N = x[:, self.col_len:]

        x_seq_N = tf.expand_dims(x_seq_N, axis = 1)
        x_seq_N = tf.repeat(x_seq_N, 60, axis = 1)

        x = tf.concat([x_seq, x_seq_N], axis = -1)
        return x

class Reshape2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, x_pred, x_confidence):
        x = x_pred
        x_seq = x[:, :, :6]
        x_seq = tf.transpose(x_seq, [0,2,1])
        x_seq = tf.reshape(x_seq, [-1, 60*6])
        x_seq_N = x[:, :, 6:]
        x_seq_N = tf.reduce_mean(x_seq_N, axis = 1)
        x1 = tf.concat([x_seq, x_seq_N], axis = -1)

        x = x_confidence
        x_seq = x[:, :, :6]
        x_seq = tf.transpose(x_seq, [0,2,1])
        x_seq = tf.reshape(x_seq, [-1, 60*6])
        x_seq_N = x[:, :, 6:]
        x_seq_N = tf.reduce_mean(x_seq_N, axis = 1)
        x2 = tf.concat([x_seq, x_seq_N], axis = -1)

        x = tf.concat([x1, x2], axis = -1)
        return x

loss_mask = (stds_new!=0)[None]
loss_mask = tf.cast(loss_mask, tf.float32)
temp = np.ones((loss_mask.shape))
temp[0, 120+12:120+26] = 0
loss_mask = loss_mask*temp
loss_mask = tf.pad(loss_mask, [[0,0], [0,col_num_y - len(stds_new)]], constant_values = 1)

def loss_fn(x, preds):
    confidence = preds[:, col_num_y:]
    preds = preds[:, :col_num_y]
    loss = tf.math.abs(x-preds)
    loss = loss*loss_mask
    loss_2 = tf.math.abs(loss-confidence)
    loss_2 = loss_2*loss_mask
    return tf.reduce_mean(loss+loss_2)

def metric_mae(x, preds):
    x = x[:, :368]
    preds = preds[:, :368]
    loss = tf.math.abs(x-preds)
    loss = loss*loss_mask[:, :368]
    return tf.reduce_mean(loss)

def get_model(dim = 384, head_dim = 2048):
    with strategy.scope():
        inp1 = tf.keras.Input([col_len+col_not_len])
        x = inp1

        x = Reshape1(col_len)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        groups = 1
        conv_filter = 15
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)

        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)

        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)

        x = HeadDense(head_dim)(x)
        x = GLUMlp(head_dim*2, head_dim)(x)

        x_pred = tf.keras.layers.Dense(20)(x)
        x_confidence = tf.keras.layers.Dense(20)(x)

        x = Reshape2()(x_pred, x_confidence)

        model = tf.keras.Model(inp1, x)
        return model

if DEBUG:
    model = get_model(4, head_dim = 4)
else:
    model = get_model()

with strategy.scope():
    loss = loss_fn
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005)
    if DEBUG:
        model.compile(loss=loss, optimizer=optimizer)
    else:
        model.compile(loss=loss, optimizer=optimizer, metrics = [metric_mae], steps_per_execution = 100)

model(batch[0])
model.summary()

val_1_labels = [x.numpy() for x in val_1_labels_ds]
val_1_labels = np.concatenate(val_1_labels, axis = 0)
val_1_labels = val_1_labels[:val_samples_num_1]

val_1_labels_2 = [x.numpy() for x in val_1_labels_ds_2]
val_1_labels_2 = np.concatenate(val_1_labels_2, axis = 0)
val_1_labels_2 = val_1_labels_2[:val_samples_num_1]

val_1_q0002 = [x.numpy() for x in val_1_ds_q0002]
val_1_q0002 = np.concatenate(val_1_q0002, axis = 0)
val_1_q0002 = val_1_q0002[:val_samples_num_1]

val_metric_list = []

jump = 2
if DEBUG:
    jump = 1

def valid_sub_func(model, val_dataset, val_labels, val_labels_valid2, val_q0002, to_save = False, idx = -1, epoch = -1):
    val_labels = val_labels.astype(np.float64)

    preds = model.predict(val_dataset, verbose = 0, batch_size = batch_size, steps = val_steps_per_epoch)
    preds = preds[:, :368]
    if to_save:
        pickle.dump(preds, open(f'{save_folder_val}/preds_{idx}_{epoch}.p', 'bw'))
    preds = preds.astype(np.float64)

    metrics = np.asarray([sklearn.metrics.r2_score(val_labels[:, i], preds[:, i]) for i in range(368)])

    for i in range(len(metrics)):
        if metrics[i]<0:
            preds[:,i] = 0

    #print('Metric: ------------------')
    metric = sklearn.metrics.r2_score(val_labels, preds)
    #print(metric)

    preds = preds + mean_y.reshape(1,-1)*stds
    preds[:, np.where(stds_new == 0)] = 0
    preds = preds/np.where(stds>0, stds, 1)
    val_labels_valid2 = np.where(stds_new == 0, 0, val_labels_valid2)

    #print('Metric: ------------------')
    metric = sklearn.metrics.r2_score(val_labels_valid2,preds)
    #print(metric)

    preds[:, 120+12:120+28] = (-val_q0002[:, 12:28]/1200)

    #print('Metric: ------------------')
    metric = sklearn.metrics.r2_score(val_labels_valid2,preds)
    print(metric)
    #print('***')
    return metrics

K.clear_session()
if DEBUG:
    model = get_model(4, head_dim = 4)
    #model = get_model()
else:
    model = get_model()

model.load_weights(f'/kaggle/input/leap-model-weights-lr/0p0-384dim-noSpaceTime-3epochs-E30.weights.h5')

metrics_1 = valid_sub_func(model, val_1_ds, val_1_labels_2, val_1_labels, val_1_q0002)

del val_1_labels, val_1_q0002
gc.collect()

def arrange_x_y(X_col, X_col_not,y):
    x = {}
    x['X_col'] = X_col
    x['X_col_not'] = X_col_not
    x['y'] = y
    return x

def get_extra_vars_for_test_ds(x):
    x['lat']  = 0.0
    x['lon']  = 0.0
    x['day_sin']  = 0.0
    x['day_cos']  = 0.0
    x['sec_sin']  = 0.0
    x['sec_cos']  = 0.0
    return x

def get_dataset(X_train_col, X_train_col_not,y, batch_size, shuffle, split = 'valid', to_repeat = False, drop_reminder = True):
    ds = tf.data.Dataset.from_tensor_slices((X_train_col, X_train_col_not,y))
    ds = ds.map(arrange_x_y, tf.data.AUTOTUNE)
    ds = ds.map(get_extra_vars_for_test_ds, tf.data.AUTOTUNE)

    if DEBUG:
        ds = ds.take(64)

    ds = ds.cache()
    samples_num = ds.reduce(0, lambda x,_: x+1).numpy()

    if shuffle:
        ds = ds.shuffle(shuffle, reshuffle_each_iteration = True)
    if to_repeat:
        ds = ds.repeat()

    ds_label = ds.map(get_labels, tf.data.AUTOTUNE)
    ds_q002 = ds.map(get_stateq002, tf.data.AUTOTUNE)
    ds = ds.map(get_normed, tf.data.AUTOTUNE)
    ds = ds.map(get_concats, tf.data.AUTOTUNE)
    ds_label_2 = ds.map(get_labels_2, tf.data.AUTOTUNE)

    ds = ds.map(get_final_output, tf.data.AUTOTUNE)

    ds = ds.padded_batch(
                batch_size, padding_values=(PAD, PAD),
        padded_shapes=([col_len+col_not_len],[col_num_y]), drop_remainder=drop_reminder)

    ds_label = ds_label.batch(batch_size)
    ds_label_2 = ds_label_2.batch(batch_size)
    ds_q002 = ds_q002.batch(batch_size)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, ds_label, ds_label_2, ds_q002

import polars as pl

def get_test_ds():
    test_df = pl.read_csv('/kaggle/input/leap-atmospheric-physics-ai-climsim/test.csv')

    X_train_col_not = test_df.select(X_cols_series_NOT).to_numpy()
    X_train_col = test_df.select(X_cols_series).to_numpy()
    y = X_train_col[:, :368]

    test_ds, test_labels_ds, val_1_labels_ds_2, val_1_ds_q0002 = get_dataset(
        X_train_col, X_train_col_not,y, val_batch_size, shuffle = False, drop_reminder = False)
    del y, X_train_col, X_train_col_not, test_df
    gc.collect()
    return test_ds, test_labels_ds, val_1_labels_ds_2, val_1_ds_q0002

test_ds, test_labels_ds, val_1_labels_ds_2, val_1_ds_q0002 = get_test_ds()
metrics_best_epoch = metrics_1

test_q0002 = [x.numpy() for x in val_1_ds_q0002]
test_q0002 =np.concatenate(test_q0002, axis = 0)

shutil.rmtree('/kaggle/working/input')

def get_preds(model, epoch):
    preds = model.predict(test_ds, verbose = 0)
    pickle.dump(preds, open(f'raw_preds_{epoch}.p', 'bw'))
    preds = preds[:, :368]
    preds = preds.astype(np.float64)

    for i in range(len(metrics_best_epoch)):
        if metrics_best_epoch[i]<0:
            preds[:,i] = 0

    preds = preds + mean_y.reshape(1,-1)*stds
    preds[:, np.where(stds_new == 0)] = 0
    preds = preds/np.where(stds>0, stds, 1)
    
    preds[:, 120+12:120+28] = (-test_q0002[:, 12:28]/1200)
    pickle.dump(preds, open(f'preds_{epoch}.p', 'bw'))

model = get_model()
model.load_weights(f'/kaggle/input/leap-model-weights-lr/0p0-384dim-noSpaceTime-3epochs-E30.weights.h5')
get_preds(model, 90)