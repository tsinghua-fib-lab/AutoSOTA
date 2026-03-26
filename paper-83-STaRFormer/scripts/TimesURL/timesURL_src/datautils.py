import os
import time

import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from timesURL_src.utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from timesURL_src.utils import generate_mask
from timesURL_src.utils import interpolate_cubic_spline
from timesURL_src.utils import normalize_with_mask


def load_UCR(dataset, load_tp: bool = True):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # extend dim to NTC
    train, test = train[..., np.newaxis], test[..., np.newaxis]
    p = 1
    mask_tr, mask_te = generate_mask(train, p), generate_mask(test, p)

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ] or p != 1:
        scaler = StandardScaler()
        train, test = normalize_with_mask(train, mask_tr, test, mask_te, scaler)
        # mean = np.nanmean(train)
        # std = np.nanstd(train)
        # train = (train - mean) / std
        # test = (test - mean) / std

    if load_tp:
        tp = np.linspace(0, 1, train.shape[1], endpoint=True).reshape(1, -1, 1)
        train = np.concatenate((train, np.repeat(tp, train.shape[0], axis=0)), axis=-1)
        test = np.concatenate((test, np.repeat(tp, test.shape[0], axis=0)), axis=-1)

    return {'x': train, 'mask': mask_tr}, train_labels, {'x': test, 'mask': mask_te}, test_labels
    # return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_others(dataset, load_tp: bool = True):
    data = np.load(f'datasets/Others/{dataset}.npy', allow_pickle=True).item()
    train_X, train_mask, train_y, test_X, test_mask, test_y = \
        data["tr_x"], data["tr_mask"], data["tr_y"], data["te_x"], data["te_mask"], data["te_y"]

    scaler = MinMaxScaler()

    train_X, test_X = normalize_with_mask(train_X, train_mask, test_X, test_mask, scaler)

    train_tp, test_tp = data['tr_t'], data['te_t']
    if load_tp:
        train_X = np.concatenate((train_X, train_tp.reshape(train_tp.shape[0], -1, 1)), axis=-1)
        test_X = np.concatenate((test_X, test_tp.reshape(test_tp.shape[0], -1, 1)), axis=-1)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return {'x': train_X, 'mask': train_mask}, train_y, {'x': test_X, 'mask': test_mask}, test_y


def load_UEA(dataset, load_tp: bool = False):
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([d.tolist() for d in t_data])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

    try:
        train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
        test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]

        train_X, train_y = extract_data(train_data)
        test_X, test_y = extract_data(test_data)
    except:
        data = np.load(f'datasets/UEA/{dataset}/{dataset}.npy', allow_pickle=True).item()
        train_X, train_y, test_X, test_y = data["train_X"], data["train_y"], data["test_X"], data["test_y"]

    p = 1
    mask_tr, mask_te = generate_mask(train_X, p), generate_mask(test_X, p)
    # scaler = MinMaxScaler()
    scaler = StandardScaler()

    train_X, test_X = normalize_with_mask(train_X, mask_tr, test_X, mask_te, scaler)

    if load_tp:
        tp = np.linspace(0, 1, train_X.shape[1], endpoint=True).reshape(1, -1, 1)
        train_X = np.concatenate((train_X, np.repeat(tp, train_X.shape[0], axis=0)), axis=-1)
        test_X = np.concatenate((test_X, np.repeat(tp, test_X.shape[0], axis=0)), axis=-1)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return {'x': train_X, 'mask': mask_tr}, train_y, {'x': test_X, 'mask': mask_te}, test_y


def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')
    if univar:
        data = data[: -1:]

    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_forecast_csv(name, offset=0 , univar=False, load_tp: bool = True):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    dt_tp = data.index
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1] if offset == 0 else 0

    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        elif name == 'WTH':
            data = data[['WetBulbCelsius']]
        else:
            data = data.iloc[:, -1:]

    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12 * 30 * 24)
        valid_slice = slice(12 * 30 * 24 - offset, 16 * 30 * 24)
        test_slice = slice(16 * 30 * 24 - offset, 20 * 30 * 24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12 * 30 * 24 * 4)
        valid_slice = slice(12 * 30 * 24 * 4 - offset, 16 * 30 * 24 * 4)
        test_slice = slice(16 * 30 * 24 * 4 - offset, 20 * 30 * 24 * 4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)

    def fixed_mask_timestamp(num, mask):
        mask_time = np.ones((mask.shape[0], mask.shape[1]))
        mask_time[np.where(mask.mean(axis=-1) == 0.)] = 0
        return np.concatenate((np.repeat(mask_time[..., np.newaxis], num, axis=-1), mask), axis=-1)

    # to N x T x C
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)

    p = 1
    mask_tr, mask_va, mask_te = generate_mask(data[:, train_slice], p), \
                                generate_mask(data[:, valid_slice], p), \
                                generate_mask(data[:, test_slice], p)
    scaler = StandardScaler()

    train_x, valid_x = normalize_with_mask(data[:, train_slice], mask_tr, data[:, valid_slice], mask_va, scaler)
    _, test_x = normalize_with_mask(data[:, train_slice], mask_tr, data[:, test_slice], mask_te, scaler)
    data = np.concatenate((train_x, valid_x, test_x), axis=1)
    mask = np.concatenate([mask_tr, mask_va, mask_te], axis=1)

    if n_covariate_cols > 0:
        dt_mask, dv_mask, d_mask = fixed_mask_timestamp(n_covariate_cols, mask_tr[:1]), \
                                   fixed_mask_timestamp(n_covariate_cols, mask_va[:1]), \
                                   fixed_mask_timestamp(n_covariate_cols, mask_te[:1])

        dt, dv, d = dt_embed[train_slice], dt_embed[valid_slice], dt_embed[test_slice]
        dt[dt_mask[0][:, :n_covariate_cols] == 0], dv[dv_mask[0][:, :n_covariate_cols] == 0], d[d_mask[0][:, :n_covariate_cols] == 0] = np.nan, np.nan, np.nan
        dt_embed = np.concatenate((dt, dv, d), axis=0)

        dt_scaler = scaler.fit(dt)
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        dt_embed[np.isnan(dt_embed)] = 0
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
        mask_tr, mask_va, mask_te = dt_mask, dv_mask, d_mask
        mask = np.concatenate([mask_tr, mask_va, mask_te], axis=1)

    if load_tp:
        dt_tp = [dt_tp[train_slice], dt_tp[valid_slice], dt_tp[test_slice]]
        tp = np.concatenate([[time.mktime(t.timetuple()) for t in tp] for tp in dt_tp])
        scaler_hat = MinMaxScaler().fit(tp.reshape(-1, 1))
        data = np.concatenate([data, np.expand_dims(scaler_hat.transform(tp.reshape(-1, 1)), 0)], axis=-1)

    if name in ('ETTh1', 'ETTh2', 'electricity', 'WTH'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]

    return {'x': data, 'mask': mask}, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(name, load_tp=False):
    res = pkl_load(f'datasets/{name}.pkl')

    p, mask_tr, mask_te = 1, [], []
    maxl = np.max([len(res['all_train_data'][k]) for k in res['all_train_data']])
    maxle = np.max([len(res['all_test_data'][k]) for k in res['all_test_data']])
    for k in res['all_train_data']:
        # generate mask
        mask_tr.append(generate_mask(res['all_train_data'][k].reshape(1, -1, 1), p, remain=1))
        mask_te.append(generate_mask(res['all_test_data'][k].reshape(1, -1, 1), p, remain=1))
        # mask
        res['all_train_data'][k] = (mask_tr[-1] * res['all_train_data'][k].reshape(1, -1, 1)).reshape(-1)
        res['all_test_data'][k] = (mask_te[-1] * res['all_test_data'][k].reshape(1, -1, 1)).reshape(-1)
        # padding mask
        mask_tr[-1] = np.concatenate((mask_tr[-1], np.full((1, maxl - mask_tr[-1].shape[1], 1), np.nan)), axis=1)
        mask_te[-1] = np.concatenate((mask_te[-1], np.full((1, maxle - mask_te[-1].shape[1], 1), np.nan)), axis=1)
    mask_tr, mask_te = np.concatenate(mask_tr, axis=0), np.concatenate(mask_te, axis=0)

    # if load_tp:
    #     tp_max, tp_min = np.max(res['all_train_timestamps']), np.min(res['all_train_timestamps'])
    #     interval = tp_max - tp_min
    #     interval = 1. if interval == 0. else interval
    #     tp_train = (res['all_train_timestamps'] - tp_min) / interval
    #     tp_test = (res['all_test_timestamps'] - tp_min) / interval
    #     res['all_train_data'] = np.concatenate((res['all_train_data'], np.repeat(tp_train, res['all_train_data'].shape[0], axis=0)), axis=-1)
    #     res['all_test_data'] = np.concatenate((res['all_test_data'], np.repeat(tp_test, res['all_test_data'].shape[0], axis=0)), axis=-1)

    return {'x': res['all_train_data'], 'mask': mask_tr}, res['all_train_labels'], res['all_train_timestamps'], \
           {'x': res['all_test_data'], 'mask': mask_te}, res['all_test_labels'], res['all_test_timestamps'], \
           res['delay']


def gen_ano_train_data(all_train_data, maxl = None, normal = False):
    maxl = np.max([len(all_train_data[k]) for k in all_train_data]) if maxl is None else maxl
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(np.array(all_train_data[k]).astype(np.float64), maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    if normal:
        data_min, data_max = np.nanmin(pretrain_data), np.nanmax(pretrain_data)
        pretrain_data = (pretrain_data - data_min) / (data_max - data_min)
    return pretrain_data