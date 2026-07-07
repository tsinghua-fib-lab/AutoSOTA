import math
import pandas as pd
from sklearn.preprocessing import scale
from scipy.interpolate import griddata
from scipy.io import loadmat
import numpy as np
import random


def read_prepared_data(args):
    data = []

    for l in range(len(args.ConType)):
        label = pd.read_csv(args.data_document_path + "/csv/" + args.name + args.ConType[l] + ".csv")
        target = []
        for k in range(args.trail_number):
            filename = args.data_document_path + "/" + args.ConType[l] + "/" + args.name + "Tra" + str(k + 1) + ".csv"
            data_pf = pd.read_csv(filename, header=None)
            eeg_data = data_pf.iloc[:, 2:]  # KUL,DTU

            data.append(eeg_data)
            target.append(label.iloc[k, args.label_col])

    return data, target


def get_data_from_mat(mat_path):
    mat_eeg_data = []
    mat_event_data = []
    matstruct_contents = loadmat(mat_path)
    matstruct_contents = matstruct_contents['data']
    mat_event = matstruct_contents[0, 0]['event']['eeg'].item()
    mat_event_value = mat_event[0]['value']  # 1*60 1=male, 2=female
    mat_eeg = matstruct_contents[0, 0]['eeg']  # 60 trials 3200*66
    for i in range(mat_eeg.shape[1]):
        mat_eeg_data.append(mat_eeg[0, i])
        mat_event_data.append(mat_event_value[i][0][0])

    return mat_eeg_data, mat_event_data


def sliding_window(eeg_datas, labels, args, out_channels):
    window_size = args.window_length
    stride = int(window_size * (1 - args.overlap))

    all_windows = []
    all_labels = []

    # labels 是一个列表，例如 [1, 2, 1, 2, ...], eeg_datas 的形状是 (num_trials, samples, channels)
    for trial_index in range(eeg_datas.shape[0]):
        trial_eeg = eeg_datas[trial_index]  # 形状: (samples, channels)
        trial_label = labels[trial_index]   # 单个值, 例如 1 或 2

        for i in range(0, trial_eeg.shape[0] - window_size + 1, stride):
            window = trial_eeg[i:i + window_size, :]
            all_windows.append(window)
            all_labels.append(trial_label)

    if not all_windows:
        return np.array([]), np.array([])

    eeg_set = np.array(all_windows)
    label_set = np.array(all_labels).reshape(-1, 1) 

    return eeg_set, label_set


def new_sliding_window(eeg_datas, labels, args, out_channels):
    window_size = args.window_length
    stride = int(128 * (1 - args.overlap))

    train_eeg = []
    test_eeg = []
    train_label = []
    test_label = []

    for m in range(len(labels)):
        eeg = eeg_datas[m]
        label = labels[m]
        windows = []
        new_label = []
        for i in range(0, eeg.shape[0] - window_size + 1, stride):
            window = eeg[i:i + window_size, :]
            windows.append(window)
            new_label.append(label)
        train_eeg.append(np.array(windows)[:int(len(windows) * 0.9)])
        test_eeg.append(np.array(windows)[int(len(windows) * 0.9):])
        train_label.append(np.array(new_label)[:int(len(windows) * 0.9)])
        test_label.append(np.array(new_label)[int(len(windows) * 0.9):])

    train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, out_channels)
    test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, out_channels)
    train_label = np.stack(train_label, axis=0).reshape(-1, 1)
    test_label = np.stack(test_label, axis=0).reshape(-1, 1)

    return train_eeg, test_eeg, train_label, test_label


def sliding_window_csp(eeg_datas, labels, args, out_channels):
    window_size = args.window_length
    stride = int(window_size * (1 - args.overlap))

    eeg_set = []
    label_set = []

    for m in range(len(labels)):  # labels 0-19
        eeg = eeg_datas[m]
        label = labels[m]
        windows = []
        new_label = []
        for i in range(0, eeg.shape[0] - window_size + 1, stride):
            window = eeg[i:i + window_size, :]
            windows.append(window)
            new_label.append(label)

        eeg_set.append(np.array(windows))
        label_set.append(np.array(new_label))

    eeg_set = np.concatenate(eeg_set, axis=0)
    label_set = np.concatenate(label_set, axis=0).reshape(-1, 1)

    return eeg_set, label_set


def within_data(eeg_datas, labels):
    train_datas = []
    test_datas = []
    train_labels = []
    test_labels = []

    for m in range(len(labels)):  # labels 0-19
        eeg = eeg_datas[m]
        label = labels[m]

        train_datas.append(np.array(eeg)[:, :int(eeg.shape[1] * 0.9)])
        test_datas.append(np.array(eeg)[:, int(eeg.shape[1] * 0.9):])
        train_labels.append(np.array(label))
        test_labels.append(np.array(label))

    train_datas = np.stack(train_datas, axis=0)
    test_datas = np.stack(test_datas, axis=0)
    train_labels = np.stack(train_labels, axis=0)
    test_labels = np.stack(test_labels, axis=0)

    return train_datas, test_datas, train_labels, test_labels
