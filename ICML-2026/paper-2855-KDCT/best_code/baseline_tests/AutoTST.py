import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.append(os.path.abspath('..'))
import time
from dataloader import load_data
import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset


def snr_score(estimator, x_test, y_test, permutations=None, discrete=False):
    # pred = estimator.predict(x_test).reshape(-1, )
    if discrete:
        pred = estimator.predict(x_test)
    else:
        pred = np.array(estimator.predict_proba(x_test))
        if pred.ndim == 2:
            pred = pred[:, 1]

    pred = np.array(pred)
    y_test = np.array(y_test).reshape(-1, )

    p_samp = pred[y_test ==1]
    q_samp = pred[y_test == 0]
    # print(len(p_samp), len(q_samp))
    c = len(p_samp) / (len(p_samp) + len(q_samp))
    # c = 1-c
    signal = (np.mean(p_samp) - np.mean(q_samp))
    if permutations is None:
        if c == 1 or c == 0:
            return -500
        noise = np.sqrt(1 / c * np.var(p_samp) + 1 / (1 - c) * np.var(q_samp))
        if noise == 0:
            return - 500
        snr = signal / noise
        # check for nan
        if snr != snr:
            return -500
        else:
            return snr
    else:
        p = 0
        for i in range(permutations):
            np.random.shuffle(pred)
            p_samp = pred[y_test == 1]
            q_samp = pred[y_test == 0]
            signal_perm = np.mean(p_samp) - np.mean(q_samp)

            if signal <= float(signal_perm):
                p += float(1 / permutations)
        # print(signal, p)
        return p  # this is the corresponding SNR

def TST_AUTO(name, N1, rs, check, n_test, alpha=0.05):
    np.random.seed(rs)
    X_train, Y_train = load_data(name, N1, rs, check)
    S_train = np.concatenate((X_train, Y_train), axis=0)
    label_train = np.concatenate(([1] * N1, [0] * N1))
    df_train = pd.DataFrame({"data"+str(i): S_train[:, i] for i in range(len(S_train[0]))})
    df_train["label"] = label_train
    train_data = TabularDataset(df_train)
    start_time = time.time()
    AutoML_predictor = TabularPredictor(label="label", problem_type="binary", eval_metric="accuracy", verbosity=0).fit(train_data, presets='best_quality', time_limit=60)
    train_time = time.time() - start_time

    H_AUTO = np.zeros(n_test)
    N_test_all = 10 * N1
    X_test_all, Y_test_all = load_data(name, N_test_all, rs + 283, check)
    test_time = 0
    # test by C2ST-L
    for k in range(n_test):
        ind_test = np.random.choice(N_test_all, N1, replace=False)
        X_test = X_test_all[ind_test]
        Y_test = Y_test_all[ind_test]
        S_test = np.concatenate((X_test, Y_test), axis=0)
        label_test = np.concatenate(([1] * len(ind_test), [0] * len(ind_test)))
        df_test = pd.DataFrame({"data" + str(i): S_test[:, i] for i in range(len(S_test[0]))})
        df_test["label"] = label_test
        test_data = TabularDataset(df_test)
        start_time = time.time()
        p_AutoML = snr_score(AutoML_predictor, test_data, label_test, permutations = 100, discrete=False)
        test_time += time.time() - start_time
        h_AutoML = int(p_AutoML < alpha)
        H_AUTO[k] = h_AutoML
    
    return H_AUTO, train_time, test_time

