import importlib as ipb
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import keras
from keras.layers import Dropout, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, clone_model
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import RidgeCV
import utils_EnbPI_journal as util
import scipy
import itertools
from pathlib import Path
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
# For imputation followed by detrending, at SENSOR level
# Training data is pre-processed before using this

'''1.'''
# Impute the training data with sklearn impute, restrict flow to be non-negative &
# use extratree regressor: https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html
# Then define {t: (X_tk,Y_tk)_k=1^K} for all t \in {m,...,T+T1}

# NOTE: run this ONCE is enough, as it takes about several minutes to do


def imput_and_get_X_Y_time_dic(frac, first_time=False):
    with open(f'Data/Anomaly_Detection_Data/flow_frame_train_{str(frac)}.p', 'rb') as fp:
        train = pickle.load(fp)
    est = ExtraTreesRegressor(n_estimators=10, random_state=0)
    imputed_train = IterativeImputer(imputation_order='random', min_value=0).fit_transform(train)
    imputed_train = pd.DataFrame(imputed_train, index=train.index, columns=train.columns)
    # NOTE: read in test data as well to define X_tk. In reality, we just sequentially define X_tk
    with open(f'Data/Anomaly_Detection_Data/flow_frame_test_{str(frac)}.p', 'rb') as fp:
        test = pickle.load(fp)
    T, T1 = imputed_train.shape[0], test.shape[0]
    K = imputed_train.shape[1]
    full = pd.concat((imputed_train, test))
    sensors = list(imputed_train.columns)
    with open('Data/Anomaly_Detection_Data/sensor_neighbors.p', 'rb') as fp:
        sensor_neighbors = pickle.load(fp)
    m = 8  # memory depth, same as size of neighbors being considered
    if first_time:
        # This is true if we run it for the FIRST time (e.g. X_Y_time_dic has not been saved)
        X_Y_time_dic = {}  # NOTE that the first index is NOT time 0, but time m, because first m are throwed away
        # Take 2 min 21 sec
        for t in range(T+T1-m):
            X_all_loc = []
            Y_all_loc = []
            for i in range(K):
                k = sensors[i]
                N_k_hat = [np.where(sensors == s)[0][0]
                           for s in sensor_neighbors[k][:m]]
                Y_all_loc.append(full.iloc[t+m, i])
                X_all_loc.append(full.iloc[t:t+m, N_k_hat].to_numpy().flatten())
            X_Y_time_dic[t] = {'X_t': np.stack(X_all_loc), 'Y_t': np.array(Y_all_loc)}
        X_Y_time_dic[0]['X_t'].shape
        X_Y_time_dic[0]['Y_t'].shape
        with open(f'Data/Anomaly_Detection_Data/X_Y_time_dic_{str(frac)}.p', 'wb') as fp:
            pickle.dump(X_Y_time_dic, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'Data/Anomaly_Detection_Data/X_Y_time_dic_{str(frac)}.p', 'rb') as fp:
            X_Y_time_dic = pickle.load(fp)
    return [X_Y_time_dic, m, T-m, T1, K, full.index, full.columns]


'''2.'''
# Start revised ECAD, by outputing residuals from RNN.
# Deploy this on Google Colab for speed.
# Helper #1


def generate_bootstrap_samples(n, m, B):
    '''
      Return: B-by-m matrix, where row b gives the indices for b-th bootstrap sample
    '''
    samples_idx = []
    for b in range(B):
        sample_idx = np.unique(np.random.choice(n, m))
        samples_idx.append(sample_idx)
    return(samples_idx)

# Helper #2


def dict_to_data(X_Y_time_dic, time_range):
    X = []
    Y = []
    for t in time_range:
        X.append(X_Y_time_dic[t]['X_t'])
        Y.append(X_Y_time_dic[t]['Y_t'])
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return([X, Y])

# Main Functions


def ECAD(X_Y_time_dic, model, args):
    '''
        Changes to original EnbPI:
        All matrices dimension must multiply by K, and I implicitly assume the columns in
        "boot_predictions" are
        (T-m)1,...,(T-m)K,...,T1....,TK,...,(T+1)1...., then as I loop over columns, just need mod and then
        check if index in bootstrap time-index
    '''
    m, T_minus_m, T1, K, B, idx, sensors = args
    np.random.seed(1)
    boot_samples_idx = generate_bootstrap_samples(T_minus_m, T_minus_m, B)
    # hold predictions from each f^b
    boot_predictions = np.zeros((B, (m+T1)*K), dtype=float)
    # for (t,k)^th column, it shows which f^b uses t in training (so exclude in aggregation)
    in_boot_sample = np.zeros((B, m*K), dtype=bool)
    out_sample_predict = np.zeros((m*K, T1*K))
    # Get train & test data
    X_train, Y_train = dict_to_data(X_Y_time_dic, range(T_minus_m))
    X_pred, Y_pred = dict_to_data(X_Y_time_dic, range(T_minus_m, T_minus_m+T1))
    if model.__class__.__name__ == 'Sequential' and model.name == 'RNN':
        n, n1, p = X_train.shape[0], X_pred.shape[0], X_train.shape[1]
        X_train = X_train.reshape((n, 1, p))
        X_pred = X_pred.reshape((n1, 1, p))
    models = {}
    for b in range(B):
        print(f'Training {b+1}/{B}th Bootstrap Model')
        start = time.time()
        # In expectation, X_train_b size is (T-m)*(1-e^-1)*m*|\hat{N}^k|-by-K
        X_train_b, Y_train_b = dict_to_data(X_Y_time_dic, boot_samples_idx[b])
        if model.__class__.__name__ == 'Sequential':
            start1 = time.time()
            model_cloned = clone_model(model)
            opt = Adam(0.0005)
            model_cloned.compile(loss='mean_squared_error', optimizer=opt)
            callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
            bsize = int(0.1*len(np.unique(boot_samples_idx[b])))
            if model.name == 'NeuralNet':
                model_cloned.fit(X_train_b, Y_train_b, epochs=25,
                                 batch_size=bsize, callbacks=[callback], verbose=0)
            else:
                # This is RNN, mainly have different shape and decrease epochs
                n, p = X_train_b.shape[0], X_train_b.shape[1]
                X_train_b = X_train_b.reshape((n, 1, p))
                model_cloned.fit(X_train_b, Y_train_b,
                                 epochs=10, batch_size=bsize, callbacks=[callback], verbose=0)
            models[b] = model_cloned
        else:
            model = model.fit(X_train_b, Y_train_b)
            models[b] = model
        boot_predictions[b] = models[b].predict(np.r_[X_train[-m*K:], X_pred]).flatten()
        for i in range(m):
            # Populate all multiples of m by whether the time index is in the b^th bootstrap index
            in_boot_sample[b, np.arange(i, m*K, step=m)] = T_minus_m-m+i in boot_samples_idx[b]
        print(f'Took {time.time()-start} secs')
    # Get LOO training resid
    resid_in_sample = []
    for i in range(m*K):
        b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1)
        if(len(b_keep) > 0):
            resid_LOO = Y_train[-m*K:][i] - boot_predictions[b_keep, i].mean()
            resid_in_sample.append(resid_LOO)
            out_sample_predict[i] = boot_predictions[b_keep, m*K:].mean(0)
        else:  # if aggregating an empty set of models, predict zero everywhere
            resid_in_sample.append(Y_train[i])
            out_sample_predict[i] = np.zeros(T1*K)
    resid_in_sample = np.array(resid_in_sample).reshape((m, K))
    # Get Pred residual (NOTE: NOT LOO or rank the prediction from previous m*|hat{N}_k| LOO models, as doing so is too tedious without too much difference)
    resid_out_sample = Y_pred.reshape((len(Y_pred),))-out_sample_predict.mean(axis=0)
    resid_out_sample = resid_out_sample.reshape((T1, K))
    full_resid = pd.DataFrame(np.r_[resid_in_sample, resid_out_sample],
                              index=idx[-(m+T1):], columns=sensors)
    return full_resid


def detect_anomalies(full_resid, args, return_pval=False):
    # Basedon the "actual_anomalies" function (with small variants) to get predicted anomalies
    # BUT here I start at the m^th row, because first m are T-m,...,T-1.
    # alpha% for defining true anomalies. As I typically compare with longer past and more neigbors, make this larger
    m, T1, K, alpha = args
    pred_anomalies = np.zeros((T1, K))
    pvals = np.zeros((T1, K))
    with open('Data/Anomaly_Detection_Data/sensor_neighbors.p', 'rb') as fp:
        sensor_neighbors = pickle.load(fp)
    sensors = list(full_resid.columns)
    # 1.5min
    for i in range(K):
        k = sensors[i]
        N_k = [np.where(sensors == s)[0][0]
               for s in sensor_neighbors[k][:m]]
        print(f'Sensor {i}')
        for t in range(m, m+T1):  # Skip first m row
            prev_flow = full_resid.iloc[t-m:t, N_k].to_numpy().flatten()
            # NOTE: here is where binning is used!
            beta_hat_bin = util.binning(prev_flow, alpha)
            if full_resid.iloc[t, i] >= np.percentile(prev_flow, 100*(1-alpha+beta_hat_bin)) or full_resid.iloc[t, i] <= np.percentile(prev_flow, 100*beta_hat_bin):
                pred_anomalies[t-m, i] = 1
            if return_pval:
                pvals[t-m, i] = np.sum(full_resid.iloc[t, i] <= prev_flow)/(m*len(N_k))
    pred_anomalies = pd.DataFrame(pred_anomalies, index=full_resid.index[-T1:], columns=sensors)
    pvals = pd.DataFrame(pvals, index=full_resid.index[-T1:], columns=sensors)
    if return_pval:
        return [pred_anomalies, pvals]
    else:
        return pred_anomalies


def get_stat(pred_anomalies, T1):
    # Get F1 score etc:
    with open('Data/Anomaly_Detection_Data/true_anomalies.p', 'rb') as fp:
        true_anomalies = pickle.load(fp)
    sensors = list(true_anomalies.columns)
    F1_score = np.zeros(len(sensors))
    precision = np.zeros(len(sensors))
    recall = np.zeros(len(sensors))
    for i in range(len(sensors)):
        k = sensors[i]
        F1_score[i] = f1_score(true_anomalies[k][-T1:], pred_anomalies[k])
        precision[i] = precision_score(true_anomalies[k][-T1:], pred_anomalies[k])
        recall[i] = recall_score(true_anomalies[k][-T1:], pred_anomalies[k])
    summary = pd.DataFrame(np.c_[np.mean(true_anomalies, axis=0), np.mean(pred_anomalies, axis=0), F1_score, precision, recall],
                           columns=['True % Anomalies', 'Predicted % Anomalies', 'F1 score', 'Precision', 'Recall'], index=sensors)
    return summary.sort_values(by='F1 score', ascending=False)


'''3.'''
# Compare against competing methods, which return the same results (so briefly transform
# "X_Y_time_dic" into "X_Y_loc_dic={k:(X_k,Y_k)}" for these methods to work on each location individually)


def time_dic_to_loc_dic(frac, supervised=False):
    with open(f'Data/Anomaly_Detection_Data/X_Y_time_dic_{str(frac)}.p', 'rb') as fp:
        X_Y_time_dic = pickle.load(fp)
    keys = X_Y_time_dic.keys()
    K, p = X_Y_time_dic[0]['X_t'].shape[0], X_Y_time_dic[0]['X_t'].shape[1]
    X_Y_loc_dic = {}  # k: X_k or (X_k,Y_k) over all t. Y_k here is binary
    if supervised:
        with open('Data/Anomaly_Detection_Data/true_anomalies.p', 'rb') as fp:
            true_anomalies = pickle.load(fp)
    for k in range(K):
        X_loc = np.zeros((len(keys), p))
        for t in keys:
            X_loc[t] = X_Y_time_dic[t]['X_t'][k]
        if supervised:
            Y_loc = np.array(true_anomalies.iloc[:len(keys), k])
            X_Y_loc_dic[k] = {'X_k': X_loc, 'Y_k': Y_loc}
        else:
            X_Y_loc_dic[k] = {'X_k': X_loc}
    return X_Y_loc_dic


def mod_to_result(regr_name, X_Y_loc_dic, train_size, frac, supervised=False):
    K = len(X_Y_loc_dic)
    predictions = np.zeros((len(X_Y_loc_dic[0]['X_k'])-train_size, K))
    for k in range(K):
        print(f'Node {k}')
        X_Y_loc = X_Y_loc_dic[k]
        X_loc = X_Y_loc['X_k']
        X_train = X_loc[:train_size]
        X_predict = X_loc[train_size:]
        mod = eval(regr_name)
        if supervised:
            Y_loc = X_Y_loc['Y_k']
            Y_train = Y_loc[:train_size]
            mod.fit(X_train, Y_train)
        else:
            mod.fit(X_train)
        predictions[:, k] = mod.predict(X_predict)
    with open(f'Data/Anomaly_Detection_Data/flow_frame_test_{str(frac)}.p', 'rb') as fp:
        test = pickle.load(fp)
    predictions = pd.DataFrame(predictions, index=test.index, columns=test.columns)
    return predictions


'''4.'''


def plt_prec_recall_F1(df):
    """Plot precision, recall, and F1 score in a row, for all methods
       x-axis is the train_frac
    """
    fig, ax = plt.subplots(1, 3, figsize=(20, 4), sharex=True, sharey=True)
    methods = np.unique(df['method'])
    train_frac = np.unique(df['train_frac'])
    num_mtds = len(methods)
    axisfont = 22
    tickfont = axisfont - 2
    for j in range(num_mtds):
        mtd = methods[j]
        df_mtd = df.loc[df.method == mtd]
        precision, recall, F1 = df_mtd['precision'], df_mtd['recall'], df_mtd['F1']
        idx_to_name = {0: precision, 1: recall, 2: F1}
        idx_to_name_1 = {0: 'Precision', 1: 'Recall', 2: r'$F_1$'}
        for i in range(3):
            current_fig = ax[i]
            current_fig.plot(train_frac, idx_to_name[i], label=mtd, linestyle='-',
                             marker='o')
            # Title, labels, ticks, etc.
            current_fig.tick_params(axis='both', which='major', labelsize=tickfont)
            current_fig.set_xticks(train_frac)
            if i == 0:
                current_fig.legend(loc='upper left', bbox_to_anchor=(0, 1.35),
                                   ncol=int(num_mtds/2), fontsize=axisfont-2)
            current_fig.set_ylabel(idx_to_name_1[i], fontsize=axisfont)
            current_fig.set_xlabel('Train Fraction', fontsize=axisfont)
    fig.tight_layout(pad=0)
    fig.savefig('Traffic_ECAD_plts.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
