''' This is used because the other file "utils_EnbPI" can be too clumsy, and it is not helpful if I just run quick experiments. In particular, this util is predominantly "class" based so it is a lot more dynamic'''
import numpy as np
import pickle
import matplotlib.pyplot as plt
import enbPI.utils_EnbPI as utils_EnbPI
import enbPI.PI_class_EnbPI as PI_class_EnbPI
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from argparse import Namespace


''' (Main) EnbPI training '''


def run_result_full(data_type, mathcalA, univariate=True):
    '''
        The '_old' and '_new' have all except X the same. '_old' contains no time feature
    '''
    # A. Get Data
    data_container = data_loader()
    if data_type == 'simulation':
        Data_dc_old, Data_dc_new = data_container.get_non_stationary_simulate()
        alpha = 0.1
    else:
        data_y, data_x_old, data_x_new = data_container.get_non_stationary_real(
            univariate=univariate)
        alpha = 0.1
    # B. Run model and save result
    result_old_ls_cov, result_new_ls_cov, result_old_ls_width, result_new_ls_width = [], [], [], []
    train_fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # train_fracs = [0.5, 0.6, 0.7, 0.8]
    for train_frac in train_fracs:
        old_tmp_cov, new_tmp_cov, old_tmp_width, new_tmp_width = [], [], [], []
        # for itrial in [0, 1, 2]:
        for itrial in [0]:
            arg = Namespace(train_frac=train_frac,
                            mathcalA=mathcalA,
                            itrial=itrial,
                            alpha=alpha)
            if data_type == 'simulation':
                result_old, current_mod_old = utils_EnbPI.split_and_train(
                    Data_dc_old, arg.train_frac, arg.mathcalA, arg.alpha, itrial=arg.itrial, return_full=True)
                result_new, current_mod_new = utils_EnbPI.split_and_train(
                    Data_dc_new, arg.train_frac, arg.mathcalA, arg.alpha, itrial=arg.itrial, return_full=True)
            else:
                X_train_old, X_predict_old, Y_train, Y_predict = train_test_split(
                    arg.train_frac, data_x_old, data_y)
                arg.smallT = True  # this is because sometimes for large training size, using too many residuals can actually be bad
                result_old, current_mod_old = EnbPI_training(
                    arg, X_train_old, X_predict_old, Y_train, Y_predict)
                X_train_new, X_predict_new, _, _ = train_test_split(
                    arg.train_frac, data_x_new, data_y)
                result_new, current_mod_new = EnbPI_training(
                    arg, X_train_new, X_predict_new, Y_train, Y_predict)
            old_tmp_cov.append(result_old['coverage'].item())
            new_tmp_cov.append(result_new['coverage'].item())
            old_tmp_width.append(result_old['width'].item())
            new_tmp_width.append(result_new['width'].item())
        result_old_ls_cov.append(old_tmp_cov)
        result_new_ls_cov.append(new_tmp_cov)
        result_old_ls_width.append(old_tmp_width)
        result_new_ls_width.append(new_tmp_width)
    cov_old = get_stat(result_old_ls_cov)
    cov_new = get_stat(result_new_ls_cov)
    width_old = get_stat(result_old_ls_width)
    width_new = get_stat(result_new_ls_width)
    cov_table = get_df(cov_old, cov_new, train_fracs)
    width_table = get_df(width_old, width_new, train_fracs)
    return cov_table, width_table


def EnbPI_training(arg, X_train, X_predict, Y_train, Y_predict):
    current_mod = PI_class_EnbPI.prediction_interval(
        arg.mathcalA,  X_train, X_predict, Y_train, Y_predict)
    current_mod.fit_bootstrap_models_online(25, miss_test_idx=[])
    result = current_mod.run_experiments(
        alpha=arg.alpha, stride=1, data_name='Anything', itrial=arg.itrial, methods=['Ensemble'], smallT=arg.smallT)
    return result, current_mod


''' Data class '''


class data_loader():
    def __init__(self):
        pass

    def get_non_stationary_simulate(self):
        with open(f'Data_nochangepts_nonlinear.p', 'rb') as fp:
            Data_dc_old = pickle.load(fp)
        fXold = Data_dc_old['f(X)']
        gX = non_stationarity(len(fXold))
        fXnew = gX*fXold
        for _ in ['quick_plot']:
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(fXold, label='old f(X)')
            ax.plot(fXnew, label='new f(X)')
            ax.legend()
        Data_dc_new = {}
        for key in Data_dc_old.keys():
            if key == 'Y':
                continue
            if key == 'X':
                Data_dc_new[key] = np.c_[
                    np.arange(Data_dc_old[key].shape[0]) % 12, Data_dc_old[key]]
            elif key == 'f(X)':
                Data_dc_new[key] = fXnew
            else:
                Data_dc_new[key] = Data_dc_old[key]
        Data_dc_new['Y'] = Data_dc_new['f(X)']+Data_dc_new['Eps']
        Data_dc_old['Y'] = Data_dc_new['Y']
        Data_dc_old['f(X)'] = Data_dc_new['f(X)']
        return Data_dc_old, Data_dc_new

    def get_non_stationary_real(self, univariate=True, max_N=2000, path='Data/Solar_Atl_data.csv'):
        # Stationary real
        data = utils_EnbPI.read_data(3, path, 10000)
        data_y = data['DHI'].to_numpy()  # Convert to numpy
        if univariate:
            # Univariate feature
            data_x_old = rolling(data_y, window=10)
        else:
            # Multivariate feature
            data_x_old = data.loc[:, data.columns
                                  != 'DHI'].to_numpy()  # Convert to numpy
        # Add one-hot-encoded DAY features using // (or hour features using %)
        hours = int(data_y.shape[0]/365)
        N = data_x_old.shape[0]
        day_feature = False
        if day_feature:
            # Day one-hot 0,...,364
            one_hot_feature = (np.arange(N) // hours).reshape(-1, 1)
        else:
            # Hourly one-hot 0,...,23
            one_hot_feature = (np.arange(N) % hours).reshape(-1, 1)
        one_hot_feature = OneHotEncoder().fit_transform(one_hot_feature).toarray()
        data_x_new = np.c_[one_hot_feature, data_x_old]
        if univariate:
            # This return is for univariate feature
            return data_y[-(N-1):], data_x_old[:N-1], data_x_new[:N-1]
        else:
            # This return is for multivariate feature
            return data_y[-max_N:], data_x_old[-max_N:], data_x_new[-max_N:]


''' Other helpers '''


def plot_int_on_data(current_mod, frac=0.1):
    '''
        current_mod is an object of 'PI_class_EnbPI.prediction_interval()'
        frac: what fraction of total points to plot
    '''
    self = current_mod
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ends = self.Ensemble_pred_interval_ends
    truth = self.Y_predict
    est = self.Ensemble_pred_interval_centers
    end = int(len(truth)*frac)
    T = range(end)
    ax[0].scatter(T, truth[:end], color='blue', s=1.5)
    ax[0].scatter(T, est[:end], color='orange', s=1.5)
    low, up = ends['lower'].to_numpy(), ends['upper'].to_numpy()
    ax[0].fill_between(T, low[:end], up[:end], alpha=0.25)
    status = ((truth[:end] >= low[:end]) & (truth[:end] <= up[:end]))
    cov = status.mean()
    ax[0].set_title(f'Coverage over {end} pts is {cov:.3f}')
    T_nocov = (~status).sum()
    ax[1].scatter(range(T_nocov), truth[:end][~status], color='blue', s=1.5)
    ax[1].scatter(range(T_nocov), est[:end][~status], color='orange', s=1.5)
    ax[1].fill_between(range(T_nocov), low[:end][~status],
                       up[:end][~status], alpha=0.25)
    ax[1].set_title(f'Result when points are not covered')
    fig.tight_layout()


def get_stat(result_ls):
    result_ls = np.array(result_ls)
    res_mean = np.round(result_ls.mean(axis=1), 3)
    res_std = np.round(result_ls.std(axis=1)/np.sqrt(result_ls.shape[1]), 3)
    res = [f'{a} ({b})' for a, b in zip(res_mean, res_std)]
    return res


def get_df(res_old, res_new, cols):
    res_table = pd.DataFrame([res_old, res_new])
    res_table.index = ['EnbPI old', 'EnbPI new']
    res_table.column = cols
    return res_table


def non_stationarity(N):
    '''
        Compute g(t)=t'*sin(2*pi*t'/12), which is multiplicative on top of f(X), where
        t' = t mod 12 (for seaonality)
    '''
    cycle = 12
    trange = np.arange(N)
    tprime = trange % cycle
    term2 = np.sin(2*np.pi*tprime/cycle)
    return tprime*term2


def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def train_test_split(train_frac, data_x, data_y):
    train_size = int(len(data_y)*train_frac)
    X_train = data_x[:train_size, :]
    X_predict = data_x[train_size:, :]
    Y_train = data_y[:train_size]
    Y_predict = data_y[train_size:]
    return X_train, X_predict, Y_train, Y_predict
