from scipy.stats import skewnorm
from scipy.linalg import norm
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn import preprocessing
import seaborn as sns
import pickle
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
import math
from numpy.random import choice
from scipy.stats import norm
from scipy.sparse import random
import warnings
import PI_class_EnbPI_journal as EnbPI  # For me
import os
import matplotlib.cm as cm
import time
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import calendar
import matplotlib.transforms as transforms
import importlib
import sys
importlib.reload(sys.modules['PI_class_EnbPI_journal'])  # For me
'''Simulation Section '''
'''Define True Models and Errors '''


def F_inv(alpha):
    '''
    Description:
        Used to compute oracle width when errors are not strongly mixing. It is a skewed normal.
    '''
    rv = skewnorm(a=5, loc=0, scale=1)  # a is skewness parameter
    return rv.ppf(alpha)


def F_inv_stronglymixing(alpha):
    '''
    Description:
        Used to compute oracle width when errors are strongly mixing. Hidden xi_t follow normal distribution
    '''
    rho = 0.6
    mean = 0 / (1 - rho)
    std = np.sqrt(0.1 / (1 - rho**2))
    return norm.ppf(alpha, loc=mean, scale=std)


def F_inv_stronglymixingDGP(alpha):
    return norm.ppf(alpha, loc=0, scale=np.sqrt(0.1))


def beta_star_comp(alpha, stronglymixing):
    # NOTE, just do this numerically, since F_inv typically do not have closed form so taking gradient to minimize the difference is not needed
    bins = 1000
    if stronglymixing:
        Finv = F_inv_stronglymixing
    else:
        Finv = F_inv
    beta_is = np.linspace(start=0, stop=alpha, num=bins)
    width = np.zeros(bins)
    for i in range(bins):
        width[i] = Finv(1 - alpha + beta_is[i]) - Finv(beta_is[i])
    i_star = np.argmin(width)
    return beta_is[i_star]


def True_mod_linear_pre(feature):
    '''
    Input:
    Output:
    Description:
        f(feature): R^d -> R
    '''
    # Attempt 0: Fit Linear model on this data
    d = len(feature)
    np.random.seed(0)
    beta0 = np.random.uniform(size=d)  # fully non-missing
    return beta0.dot(feature)


def True_mod_linear_post(feature):
    '''
    Input:
    Output:
    Description:
        f(feature): R^d -> R
    '''
    # Attempt 0: Fit Linear model on this data
    d = len(feature)
    np.random.seed(0)
    beta0 = np.random.uniform(high=5, size=d)  # fully non-missing
    return beta0.dot(feature)


def True_mod_lasso_pre(feature):
    '''
    Input:
    Output:
    Description:
        f(feature): R^d -> R
    '''
    # Attempt 2, pre change: High-dimensional linear model; coincide with the example I give for the assumption
    d = len(feature)
    np.random.seed(0)
    # e.g. 20% of the entries are NON-missing
    beta1 = random(1, d, density=0.2).A
    return beta1.dot(feature)


def True_mod_lasso_post(feature):
    '''
    Input:
    Output:
    Description:
        f(feature): R^d -> R
    '''
    # Attempt 2, post change: High-dimensional linear model; coincide with the example I give for the assumption
    d = len(feature)
    np.random.seed(1)
    # e.g. 40% of the entries are NON-missing
    beta1 = random(1, d, density=0.4).A
    return beta1.dot(feature)


def True_mod_nonlinear_pre(feature):
    '''
    Input:
    Output:
    Description:
        f(feature): R^d -> R
    '''
    # Attempt 3 Nonlinear model:
    # f(X)=sqrt(1+(beta^TX)+(beta^TX)^2+(beta^TX)^3), where 1 is added in case beta^TX is zero
    d = len(feature)
    np.random.seed(0)
    # e.g. 20% of the entries are NON-missing
    beta1 = random(1, d, density=0.2).A
    betaX = np.abs(beta1.dot(feature))
    return np.sqrt(betaX + betaX**2 + betaX**3)


def True_mod_nonlinear_post(feature, tseries=False):
    d = len(feature)
    np.random.seed(0)
    # e.g. 20% of the entries are NON-missing
    beta1 = random(1, d, density=0.2).A
    betaX = np.abs(beta1.dot(feature))
    if tseries:
        return betaX + betaX**2 + betaX**3
    else:
        return (betaX + betaX**2 + betaX**3)**(2 / 3)


def DGP(True_mod_pre, True_mod_post='', T_tot=1000, tseries=False, high_dim=True, change_points=False, change_frac=0.6, stronglymixing=False):
    '''
    Description:
        Create Y_t=f(X_t)+eps_t, eps_t ~ F from above
        To draw eps_t ~ F, just use F^-1(U).
    '''
    np.random.seed(0)
    Y = np.zeros(T_tot)
    FX = np.zeros(T_tot)
    U = np.random.uniform(size=T_tot)
    Errs = np.zeros(T_tot)
    if stronglymixing:
        Finv = F_inv_stronglymixingDGP
        rho = 0.6
    else:
        Finv = F_inv
        rho = 0
    Errs[0] = Finv(U[0])
    for i in range(1, T_tot):
        Errs[i] = rho * Errs[i - 1] + Finv(U[i])
    # NOTE; T_tot is NOT Ttrain, so if d is too large, we may never recover it well...
    if tseries:
        if change_points:
            # where change point appears
            T_cut = math.ceil(change_frac * (T_tot - 100))
            pre_change = DGP_tseries(
                True_mod_pre, T_cut + 100, Errs[:T_cut + 100])
            post_change = DGP_tseries(
                True_mod_post, T_tot - T_cut, Errs[T_cut:], tseries=True)
            data_full = {}
            for key in pre_change.keys():
                # Note, CANNOT use np.append, as arrays are 2D
                data_full[key] = np.concatenate(
                    (pre_change[key], post_change[key]))
            return data_full
        else:
            return DGP_tseries(True_mod_pre, T_tot, Errs)
    else:
        if high_dim:
            # NOTE: When ||d||_0=c d I need d ~ (1-e^{-1})/c T = (1-e^{-1})/c * (T_tot * train_frac) to AT LEAST allow possible recovery by each S_b. So if I want better approximation (e.g. ||d||_0 = c2 |S_b|), I would let d ~ (1-e^{-1})/c * T_tot*train_frac*c_2. HERE, train_frac=0.5, c=0.2, so we can tweak c2 to roughly have d ~ 0.8 T_tot
            d = math.ceil(T_tot * 0.8)
        else:
            d = math.ceil(T_tot / 10)
        X = np.random.random((T_tot, d))
        if change_points:
            # where change point appears
            T_cut = math.ceil(change_frac * T_tot)
            for i in range(T_cut):
                FX[i] = True_mod_pre(X[i])
                Y[i] = FX[i] + Errs[i]
            for i in range(T_cut, T_tot):
                FX[i] = True_mod_post(X[i])
                Y[i] = FX[i] + Errs[i]
        else:
            for i in range(T_tot):
                FX[i] = True_mod_pre(X[i])
                Y[i] = FX[i] + Errs[i]
        return {'Y': Y, 'X': X, 'f(X)': FX, 'Eps': Errs}


def DGP_tseries(True_mod, T_tot, Errs, tseries=False):
    '''
    Description:
        Create Y_t=f(X_t)+eps_t, eps_t ~ F from above
        To draw eps_t ~ F, just use F^-1(U).
    '''
    np.random.seed(0)
    Y = np.zeros(T_tot)
    FX = np.zeros(T_tot)
    # NOTE; T_tot is NOT Ttrain, so if d is too large, we may never recover it well...
    d = 100  # Can be anything
    X = np.zeros((T_tot - d, d))
    # Initialize the first two by hand, because "True_mod" must take a vector
    Y[0] = Errs[0]
    # Because I assume features are normalized
    FX[1] = np.random.uniform(size=1)
    Y[1] = FX[1] + Errs[1]
    for t in range(2, T_tot):
        if t < d:
            X_t = Y[:t]
        if t > d:
            X_t = Y[t - d:t]
        X_t = (X_t - np.mean(X_t)) / np.std(X_t)
        if t > d:
            X[t - d] = X_t
        if tseries:
            FX[t] = True_mod(X_t, tseries=True)
        else:
            FX[t] = True_mod(X_t)
        Y[t] = FX[t] + Errs[t]
    Y = Y[d:]
    FX = FX[d:]
    Errs = Errs[d:]
    return {'Y': Y, 'X': X, 'f(X)': FX, 'Eps': Errs}


def quick_plt(Data_dc, current_regr, tseries, stronglymixing, change_points=False, args=[]):
    # Easy visualization of data
    fig, ax = plt.subplots(figsize=(3, 3))
    if change_points:
        Tstar, _ = args
        start_plt = Tstar - 50
        end_plt = Tstar + 50

    else:
        start_plt = -100
        end_plt = -1
    ax.plot(Data_dc['Y'][start_plt:end_plt], label=r'$Y_t$')
    ax.plot(Data_dc['f(X)'][start_plt:end_plt], label=r'$f(X_t)$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.27), ncol=2)
    tse = '_tseries' if tseries else ''
    strongm = '_mixing' if stronglymixing else ''
    regr_name = '_' + current_regr.__class__.__name__
    if change_points:
        plt.savefig(
            f'Simulation/Raw_data_changepts{tse}{strongm}{regr_name}.pdf', dpi=300, bbox_inches='tight',
            pad_inches=0)
    else:
        plt.savefig(
            f'Simulation/Raw_data_nochangepts{tse}{strongm}{regr_name}.pdf', dpi=300, bbox_inches='tight',
            pad_inches=0)
    plt.show()


'''Fitting part'''


def split_and_train(Data_dc, train_frac, mathcalA, alpha, itrial, return_full=False):
    '''
    Input:
        alpha and itrial allow us to iterate over alpha and trial number
    '''
    data_y_numpy, data_x_numpy = Data_dc['Y'], Data_dc['X']
    total_data_points = data_y_numpy.shape[0]
    train_size = math.ceil(train_frac * total_data_points)
    # Optional, just because sometimes I want to cut of the single digits (e.g. 501->500)
    train_size = round(train_size / 10) * 10
    X_train = data_x_numpy[:train_size, :]
    X_predict = data_x_numpy[train_size:, :]
    Y_train = data_y_numpy[:train_size]
    Y_predict = data_y_numpy[train_size:]
    current_mod = EnbPI.prediction_interval(
        mathcalA,  X_train, X_predict, Y_train, Y_predict)
    if mathcalA.__class__.__name__ == 'Sequential':
        B = 25
    else:
        B = 50
    result = current_mod.run_experiments(
        alpha=alpha, B=B, stride=1, data_name='Anything', itrial=itrial, miss_test_idx=[], methods=['Ensemble'])
    # NOTE: 'current_mod' include estimated interval centers and widths, and 'results' JUST include average results and name
    if return_full:
        # For more detailed plot
        return [result, current_mod]
    else:
        # For average result
        return result


'''Visualize Actual vs. Predicted Error and intervals'''


def visualize_everything(Data_dc, results, train_frac=0.2, alpha=0.05, change_pts=False, refit=False, arg=[], save_fig=True, tseries=False, stronglymixing=False, first_run=True):
    # 'results' comes from 'split_and_train' above
    result_ave, result_mod = results
    true_errs = Data_dc['Eps']  # Include training data
    Ttrain = math.ceil(train_frac * len(true_errs))
    FX = Data_dc['f(X)']  # Include training data
    Y_predict = Data_dc['Y'][math.ceil(len(FX) * train_frac):]
    FXhat = result_mod.Ensemble_pred_interval_centers  # Only for T+1,...,T+T1
    PI = result_mod.Ensemble_pred_interval_ends  # Only for T+1,...,T+T1
    past_resid = result_mod.Ensemble_online_resid  # Include training LOO residuals
    beta_hat_bin = binning(past_resid, alpha)
    beta_hat_bin  # Estimate
    print(f'Beta^hat_bin is {beta_hat_bin}')
    if stronglymixing:
        savename = 'beta_star_stronglymixing.p'
    else:
        savename = 'beta_star_nostronglymixing.p'
    if first_run:
        beta_star = beta_star_comp(alpha, stronglymixing)  # Actual
        with open(savename, 'wb') as fp:
            pickle.dump(beta_star, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(savename, 'rb') as fp:
            beta_star = pickle.load(fp)
    print(f'Beta^* is {beta_star}')
    # # NOTE: 0-3 below are useful and took me a while to make, but they may NOT be needed now.
    # # 0. Compare f(X_t) & hat f(X_t), t>T
    # fig_fx = EmpvsActual_F([FX[Ttrain:], FXhat])
    # # 1. & 2. Compare actual vs. empirical CDF & PDF
    # if change_pts:
    #     # at T*+T/2+1, for past T/2.
    #     # Same for refit/not refit because we just want to illustrate the benefit refitting brings
    #     Tstar, Thalf = arg
    #     fig_cdf = EmpvsActual_CDF(true_errs[Tstar:Tstar+Thalf], past_resid[Tstar:Tstar+Thalf])
    #     fig_pdf = EmpvsActual_Err(true_errs[Tstar:Tstar+Thalf], past_resid[Tstar:Tstar+Thalf])
    # else:
    #     # at T+1, for past T
    #     fig_cdf = EmpvsActual_CDF(true_errs[:Ttrain], past_resid[:Ttrain])
    #     fig_pdf = EmpvsActual_Err(true_errs[:Ttrain], past_resid[:Ttrain])
    # # 3. Compare actual vs. empirical f(X_t) +/- width, t>T
    # fig_ptwisewidth = EmpvsActual_PtwiseWidth(
    #     beta_star, alpha, FX[-len(FXhat):], FXhat, PI, Y_predict, stronglymixing)
    # 4. Create a simple version
    fig_ptwisewidth_simple = EmpvsActual_PtwiseWidth_simple(
        beta_star, alpha, FX[-len(FXhat):], FXhat, PI, Y_predict, stronglymixing)
    name = 'Simulation'
    if save_fig:
        if change_pts:
            if refit:
                string = '_refit_changepts'
            else:
                string = '_norefit_changepts'
        else:
            string = '_nochangepts'
        regr_name = result_ave['muh_fun'][0]
        tse = '_tseries' if tseries else ''
        strongm = '_mixing' if stronglymixing else ''
        # fig_fx.savefig(
        #     f'{name}/EmpvsActual_FX{string}_{regr_name}{tse}{strongm}.pdf', dpi=300, bbox_inches='tight',
        #     pad_inches=0)
        # fig_cdf.savefig(
        #     f'{name}/EmpvsActual_CDF{string}_{regr_name}{tse}{strongm}.pdf', dpi=300, bbox_inches='tight',
        #     pad_inches=0)
        # fig_pdf.savefig(
        #     f'{name}/EmpvsActual_PDF{string}_{regr_name}{tse}{strongm}.pdf', dpi=300, bbox_inches='tight',
        #     pad_inches=0)
        # fig_ptwisewidth.savefig(
        #     f'{name}/EmpvsActual_PtwiseWidth{string}_{regr_name}{tse}{strongm}.pdf', dpi=300, bbox_inches='tight',
        #     pad_inches=0)
        fig_ptwisewidth_simple.savefig(
            f'{name}/EmpvsActual_PtwiseWidth{string}_{regr_name}{tse}{strongm}_simple.pdf', dpi=300, bbox_inches='tight',
            pad_inches=0)


'''Real-data Section'''
'''Helpers for read data '''


def read_data(i, filename, max_data_size):
    if i == 0:
        '''
            All datasets are Multivariate time-series. They have respective Github for more details as well.
            1. Greenhouse Gas Observing Network Data Set
            Time from 5.10-7.31, 2010, with 4 samples everyday, 6 hours apart between data poits.
            Goal is to "use inverse methods to determine the optimal values of the weights in the weighted sum of 15 tracers that best matches the synthetic observations"
            In other words, find weights so that first 15 tracers will be as close to the last as possible.
            Note, data at many other grid cells are available. Others are in Downloads/🌟AISTATS Data/Greenhouse Data
            https://archive.ics.uci.edu/ml/datasets/Greenhouse+Gas+Observing+Network
        '''
        data = pd.read_csv(filename, header=None, sep=' ').T
        # data.shape  # 327, 16Note, rows are 16 time series (first 15 from tracers, last from synthetic).
    elif i == 1:
        '''
            2. Appliances energy prediction Data Set
            The data set is at 10 min for about 4.5 months.
            The column named 'Appliances' is the response. Other columns are predictors
            https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
        '''
        data = pd.read_csv(filename, delimiter=',')
        # data.shape  # (19736, 29)
        data.drop('date', inplace=True, axis=1)
        data.loc[:, data.columns != 'Appliances']
    elif i == 2:
        '''
            3. Beijing Multi-Site Air-Quality Data Data Set
            This data set includes hourly air pollutants data from 12 nationally-controlled air-quality monitoring sites.
            Time period from 3.1, 2013 to 2.28, 2017.
            PM2.5 or PM10 would be the response.
            https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data
        '''
        data = pd.read_csv(filename)
        # data.shape  # 35064, 18
        # data.columns
        data.drop(columns=['No', 'year', 'month', 'day', 'hour',
                           'wd', 'station'], inplace=True, axis=1)
        data.dropna(inplace=True)
        # data.shape  # 32907, 11
        # data.head(5)
    else:
        """
            4 (Alternative). NREL Solar data at Atlanta Downtown in 2018. 24 observations per day and separately equally by 1H @ half an hour mark everytime
            Data descriptions see Solar Writeup
            Data download:
            (With API) https://nsrdb.nrel.gov/data-sets/api-instructions.html
            (Manual) https://maps.nrel.gov/nsrdb-viewer
        """
        data = pd.read_csv(filename, skiprows=2)
        # data.shape  # 8760, 14
        data.drop(columns=data.columns[0:5], inplace=True)
        data.drop(columns='Unnamed: 13', inplace=True)
        # data.shape  # 8760, 8
        # data.head(5)
    # pick maximum of X data points (for speed)
    data = data.iloc[:min(max_data_size, data.shape[0]), :]
    print(data.shape)
    return data

# Extra real-data for CA and Wind


def read_CA_data(filename):
    data = pd.read_csv(filename)
    # data.shape  # 8760, 14
    data.drop(columns=data.columns[0:6], inplace=True)
    return data


def read_wind_data():
    ''' Note, just use the 8760 hourly observation in 2019
    Github repo is here: https://github.com/Duvey314/austin-green-energy-predictor'''
    data_wind_19 = pd.read_csv('Data/Wind_Hackberry_Generation_2019_2020.csv')
    data_wind_19 = data_wind_19.iloc[:24 * 365, :]
    return data_wind_19


'''Binning Subroutine (used everywhere)'''


def binning(past_resid, alpha):
    '''
    Input:
        past residuals: evident
        alpha: signifance level
    Output:
        beta_hat_bin as argmin of the difference
    Description:
        Compute the beta^hat_bin from past_resid, by breaking [0,alpha] into bins (like 20). It is enough for small alpha
        number of bins are determined rather automatic, relative the size of whole domain
    '''
    bins = 5  # For computation, can just reduce it to like 10 or 5 in real data
    beta_is = np.linspace(start=0, stop=alpha, num=bins)
    width = np.zeros(bins)
    for i in range(bins):
        width[i] = np.percentile(past_resid, math.ceil(100 * (1 - alpha + beta_is[i]))) - \
            np.percentile(past_resid, math.ceil(100 * beta_is[i]))
    i_star = np.argmin(width)
    return beta_is[i_star]


'''Helper for Multi-step ahead inference'''


def missing_data(data, missing_frac, update=False):
    n = len(data)
    idx = np.random.choice(n, size=math.ceil(missing_frac * n), replace=False)
    if update:
        data = np.delete(data, idx, 0)
    idx = idx.tolist()
    return (data, idx)


'''Neural Networks Regressors'''


def keras_mod():
    model = Sequential(name='NeuralNet')
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='relu'))
    return model


def keras_rnn():
    model = Sequential(name='RNN')
    # For fast cuDNN implementation, activation = 'relu' does not work
    model.add(LSTM(100, activation='tanh', return_sequences=True))
    model.add(LSTM(100, activation='tanh'))
    model.add(Dense(1, activation='relu'))
    return model


'''Helper for ensemble'''


def generate_bootstrap_samples(n, m, B):
    '''
      Return: B-by-m matrix, where row b gives the indices for b-th bootstrap sample
    '''
    samples_idx = np.zeros((B, m), dtype=int)
    for b in range(B):
        sample_idx = np.random.choice(n, m)
        samples_idx[b, :] = sample_idx
    return(samples_idx)


def one_dimen_transform(Y_train, Y_predict, d):
    n = len(Y_train)
    n1 = len(Y_predict)
    X_train = np.zeros((n - d, d))  # from d+1,...,n
    X_predict = np.zeros((n1, d))  # from n-d,...,n+n1-d
    for i in range(n - d):
        X_train[i, :] = Y_train[i:i + d]
    for i in range(n1):
        if i < d:
            X_predict[i, :] = np.r_[Y_train[n - d + i:], Y_predict[:i]]
        else:
            X_predict[i, :] = Y_predict[i - d:i]
    Y_train = Y_train[d:]
    return([X_train, X_predict, Y_train, Y_predict])


'''Helper for doing online residual'''


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


'''Helper for Weighted ICP'''


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


"""
For comparing and plotting
(a) f(X_t) vs hat f(X_t)
(b) F vs. F_hat and {eps_t} vs. {eps_t hat}
"""
# (a)


def EmpvsActual_F(value_ls):
    ''' Used for comparing actual vs. estimated CDF and PDF (Histogram)
        value_ls=[actual_errors,estimate_errors]
        which='CDF' or 'PDF' (e.g. Histogram)
    '''
    plt.rcParams.update({'font.size': 18})
    FX, FXhat = value_ls
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(FX[-100:], color="black",
            label=r"$f(X_t)$")
    ax.plot(FXhat[-100:], color="blue",
            label=r"$\hat{f}(X_t)$")
    ax.legend(loc='center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.show()
    return fig

# (b)


def val_to_pdf_or_cdf(value_ls, which):
    ''' Used for comparing actual vs. estimated CDF and PDF (Histogram)
        value_ls=[actual_errors,estimate_errors]
        which='CDF' or 'PDF' (e.g. Histogram)
    '''
    plt.rcParams.update({'font.size': 18})
    bins = 50
    # First on CDF
    count_t, bins_count_t = np.histogram(value_ls[0], bins=bins)
    count_e, bins_count_e = np.histogram(value_ls[1], bins=bins)
    pdf_t = count_t / sum(count_t)
    pdf_e = count_e / sum(count_e)
    cdf_t = np.cumsum(pdf_t)
    cdf_e = np.cumsum(pdf_e)
    fig, ax = plt.subplots(figsize=(3, 3))
    if which == 'PDF':
        ax.plot(bins_count_t[1:], pdf_t, color="black",
                label=r"$\{\epsilon_t\}_{t=1}^T$")
        ax.plot(bins_count_e[1:], pdf_e, color="blue",
                label=r"$\{\hat{\epsilon_t}\}_{t=1}^T$")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.27), ncol=2)
    else:
        ax.plot(bins_count_t[1:], cdf_t, color="black", label=r"$F_{T+1}$")
        ax.plot(bins_count_e[1:], cdf_e, color="blue",
                label=r'$\hat{F}_{T+1}$')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.27), ncol=2)
    plt.show()
    return fig


def EmpvsActual_CDF(true_errs, past_resid):
    '''
    Description:
        (Before Prediction) Overlap empirical CDF on top of actual CDF
    '''
    # Example: https://www.geeksforgeeks.org/how-to-calculate-and-plot-a-cumulative-distribution-function-with-matplotlib-in-python/
    return(val_to_pdf_or_cdf([true_errs, past_resid], which='CDF'))


def EmpvsActual_Err(true_errs, past_resid):
    '''
    Description:
        (Before Prediction) Overlap empirical histogram/PDF on top of actual errors
    '''
    return(val_to_pdf_or_cdf([true_errs, past_resid], which='PDF'))


def EmpvsActual_PtwiseWidth(beta_star, alpha, FX, FXhat, PI, Y_predict, stronglymixing, change_pts=False, args=[]):
    '''
    Input:
        FX: True FX, size T1 for a particular T
        FXhat: Estimated FX from estimator, size T1 for a particular T
        PI: estimated upper and lower intervals, size [T1,2] for a particular T
    Description:
        (After Prediction)
        Side-by-side Plot f(X)+/- actual width vs. hat f(X)+/- estimated width.
        This is for a particular trial, since we plot over t >= T
    '''
    plt.rcParams.update({'font.size': 18,
                         'legend.fontsize': 15})
    if stronglymixing:
        Finv = F_inv_stronglymixing
    else:
        Finv = F_inv
    upper_t = FX + Finv(1 - alpha + beta_star)
    lower_t = FX + Finv(beta_star)
    upper_e, lower_e = np.array(PI['upper']), np.array(PI['lower'])
    fig, ax = plt.subplots(1, 2, figsize=(14, 3), sharey=True)
    legend_loc = (0.5, 1.65)
    if change_pts:
        Tstar, Tnew = args
        start_plt = Tstar - 50
        end_plt = Tstar + Tnew + 50

    else:
        start_plt = -100
        end_plt = -1
    # True values
    ax[0].plot(FX[start_plt:end_plt], color='black', label=r'$f(X_t)$')
    wid = 1
    ax[0].plot(Y_predict[start_plt:end_plt], color='red',
               label=r'$Y_t$', linewidth=wid)
    ax[0].plot(upper_t[start_plt:end_plt], color='blue',
               label=r'$f(X_t)+F_t^{-1}(1-\alpha+\beta^*)$')
    ax[0].plot(lower_t[start_plt:end_plt], color='orange',
               label=r'$f(X_t)+F_t^{-1}(\beta^*)$')
    ax[0].set_xlabel('Prediction Time Index')
    ax[0].legend(loc='upper center', bbox_to_anchor=legend_loc,
                 title=r'Oracle Intervals $C^{\alpha}_t$', ncol=2)
    # Estimated values
    ax[1].plot(FXhat[start_plt:end_plt],
               color='black', label=r'$\hat{f}(X_t)$')
    ax[1].plot(Y_predict[start_plt:end_plt], color='red',
               label=r'$Y_t$', linewidth=wid)
    ax[1].plot(upper_e[start_plt:end_plt], color='blue',
               label=r'$\hat{f}(X_t)+\hat{F}_t^{-1}(1-\alpha+\hat{\beta}_{\rm{bin}})$')
    ax[1].plot(lower_e[start_plt:end_plt], color='orange',
               label=r'$\hat{f}(X_t)+\hat{F}_t^{-1}(\hat{\beta}_{\rm{bin}})$')
    legend_loc = (0.5, 1.65)
    ax[1].legend(loc='upper center', bbox_to_anchor=legend_loc,
                 title=r'Estimated Intervals $\hat{C}^{\alpha}_t$', ncol=2)
    ax[1].set_xlabel('Prediction Time Index')
    # hide tick and tick label of the big axis
    plt.show()
    # plt.tick_params(labelcolor='none', which='both', top=False,
    #                 bottom=False, left=False, right=False)
    # plt.xlabel(r"Prediction Index $t$")
    # plt.ylabel(r"Response Value $Y$")
    return fig


def EmpvsActual_PtwiseWidth_simple(beta_star, alpha, FX, FXhat, PI, Y_predict, stronglymixing, change_pts=False, args=[]):
    '''
    # NOTE: this is modified from "EmpvsActual_PtwiseWidth", so some inputs are not used
    Input:
        FX: True FX, size T1 for a particular T
        FXhat: Estimated FX from estimator, size T1 for a particular T
        PI: estimated upper and lower intervals, size [T1,2] for a particular T
    Description:
        (After Prediction)
        Side-by-side Plot f(X)+/- actual width vs. hat f(X)+/- estimated width.
        This is for a particular trial, since we plot over t >= T
    '''
    plt.rcParams.update({'font.size': 18,
                         'legend.fontsize': 15})
    if stronglymixing:
        Finv = F_inv_stronglymixing
    else:
        Finv = F_inv
    upper_e, lower_e = np.array(PI['upper']), np.array(PI['lower'])
    fig, ax = plt.subplots(figsize=(12, 3))
    legend_loc = (0.5, 1.65)
    if change_pts:
        Tstar, Tnew = args
        start_plt = Tstar - 50
        end_plt = Tstar + Tnew + 50

    else:
        start_plt = -100
        end_plt = -1
    # True values
    ax.plot(Y_predict[start_plt:end_plt], color='orange', label=r'$Y_t$')
    ax.plot(FXhat[start_plt:end_plt], color='blue', label=r'$\hat{Y_t}$')
    ax.fill_between(range(np.abs(start_plt - end_plt)),
                    lower_e[start_plt:end_plt], upper_e[start_plt:end_plt], color='blue', alpha=0.2)
    ax.set_xlabel('Prediction Time Index')
    # legend_loc = (0.5, 1.1)
    ax.legend(loc='upper center', ncol=2)
    # ax.set_title('Estimation and Prediction Intervals')
    plt.show()
    return fig


def EmpvsActual_AveWidth(beta_star, alpha, mean_width_dic, mean_cov_dic, stronglymixing=False, cond_cov=False):
    '''
    Input:
        mean_width_dic: {T_ls: est_mean_width}, contains what is the average
            width of intervals over test points when T_ls fraction of total data is used for training
    Description:
        (After Prediction) average est. widths vs. oracle widths (horizontal line)
        This is for different training sizes T
    '''
    plt.rcParams.update({'font.size': 18,
                         'legend.fontsize': 15})
    if cond_cov:
        fig, ax = plt.subplots(figsize=(9, 3))
    else:
        fig, ax = plt.subplots(figsize=(12, 3))
    dic = pd.DataFrame(mean_width_dic.items(), columns=[
                       'T_ls', 'est_mean_width'])
    est_mean_width, T_ls = dic['est_mean_width'], dic['T_ls']
    ax.plot(T_ls, est_mean_width, marker='o', color='blue',
            label='EnbPI')  # plot x and y using blue circle markers
    if stronglymixing:
        Finv = F_inv_stronglymixing
    else:
        Finv = F_inv
    oracle = Finv(1 - alpha + beta_star) - Finv(beta_star)
    ax.axhline(y=oracle,
               color='blue', linestyle='dashed', label='Oracle')
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, oracle, "{:.1f}".format(oracle), color='blue', transform=trans,
            ha="right", va="center", weight='bold')
    ax.tick_params(axis='y', pad=25)
    [t.set_color('blue') for t in ax.yaxis.get_ticklabels()]
    ax.set_xlabel(r'$\%$ of Total Data')
    # ax.legend(loc='center right', bbox_to_anchor=(0.5, 1.2), title='Width', ncol=2)
    ax2 = ax.twinx()
    dic = pd.DataFrame(mean_cov_dic.items(), columns=['T_ls', 'est_mean_cov'])
    est_mean_cov, T_ls = dic['est_mean_cov'], dic['T_ls']
    # plot x and y using blue circle markers
    ax2.plot(T_ls, est_mean_cov, marker='o', color='red', label='EnbPI')
    [t.set_color('red') for t in ax2.yaxis.get_ticklabels()]
    ax2.axhline(y=1 - alpha,
                color='red', linestyle='dotted', label='Target')
    ax2.set_ylim(0.8, 1)
    ax.tick_params(axis='y', color='red')
    # ax2.legend(loc='center left', bbox_to_anchor=(0.5, 1.2), title='Marginal Coverage', ncol=2)
    if cond_cov:
        plt.rcParams.update(
            {'legend.fontsize': 11, 'legend.title_fontsize': 11})
        x1 = 0
        same = 0.24
        ax.legend(loc='lower left', title='Conditional Width',
                  bbox_to_anchor=(x1, same), ncol=2)
        ax2.legend(loc='lower left', bbox_to_anchor=(
            x1 + 0.32, same), title='Conditional Coverage', ncol=2)
        ax2.set_ylim(0.85, 1.05)
    else:
        same = 0.05
        ax.legend(loc='lower left', title='Width',
                  bbox_to_anchor=(0, same), ncol=2)
        ax2.legend(loc='lower left', bbox_to_anchor=(
            0.335, same), title='Marginal Coverage', ncol=2)
    return fig


"""
For Plotting results: average width and marginal coverage plots
"""


def plot_average_new(x_axis, x_axis_name, save=True, Dataname=['Solar_Atl'], two_rows=True):
    """Plot mean coverage and width for different PI methods and regressor combinations side by side,
       over rho or train_size or alpha_ls
       Parameters:
        data_type: simulated (2-by-3) or real data (2-by-2)
        x_axis: either list of train_size, or alpha
        x_axis_name: either train_size or alpha
    """
    ncol = 2
    Dataname.append(Dataname[0])  # for 1D results
    if two_rows:
        fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
    else:
        fig, ax = plt.subplots(1, 4, figsize=(16, 4), sharex=True)
    j = 0
    filename = {'alpha': 'alpha', 'train_size': 'train'}
    one_D = False
    for data_name in Dataname:
        # load appropriate data
        if j == 1 or one_D:
            results = pd.read_csv(
                f'Results/{data_name}_many_{filename[x_axis_name]}_new_1d.csv')
        else:
            results = pd.read_csv(
                f'Results/{data_name}_many_{filename[x_axis_name]}_new.csv')
        methods_name = ['ARIMA', 'ExpSmoothing', 'DynamicFactor', 'Ensemble']
        cov_together = []
        width_together = []
        # Loop through dataset name and plot average coverage and width for the particular regressor
        muh_fun = np.unique(results[results.method == 'Ensemble']
                            ['muh_fun'])  # First ARIMA, then Ensemble
        tseries_mtd = methods_name[:3]
        for method in methods_name:
            print(method)
            if method in tseries_mtd:
                results_method = results[(results['method'] == method)]
                if data_name == 'Network':
                    method_cov = results_method.groupby(
                        by=[x_axis_name, 'node'], as_index=False).mean().groupby(x_axis_name)['coverage'].describe()  # Column with 50% is median
                    method_width = results_method.groupby(
                        by=[x_axis_name, 'node'], as_index=False).mean().groupby(x_axis_name)['width'].describe()  # Column with 50% is median
                else:
                    method_cov = results_method.groupby(
                        x_axis_name)['coverage'].describe()  # Column with 50% is median
                    method_width = results_method.groupby(
                        x_axis_name)['width'].describe()  # Column with 50% is median
                    method_cov['se'] = method_cov['std'] / \
                        np.sqrt(method_cov['count'])
                    method_width['se'] = method_width['std'] / \
                        np.sqrt(method_width['count'])
                    cov_together.append(method_cov)
                    width_together.append(method_width)
            else:
                for fit_func in muh_fun:
                    results_method = results[(results['method'] == method) &
                                             (results['muh_fun'] == fit_func)]
                    if data_name == 'Network':
                        method_cov = results_method.groupby(
                            by=[x_axis_name, 'node'], as_index=False).mean().groupby(x_axis_name)['coverage'].describe()  # Column with 50% is median
                        method_width = results_method.groupby(
                            by=[x_axis_name, 'node'], as_index=False).mean().groupby(x_axis_name)['width'].describe()  # Column with 50% is median
                    else:
                        method_cov = results_method.groupby(
                            x_axis_name)['coverage'].describe()  # Column with 50% is median
                        method_width = results_method.groupby(
                            x_axis_name)['width'].describe()  # Column with 50% is median
                    method_cov['se'] = method_cov['std'] / \
                        np.sqrt(method_cov['count'])
                    method_width['se'] = method_width['std'] / \
                        np.sqrt(method_width['count'])
                    cov_together.append(method_cov)
                    width_together.append(method_width)
        # Plot
        # Parameters
        num_method = len(tseries_mtd) + len(muh_fun)  # ARIMA + EnbPI
        colors = cm.rainbow(np.linspace(0, 1, num_method))
        mtds = np.append(tseries_mtd, muh_fun)
        # label_names = methods_name
        label_names = {'ARIMA': 'ARIMA',
                       'ExpSmoothing': 'ExpSmoothing',
                       'DynamicFactor': 'DynamicFactor',
                       'RidgeCV': 'EnbPI Ridge',
                       'RandomForestRegressor': 'EnbPI RF', 'Sequential': 'EnbPI NN', 'RNN': 'EnbPI RNN'}
        first = 0
        second = 1
        if one_D:
            first = 2
            second = 3
        axisfont = 20
        titlefont = 24
        tickfont = 16
        name = 'mean'
        for i in range(num_method):
            if two_rows:
                # Coverage
                ax[j, first].plot(x_axis, cov_together[i][name], linestyle='-',
                                  marker='o', label=label_names[mtds[i]], color=colors[i])
                ax[j, first].fill_between(x_axis, cov_together[i][name] - cov_together[i]['se'],
                                          cov_together[i][name] + cov_together[i]['se'], alpha=0.35, facecolor=colors[i])
                ax[j, first].set_ylim(0.7, 1)
                ax[j, first].tick_params(
                    axis='both', which='major', labelsize=tickfont)
                # Width
                ax[j, second].plot(x_axis, width_together[i][name], linestyle='-',
                                   marker='o', label=label_names[mtds[i]], color=colors[i])
                ax[j, second].fill_between(x_axis, width_together[i][name] - width_together[i]['se'],
                                           width_together[i][name] + width_together[i]['se'], alpha=0.35, facecolor=colors[i])
                ax[j, second].tick_params(
                    axis='both', which='major', labelsize=tickfont)
                # Legends, target coverage, labels...
                # Set label
                ax[j, first].plot(
                    x_axis, x_axis, linestyle='-.', color='green')
                # x_ax = ax[j, first].axes.get_xaxis()
                # x_ax.set_visible(False)
                nrow = len(Dataname)
                ax[nrow - 1, 0].set_xlabel(r'$1-\alpha$', fontsize=axisfont)
                ax[nrow - 1, 1].set_xlabel(r'$1-\alpha$', fontsize=axisfont)
            else:
                # Coverage
                ax[first].plot(x_axis, cov_together[i][name], linestyle='-',
                               marker='o', label=label_names[mtds[i]], color=colors[i])
                ax[first].fill_between(x_axis, cov_together[i][name] - cov_together[i]['se'],
                                       cov_together[i][name] + cov_together[i]['se'], alpha=0.35, facecolor=colors[i])
                ax[first].set_ylim(0.7, 1)
                ax[first].tick_params(
                    axis='both', which='major', labelsize=tickfont)
                # Width
                ax[second].plot(x_axis, width_together[i][name], linestyle='-',
                                marker='o', label=label_names[mtds[i]], color=colors[i])
                ax[second].fill_between(x_axis, width_together[i][name] - width_together[i]['se'],
                                        width_together[i][name] + width_together[i]['se'], alpha=0.35, facecolor=colors[i])
                ax[second].tick_params(
                    axis='both', which='major', labelsize=tickfont)
                # Legends, target coverage, labels...
                # Set label
                ax[first].plot(x_axis, x_axis, linestyle='-.', color='green')
                # x_ax = ax[j, first].axes.get_xaxis()
                # x_ax.set_visible(False)
                ax[first].set_xlabel(r'$1-\alpha$', fontsize=axisfont)
                ax[second].set_xlabel(r'$1-\alpha$', fontsize=axisfont)
        if two_rows:
            j += 1
        else:
            one_D = True
    if two_rows:
        ax[0, 0].set_title('Coverage', fontsize=axisfont)
        ax[0, 1].set_title('Width', fontsize=axisfont)
    else:
        ax[0].set_title('Coverage', fontsize=axisfont)
        ax[1].set_title('Width', fontsize=axisfont)
        ax[2].set_title('Coverage', fontsize=axisfont)
        ax[3].set_title('Width', fontsize=axisfont)
    if two_rows:
        ax[0, 0].set_ylabel('Multivariate', fontsize=axisfont)
        ax[1, 0].set_ylabel('Unitivariate', fontsize=axisfont)
    else:
        ax[0].set_ylabel('Multivariate', fontsize=axisfont)
        ax[2].set_ylabel('Unitivariate', fontsize=axisfont)
    fig.tight_layout(pad=0)
    if two_rows:
        # ax[0, 1].legend(loc='upper left', fontsize=axisfont-2)
        ax[1, 1].legend(loc='upper center',
                        bbox_to_anchor=(-0.08, -0.18), ncol=3, fontsize=axisfont - 2)
    else:
        # # With only ARIMA
        # ax[3].legend(loc='upper center',
        #              bbox_to_anchor=(-0.75, -0.18), ncol=5, fontsize=axisfont-2)
        ax[3].legend(loc='upper center',
                     bbox_to_anchor=(-0.68, -0.18), ncol=4, fontsize=axisfont - 2)
    if save:
        if two_rows:
            fig.savefig(
                f'{Dataname[0]}_mean_coverage_width_{x_axis_name}.pdf', dpi=300, bbox_inches='tight',
                pad_inches=0)
        else:
            fig.savefig(
                f'{Dataname[0]}_mean_coverage_width_{x_axis_name}_one_row.pdf', dpi=300, bbox_inches='tight',
                pad_inches=0)


def grouped_box_new(dataname, type, extra_save=''):
    '''First (Second) row contains grouped boxplots for multivariate (univariate) for Ridge, RF, and NN.
       Each boxplot contains coverage and width for all three PI methods over 3 (0.1, 0.3, 0.5) train/total data, so 3*3 boxes in total
       extra_save is for special suffix of plot (such as comparing NN and RNN)'''
    font_size = 18
    label_size = 20
    results = pd.read_csv(f'Results/{dataname}_many_train_new{extra_save}.csv')
    results.sort_values('method', inplace=True, ascending=True)
    results.loc[results.method == 'Ensemble', 'method'] = 'EnbPI'
    results.loc[results.method == 'Weighted_ICP', 'method'] = 'Weighted ICP'
    results_1d = pd.read_csv(
        f'Results/{dataname}_many_train_new_1d{extra_save}.csv')
    results_1d.sort_values('method', inplace=True, ascending=True)
    results_1d.loc[results_1d.method == 'Ensemble', 'method'] = 'EnbPI'
    results_1d.loc[results_1d.method ==
                   'Weighted_ICP', 'method'] = 'Weighted ICP'
    if 'Sequential' in np.array(results.muh_fun):
        results['muh_fun'].replace({'Sequential': 'NeuralNet'}, inplace=True)
        results_1d['muh_fun'].replace(
            {'Sequential': 'NeuralNet'}, inplace=True)
    regrs = np.unique(results.muh_fun)
    regrs_label = {'RidgeCV': 'Ridge', 'LassoCV': 'Lasso', 'RandomForestRegressor': "RF",
                   'NeuralNet': "NN", 'RNN': 'RNN', 'GaussianProcessRegressor': 'GP'}
    # Set up plot
    ncol = 2  # Compare RNN vs NN
    if len(regrs) > 2:
        ncol = 3  # Ridge, RF, NN
        regrs = ['RidgeCV', 'NeuralNet', 'RNN']
    if type == 'coverage':
        f, ax = plt.subplots(2, ncol, figsize=(
            3 * ncol, 6), sharex=True, sharey=True)
    else:
        # all plots in same row share y-axis
        f, ax = plt.subplots(2, ncol, figsize=(
            3 * ncol, 6), sharex=True, sharey=True)
    f.tight_layout(pad=0)
    # Prepare for plot
    d = 20
    results_1d.train_size += d  # for plotting purpose
    tot_data = math.ceil(max(results.train_size) / 0.278)
    results['ratio'] = np.round(results.train_size / tot_data, 2)
    results_1d['ratio'] = np.round(results_1d.train_size / tot_data, 2)
    j = 0  # column, denote aggregator
    ratios = np.unique(results['ratio'])
    # train_size_for_plot = [ratios[2], ratios[4], ratios[6], ratios[9]] # This was for 4 boxplots in one figure
    train_size_for_plot = ratios
    for regr in regrs:
        mtd = ['EnbPI', 'ICP', 'Weighted ICP']
        mtd_colors = ['red', 'royalblue', 'black']
        color_dict = dict(zip(mtd, mtd_colors))  # specify colors for each box
        # Start plotting
        which_train_idx = [
            fraction in train_size_for_plot for fraction in results.ratio]
        which_train_idx_1d = [
            fraction in train_size_for_plot for fraction in results_1d.ratio]
        results_plt = results.iloc[which_train_idx, ]
        results_1d_plt = results_1d.iloc[which_train_idx_1d, ]
        sns.boxplot(y=type, x='ratio',
                    data=results_plt[results_plt.muh_fun == regr],
                    palette=color_dict,
                    hue='method', ax=ax[0, j], showfliers=False, linewidth=0.4)
        sns.boxplot(y=type, x='ratio',
                    data=results_1d_plt[results_1d_plt.muh_fun == regr],
                    palette=color_dict,
                    hue='method', ax=ax[1, j], showfliers=False, linewidth=0.4)
        # # Add text, as the boxes are too things to be distinguishable
        # for i in range(len(mtd)):
        #     m = mtd[i]
        #     for k in [0, 1]:
        #         ax[k, j].text(i-1, results_plt[(results_plt.muh_fun == regr) & (
        #             results_plt.method == m)][type].max()+1, m, ha='center', color=mtd_colors[i], fontsize=12)
        for i in range(2):
            ax[i, j].tick_params(axis='both', which='major', labelsize=14)
            if type == 'coverage':
                ax[i, j].axhline(y=0.9, color='black', linestyle='dashed')
            # Control legend
            ax[i, j].get_legend().remove()
            # Control y and x-label
            if j == 0:
                # Y-label on
                ax[0, 0].set_ylabel('Multivariate', fontsize=label_size)
                ax[1, 0].set_ylabel('Univariate', fontsize=label_size)
                if i == 1:
                    # X-label on
                    ax[1, j].set_xlabel(
                        r'$\%$ of Total Data', fontsize=label_size)
                else:
                    # X-label off
                    x_axis = ax[i, j].axes.get_xaxis()
                    x_axis.set_visible(False)
            else:
                y_label = ax[i, j].axes.get_yaxis().get_label()
                y_label.set_visible(False)
                if type == 'coverage':
                    # Y-label off
                    y_axis = ax[i, j].axes.get_yaxis()
                    y_axis.set_visible(False)
                if i == 1:
                    # X-label on
                    ax[1, j].set_xlabel(
                        r'$\%$ of Total Data', fontsize=label_size)
                else:
                    # X-label off
                    x_axis = ax[i, j].axes.get_xaxis()
                    x_axis.set_visible(False)
            # Control Title
            if i == 0:
                ax[0, j].set_title(regrs_label[regr], fontsize=label_size)
        j += 1
        # Legend lastly
    # Assign to top middle
    # ax[1, 1].legend(loc='upper center',
    #                 bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=font_size)
    plt.legend(loc='upper center',
               bbox_to_anchor=(-0.15, -0.25), ncol=3, fontsize=font_size)
    plt.savefig(
        f'{dataname}_boxplot_{type}{extra_save}.pdf', dpi=300, bbox_inches='tight',
        pad_inches=0)


def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1, :].flat:
            ax.xaxis.set_tick_params(
                which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:, 1:].flat:
            ax.yaxis.set_tick_params(
                which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)


def grouped_box_new_with_JaB(dataname):
    '''First (Second) row contains grouped boxplots for multivariate (univariate) for Ridge, RF, and NN.
       Each boxplot contains coverage and width for all three PI methods over 3 (0.1, 0.3, 0.5) train/total data, so 3*3 boxes in total
       extra_save is for special suffix of plot (such as comparing NN and RNN)'''
    font_size = 18
    label_size = 20
    results = pd.read_csv(f'Results/{dataname}_many_train_new_with_JaB.csv')
    results.sort_values('method', inplace=True, ascending=True)
    results.loc[results.method == 'Ensemble', 'method'] = 'EnbPI'
    # results.loc[results.method == 'Weighted_ICP', 'method'] = 'Weighted ICP'
    results.loc[results.method == 'ICP',
                'method'] = 'Split Conformal, or Chernozhukov etal (2018,2020)'
    results.loc[results.method == 'JaB', 'method'] = 'J+aB (Kim etal 2020)'
    if 'Sequential' in np.array(results.muh_fun):
        results['muh_fun'].replace({'Sequential': 'NeuralNet'}, inplace=True)
    regrs = np.unique(results.muh_fun)
    regrs_label = {'RidgeCV': 'Ridge', 'LassoCV': 'Lasso', 'RandomForestRegressor': "RF",
                   'NeuralNet': "NN", 'RNN': 'RNN', 'GaussianProcessRegressor': 'GP'}
    # Set up plot
    ncol = 2  # Compare RNN vs NN
    if len(regrs) > 2:
        ncol = 6  # Ridge, RF, NN
        regrs = ['RidgeCV', 'NeuralNet', 'RNN']
    f, ax = plt.subplots(1, ncol, figsize=(3 * ncol, 3), sharex=True)
    # f.tight_layout(pad=0)
    set_share_axes(ax[:3], sharey=True)
    set_share_axes(ax[3:], sharey=True)
    # Prepare for plot
    tot_data = math.ceil(max(results.train_size) / 0.278)
    results['ratio'] = np.round(results.train_size / tot_data, 2)
    j = 0  # column, denote aggregator
    ratios = np.unique(results['ratio'])
    # train_size_for_plot = [ratios[2], ratios[4], ratios[6], ratios[9]] # This was for 4 boxplots in one figure
    train_size_for_plot = ratios
    for regr in regrs:
        # mtd = ['EnbPI', 'ICP', 'Weighted ICP']
        # mtd_colors = ['red', 'royalblue', 'black']
        mtd = [
            'EnbPI', 'Split Conformal, or Chernozhukov etal (2018,2020)', 'J+aB (Kim etal 2020)']
        mtd_colors = ['red', 'royalblue', 'black']
        color_dict = dict(zip(mtd, mtd_colors))  # specify colors for each box
        # Start plotting
        which_train_idx = [
            fraction in train_size_for_plot for fraction in results.ratio]
        results_plt = results.iloc[which_train_idx, ]
        ax1 = sns.boxplot(y='coverage', x='ratio',
                          data=results_plt[results_plt.muh_fun == regr],
                          palette=color_dict,
                          hue='method', ax=ax[j], showfliers=False, width=1, saturation=1, linewidth=0.4)
        ax2 = sns.boxplot(y='width', x='ratio',
                          data=results_plt[results_plt.muh_fun == regr],
                          palette=color_dict,
                          hue='method', ax=ax[j + 3], showfliers=False, width=1, saturation=1, linewidth=0.4)
        for i, artist in enumerate(ax1.artists):
            if i % 3 == 0:
                col = mtd_colors[0]
            elif i % 3 == 1:
                col = mtd_colors[1]
            else:
                col = mtd_colors[2]
            # This sets the color for the main box
            artist.set_edgecolor(col)
        for i, artist in enumerate(ax2.artists):
            if i % 3 == 0:
                col = mtd_colors[0]
            elif i % 3 == 1:
                col = mtd_colors[1]
            else:
                col = mtd_colors[2]
            # This sets the color for the main box
            artist.set_edgecolor(col)
            # for k in range(6*i, 6*(i+1)):
            #     ax2.lines[k].set_color(col)
        ax[j].tick_params(axis='both', which='major', labelsize=14)
        if j <= 2:
            ax[j].axhline(y=0.9, color='black', linestyle='dashed')
        # Control legend
        ax[j].get_legend().remove()
        ax[j + 3].get_legend().remove()
        # Control y and x-label
        ax[j].set_xlabel(r'$\%$ of Total Data', fontsize=label_size)
        ax[j + 3].set_xlabel(r'$\%$ of Total Data', fontsize=label_size)
        if j == 0:
            # Y-label on
            ax[j].set_ylabel('Coverage', fontsize=label_size)
            ax[j + 3].set_ylabel('Width', fontsize=label_size)
        else:
            y_label = ax[j].axes.get_yaxis().get_label()
            y_label.set_visible(False)
            y_axis = ax[j].axes.get_yaxis()
            y_axis.set_visible(False)
            y_label = ax[j + 3].axes.get_yaxis().get_label()
            y_label.set_visible(False)
            y_axis = ax[j + 3].axes.get_yaxis()
            y_axis.set_visible(False)
        # Control Title
        ax[j].set_title(regrs_label[regr], fontsize=label_size)
        ax[j + 3].set_title(regrs_label[regr], fontsize=label_size)
        j += 1
        # Legend lastly
    # plt.legend(loc='upper center',
    #            bbox_to_anchor=(-0.15, -0.25), ncol=3, fontsize=font_size)
    plt.tight_layout(pad=0)
    plt.legend(loc='upper right',
               bbox_to_anchor=(1, -0.3), ncol=3, fontsize=font_size)
    plt.savefig(
        f'{dataname}_boxplot_rebuttal.pdf', dpi=300, bbox_inches='tight',
        pad_inches=0)


'''For Conditional Coverage---Plotting'''


def PI_on_series_plus_cov_or_not(results, stride, which_hours, which_method, regr_method, Y_predict, no_slide=False, five_in_a_row=True):
    # Plot PIs on predictions for the particular hour
    # At most three plots in a row (so that figures look appropriately large)
    plt.rcParams.update({'font.size': 18})
    if five_in_a_row:
        ncol = 5
    else:
        ncol = 4
    nrow = np.ceil(len(which_hours) / ncol).astype(int)
    if stride == 24 or stride == 14 or stride == 15:
        # Multi-row
        fig, ax = plt.subplots(nrow * 2, ncol, figsize=(ncol * 4, nrow * 5), sharex='row',
                               sharey='row', constrained_layout=True)
    else:
        fig, ax = plt.subplots(2, 5, figsize=(5 * 4, 5), sharex='row',
                               sharey='row', constrained_layout=True)
    if stride > 24:
        # Because we focused on near-noon-data
        n1 = int(results[0].shape[0] / 5)
    else:
        n1 = int(results[0].shape[0] / stride)
    plot_length = 91  # Plot 3 months, April-June
    method_ls = {'Ensemble': 0, 'ICP': 1, 'WeightedICP': 2}
    results_by_method = results[method_ls[which_method]]
    for i in range(len(which_hours)):
        hour = which_hours[i]
        if stride > 24:
            indices_at_hour = np.arange(n1) * 5 + hour
        else:
            indices_at_hour = np.arange(n1) * stride + hour
        to_plot = indices_at_hour[:plot_length]
        row = (i // ncol) * 2
        col = np.mod(i, ncol)
        covered_or_not = []
        for j in range(n1):
            if Y_predict[indices_at_hour[j]] >= results_by_method['lower'][indices_at_hour[j]] and Y_predict[indices_at_hour[j]] <= results_by_method['upper'][indices_at_hour[j]]:
                covered_or_not.append(1)
            else:
                covered_or_not.append(0)
        coverage = np.mean(covered_or_not)
        coverage = np.round(coverage, 2)
        # Plot PI on data
        train_size = 92
        rot_angle = 15
        x_axis = np.arange(plot_length)
        if stride == 24 or stride == 14 or stride == 15:
            current_figure = ax[row, col]
        else:
            col = np.mod(i, 5)
            current_figure = ax[0, col]
        current_figure.scatter(
            x_axis, Y_predict[to_plot], marker='.', s=4, color='black')
        current_figure.plot(
            x_axis, results_by_method['center'][to_plot], color='red', linewidth=0.7)
        lower_vals = np.maximum(0, results_by_method['lower'][to_plot])
        upper_vals = np.maximum(0, results_by_method['upper'][to_plot])
        current_figure.fill_between(x_axis, lower_vals, upper_vals, alpha=0.3)
        # current_figure.plot(x_axis, np.maximum(0, results_by_method['upper'][to_plot]))
        # current_figure.plot(x_axis, np.maximum(0, results_by_method['lower'][to_plot]))
        # For axis purpose, subtract June
        xticks = np.linspace(0, plot_length - 30, 3).astype(int)
        xtick_labels = [calendar.month_name[int(i / 30) + 4]
                        for i in xticks]  # Get months, start from April
        current_figure.set_xticks(xticks)
        current_figure.set_xticklabels(xtick_labels)
        current_figure.tick_params(axis='x', rotation=rot_angle)
        # Title
        if stride == 24:
            current_figure.set_title(f'At {hour}:00 \n Coverage is {coverage}')
        elif stride == 5 or no_slide:
            current_figure.set_title(
                f'At {hour+10}:00 \n Coverage is {coverage}')
        else:
            if stride == 15:
                current_figure.set_title(
                    f'At {hour+5}:00 \n Coverage is {coverage}')
            else:
                current_figure.set_title(
                    f'At {hour+6}:00 \n Coverage is {coverage}')
        # if stride == 14:
        #     # Sub data`
        #     current_figure.set_title(f'At {hour+6}:00 \n Coverage is {coverage}')
        # elif stride == 24:
        #     # Full data
        #     current_figure.set_title(f'At {hour}:00 \n Coverage is {coverage}')
        # else:
        #     # Near noon data
        #     current_figure.set_title(f'At {hour+10}:00 \n Coverage is {coverage}')
        # Plot cover or not over test period
        x_axis = np.arange(n1)
        if stride == 24 or stride == 14 or stride == 15:
            current_figure = ax[row + 1, col]
        else:
            col = np.mod(i, 5)
            current_figure = ax[1, col]
        current_figure.scatter(x_axis, covered_or_not, marker='.', s=0.4)
        current_figure.set_ylim([-1, 2])
        # For axis purpose, subtract December
        xticks = np.linspace(0, n1 - 31, 3).astype(int)
        xtick_labels = [calendar.month_name[int(
            i / 30) + 4] for i in xticks]  # Get months
        current_figure.set_xticks(xticks)
        current_figure.set_xticklabels(xtick_labels)
        current_figure.tick_params(axis='x', rotation=rot_angle)
        yticks = [0, 1]
        current_figure.set_yticks(yticks)
        current_figure.set_yticklabels(['Uncovered', 'Covered'])
        # xticks = current_figure.get_xticks()  # Actual numbers
        # xtick_labels = [f'T+{int(i)}' for i in xticks]
        # current_figure.set_xticklabels(xtick_labels)
    # if no_slide:
    #     fig.suptitle(
    #         f'EnbPI Intervals under {regr_method} without sliding', fontsize=22)
    # else:
    #     fig.suptitle(
    #         f'EnbPI Intervals under {regr_method} with s={stride}', fontsize=22)
    return fig


def make_cond_plots(Data_name, results_ls, no_slide, missing, one_d, five_in_a_row=True):
    for data_name in Data_name:
        result_ridge, result_rf, result_nn, stride, Y_predict = results_ls[data_name]
        res = [result_ridge, result_rf, result_nn]
        if no_slide:
            which_hours = [0, 1, 2, 3, 4]  # 10AM-2PM
        else:
            if stride == 24:
                if five_in_a_row:
                    which_hours = [7, 8, 9, 16, 17, 10, 11, 12, 13, 14]
                else:
                    which_hours = [7, 8, 10, 11, 12, 13, 14, 16, 17]
            elif stride == 5:
                which_hours = [0, 1, 2, 3, 4]
            else:
                if five_in_a_row:
                    if data_name == 'Solar_Atl':
                        which_hours = [i - 6 for i in [7, 8,
                                                       9, 16, 17, 10, 11, 12, 13, 14]]
                    else:
                        which_hours = [i - 5 for i in [7, 8,
                                                       9, 16, 17, 10, 11, 12, 13, 14]]
                else:
                    if data_name == 'Solar_Atl':
                        # which_hours = [i-6 for i in [7, 8, 10, 11, 12, 13, 14, 16, 17]]
                        which_hours = [
                            i - 6 for i in [8, 9, 16, 17, 11, 12, 13, 14]]
                    else:
                        # which_hours = [i-5 for i in [7, 8, 10, 11, 12, 13, 14, 16, 17]]
                        which_hours = [
                            i - 6 for i in [8, 9, 16, 17, 11, 12, 13, 14]]
        which_method = 'Ensemble'
        regr_methods = {0: 'Ridge', 1: 'RF', 2: 'NN'}
        X_data_type = {True: 'uni', False: 'multi'}
        Xtype = X_data_type[one_d]
        slide = '_no_slide' if no_slide else '_daily_slide'
        Dtype = {24: '_fulldata', 14: '_subdata',
                 15: '_subdata', 5: '_near_noon_data'}
        if no_slide:
            dtype = ''
        else:
            dtype = Dtype[stride]
        miss = '_with_missing' if missing else ''
        for i in range(len(res)):
            regr_method = regr_methods[i]
            fig = PI_on_series_plus_cov_or_not(
                res[i], stride, which_hours, which_method, regr_method, Y_predict, no_slide, five_in_a_row)
            fig.savefig(f'{data_name}_{regr_method}_{Xtype}_PI_on_series_plus_cov_or_not{slide}{dtype}{miss}.pdf', dpi=300, bbox_inches='tight',
                        pad_inches=0)


def make_cond_plots_Solar_Atl(results_dict, regr_name, Y_predict_ls, stride_ls):
    plt.rcParams.update(
        {'font.size': 24, 'axes.titlesize': 24, 'axes.labelsize': 24})
    fig, ax = plt.subplots(4, 4, figsize=(4 * 7, 6 * 2), sharex='row',
                           sharey='row', constrained_layout=True)
    results_ls, resid_ls = results_dict[regr_name]
    i = 0
    row_ix = 0
    col_ix = 0
    hour_label = {0: [8, 9, 15, 16], 1: [10, 11, 12, 13]}
    k = 0
    for result, Y_predict, stride, resid in zip(results_ls, Y_predict_ls, stride_ls, resid_ls):
        tot_hour = min([stride, 4])
        n1 = int(Y_predict.shape[0] / stride)
        for hour in range(tot_hour):
            if k <= 3:
                row = 0
                col = k
            else:
                row = 2
                col = k - 4
            k += 1
            indices_at_hour = np.arange(n1) * tot_hour + hour
            covered_or_not = []
            for j in range(n1):
                if Y_predict[indices_at_hour[j]] >= result['lower'][indices_at_hour[j]] and Y_predict[indices_at_hour[j]] <= result['upper'][indices_at_hour[j]]:
                    covered_or_not.append(1)
                else:
                    covered_or_not.append(0)
            coverage = np.mean(covered_or_not)
            coverage = np.round(coverage, 2)
            # Plot
            current_figure = ax[row, col]
            plot_length = 92
            x_axis = np.arange(plot_length)
            to_plot = indices_at_hour[:plot_length]
            current_figure.scatter(
                x_axis, Y_predict[to_plot], marker='.', s=4, color='black')
            current_figure.plot(
                x_axis, result['center'][to_plot], color='red', linewidth=0.7)
            lower_vals = np.maximum(0, result['lower'][to_plot])
            upper_vals = np.maximum(0, result['upper'][to_plot])
            current_figure.fill_between(
                x_axis, lower_vals, upper_vals, alpha=0.3)
            xticks = np.linspace(0, plot_length - 30, 3).astype(int)  #
            xtick_labels = [calendar.month_name[int(i / 30) + 4]
                            for i in xticks]  # Get months, start from April
            current_figure.set_xticks(xticks)
            current_figure.set_xticklabels(xtick_labels)
            current_figure.tick_params(axis='x', rotation=15)
            hour_name = hour_label[i][hour]
            current_figure.set_title(
                f'At {hour_name+1}:00 \n Coverage is {coverage}')
            current_figure = ax[row + 1, col]
            # # Histogram plot
            # sns.histplot(resid, bins=15, kde=True, ax=current_figure)
            # current_figure.axes.get_yaxis().set_visible(False)
            # current_figure.set_title(r'Histogram of $\{\hat{\epsilon_t}\}_{t=1}^T$')
            # Moving coverage
            N = 30  # e.g. average over past 20 days
            moving_cov = np.convolve(
                covered_or_not, np.ones(N) / N, mode='valid')
            current_figure.plot(moving_cov, color='red',
                                label='Sliding Coverage')
            # For axis purpose, subtract December
            xticks = np.linspace(0, len(covered_or_not) -
                                 31 - N + 1, 3).astype(int)
            xtick_labels = [calendar.month_name[int(
                i / 30) + 4 + N // 30] for i in xticks]  # Get months
            current_figure.set_xticks(xticks)
            current_figure.set_xticklabels(xtick_labels)
            current_figure.tick_params(axis='x', rotation=15)
            current_figure.axhline(
                0.9, color='black', linewidth=3, linestyle='--')
        i += 1
    fig2, ax = plt.subplots(2, 1, figsize=(7, 6 * 2))
    # Histogram plot
    sns.histplot(resid_ls[0], bins=15, kde=True, ax=ax[0])
    ax[0].axes.get_yaxis().set_visible(False)
    ax[0].set_title(r'Histogram of $\{\hat{\epsilon_t}\}_{t=1}^T$')
    sns.histplot(resid_ls[1], bins=15, kde=True, ax=ax[1])
    ax[1].axes.get_yaxis().set_visible(False)
    fig2.tight_layout(pad=2)
    return [fig, fig2]
