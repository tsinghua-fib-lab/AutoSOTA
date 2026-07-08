## Code Snipptes from the Demo code notebook
##

import pickle
from random import random

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor

from enbPI import utils_latest

EnbPI_DATASETS = ['simulation_statespace', 'simulation_nonstationary', 'simulate_heteroskedastic',
                  'solar', 'solar_nonstat', 'electric', 'wind']


def demo_code_data_load(dataset_name: str, data_path):
    if dataset_name not in EnbPI_DATASETS:
        raise ValueError(f"Invalid Dataset! (name: {dataset_name}")
    dataset, _, dataset_postfix = dataset_name.partition("_")
    if dataset == 'simulation':
        if dataset_postfix == 'statespace':
            Data_dict = get_new_data_simple(num_pts=2000, alpha=0.9, beta=0.9)
        if dataset_postfix == 'nonstationary':
            data_container = utils_latest.data_loader()
            _, Data_dict = data_container.get_non_stationary_simulate()
            Data_dict['X'] = torch.from_numpy(Data_dict['X']).float()
            Data_dict['Y'] = torch.from_numpy(Data_dict['Y']).float()
        if dataset_postfix == 'heteroskedastic':
            # NOTE: somehow for this case, currently RF quantile regression does not yield shorter interval. We may tune past window to get different results (like decrease it to 250) if need
            Data_dict = get_new_data()
        X_full = Data_dict['X']
        Y_full = Data_dict['Y']
    elif dataset == 'solar':
        # Get solar data WITH time t as covariate
        dloader = utils_latest.data_loader()
        Y_full, X_full_old, X_full_nonstat = dloader.get_non_stationary_real(univariate=False, path=data_path)
        if dataset_postfix == 'nonstat':
            X_full = X_full_nonstat
        else:
            X_full = X_full_old
        X_full = torch.from_numpy(X_full).float()
        Y_full = torch.from_numpy(Y_full).float()
    elif dataset == 'electric':
        X_full, Y_full = electric_dataset(data_path)
        X_full = torch.from_numpy(X_full).float()
        Y_full = torch.from_numpy(Y_full).float()
    elif dataset == 'wind':
        X_full, Y_full = wind_dataset(data_path)
        X_full = torch.from_numpy(X_full).float()
        Y_full = torch.from_numpy(Y_full).float()
    else:
        raise ValueError(f"Invalid Dataset! (name: {dataset_name}")
    return X_full, Y_full


def dataset_depending_model_options(dataset_name: str):
    """
    :return: B (no of bootstrap batches), past_window, fit_sigmaX, fit_func
    """
    dataset, _, dataset_postfix = dataset_name.partition("_")
    if dataset == 'simulation':
        return {'no_of_bootstrap_batches': 20,
                'past_window': 500,
                'fit_sigmaX': True if dataset_postfix == 'heteroskedastic' else False,
                'fit_func': None
                }
    elif dataset == 'solar':
        return {'no_of_bootstrap_batches': 25,
                'past_window': 500,
                'fit_sigmaX': False,
                'fit_func': RandomForestRegressor(n_estimators=10, criterion='mse', bootstrap=False, n_jobs=-1)
                }
    elif dataset == 'electric':
        return {'no_of_bootstrap_batches': 25,
                'past_window': 300,
                'fit_sigmaX': False,
                'fit_func': RandomForestRegressor(n_estimators=10, max_depth=1, criterion='mse', bootstrap=False, n_jobs=-1)
                }


def get_new_data():
    ''' Note, the difference from earlier case 3 in paper is that
        1) I reduce d from 100 to 20,
        2) I let X to be different, so sigmaX differs
            The sigmaX is a linear model so this effect in X is immediate
        I keep the same AR(1) eps & everything else.'''
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
        return (betaX + betaX**2 + betaX**3)**(1/4)
    Tot, d = 2000, 20
    Fmap = True_mod_nonlinear_pre
    # Multiply each random feature by exponential component, which is repeated every Tot/rep elements
    rep = 10
    mult = np.exp(np.repeat(np.linspace(0, 2, rep), Tot/rep)).reshape(Tot, 1)
    X = np.random.rand(Tot, d)*mult
    fX = np.array([Fmap(x) for x in X]).flatten()
    beta_Sigma = 0.1*np.ones(d)
    sigmaX = np.maximum(X.dot(beta_Sigma).T, 0)
    with open(f'Data_nochangepts_nonlinear.p', 'rb') as fp:
        Data_dc = pickle.load(fp)
    eps = Data_dc['Eps']
    Y = fX + sigmaX*eps
    np.random.seed(1103)
    idx = np.random.choice(Tot, Tot, replace=False)
    Y, X, fX, sigmaX, eps = Y[idx], X[idx], fX[idx], sigmaX[idx], eps[idx]
    return {'Y': torch.from_numpy(Y).float(), 'X': torch.from_numpy(X).float(), 'f(X)': fX, 'sigma(X)': sigmaX, 'Eps': eps}


def get_new_data_simple(num_pts, alpha, beta):
    '''
        Y_t = alpha*Y_{t-1}+\eps_t
        \eps_t = beta*\eps_{t-1}+v_t
        v_t ~ N(0,1)
        So X_t = Y_{t-1}, f(X_t) = alpha*X_t
        If t = 0:
            X_t = 0, Y_t=\eps_t = v_t
    '''
    v0 = torch.randn(1)
    Y, X, fX, eps = [v0], [torch.zeros(1)], [torch.zeros(1)], [v0]
    scale = torch.sqrt(torch.ones(1)*0.1)
    for _ in range(num_pts-1):
        vt = torch.randn(1)*scale
        X.append(Y[-1])
        fX.append(alpha*Y[-1])
        eps.append(beta*eps[-1]+vt)
        Y.append(fX[-1]+eps[-1])
    Y, X, fX, eps = torch.hstack(Y), torch.vstack(
        X), torch.vstack(fX), torch.hstack(eps)
    return {'Y': Y.float(), 'X': X.float(), 'f(X)': fX, 'Eps': eps}


def electric_dataset(path):
    # ELEC2 data set
    # downloaded from https://www.kaggle.com/yashsharan/the-elec2-dataset
    data = pd.read_csv(path)
    col_names = data.columns
    data = data.to_numpy()

    # remove the first stretch of time where 'transfer' does not vary
    data = data[17760:]

    # set up variables for the task (predicting 'transfer')
    covariate_col = ['nswprice', 'nswdemand', 'vicprice', 'vicdemand']
    response_col = 'transfer'
    # keep data points for 9:00am - 12:00pm
    keep_rows = np.where((data[:, 2] > data[17, 2])
                         & (data[:, 2] < data[24, 2]))[0]

    X = data[keep_rows][:, np.where(
        [t in covariate_col for t in col_names])[0]]
    Y = data[keep_rows][:, np.where(col_names == response_col)[0]].flatten()
    X = X.astype('float64')
    Y = Y.astype('float64')

    return X, Y

def wind_dataset(path):
    df = pd.read_csv(path)
    df = df.iloc[: 24 * 365, :]
    data = df['MWH'].values
    lag = 10
    data_x = sliding_window_view(data, lag)[:-1]
    data_y = data[lag:]

    return data_x, data_y