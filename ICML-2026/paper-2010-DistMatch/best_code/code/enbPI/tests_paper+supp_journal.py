import pickle
import utils_ECAD_journal as utils_ECAD
from scipy.stats import skew
import seaborn as sns
import PI_class_EnbPI_journal as EnbPI
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN   # kNN detector
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import utils_EnbPI_journal as util
from matplotlib.lines import Line2D  # For legend handles
import calendar
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
import itertools
import importlib
import time
import pandas as pd
import numpy as np
import os
import sys
import keras
warnings.filterwarnings("ignore")
importlib.reload(sys.modules['PI_class_EnbPI_journal'])

'''Sec 5.2 Figure 3 on comparing with time-series methods
   Silimar results in the appendix are included as well'''


def big_transform(CA_cities, current_city, one_dim, train_size):
    # Used for California data
    # Next, merge these data (so concatenate X_t and Y_t for one_d or not)
    # Return [X_train, X_test, Y_train, Y_test] from data_x and data_y
    # Data_x is either multivariate (direct concatenation)
    # or univariate (transform each series and THEN concatenate the transformed series)
    big_X_train = []
    big_X_predict = []
    for city in CA_cities:
        data = eval(f'data{city}')  # Pandas DataFrame
        data_x = data.loc[:, data.columns != 'DHI']
        data_y = data['DHI']
        data_x_numpy = data_x.to_numpy()  # Convert to numpy
        data_y_numpy = data_y.to_numpy()  # Convert to numpy
        X_train = data_x_numpy[:train_size, :]
        X_predict = data_x_numpy[train_size:, :]
        Y_train_del = data_y_numpy[:train_size]
        Y_predict_del = data_y_numpy[train_size:]
        if city == current_city:
            Y_train = Y_train_del
            Y_predict = Y_predict_del
        if one_dim:
            X_train, X_predict, Y_train_del, Y_predict_del = util.one_dimen_transform(
                Y_train_del, Y_predict_del, d=20)
            big_X_train.append(X_train)
            big_X_predict.append(X_predict)
            if city == current_city:
                Y_train = Y_train_del
        else:
            big_X_train.append(X_train)
            big_X_predict.append(X_predict)
    X_train = np.hstack(big_X_train)
    X_predict = np.hstack(big_X_predict)
    return([X_train, X_predict, Y_train, Y_predict])


# Read data and initialize parameters
result_type = 'Fig3'
response_ls = {'Solar_Atl': 'DHI', 'Palo_Alto': 'DHI', 'Wind_Austin': 'MWH',
               'green_house': 15, 'appliances': 'Appliances', 'Beijing_air': 'PM2.5', }
if result_type == 'Fig3':
    # Figure 3
    max_data_size = 10000
    dataSolar_Atl = util.read_data(3, 'Data/Solar_Atl_data.csv', max_data_size)
    Data_name = ['Solar_Atl']
    CA_energy_data = False
elif result_type == 'AppendixB3':
    # Results in Appendix B.3
    CA_cities = ['Fremont', 'Milpitas', 'Mountain_View', 'North_San_Jose',
                 'Palo_Alto', 'Redwood_City', 'San_Mateo', 'Santa_Clara',
                 'Sunnyvale']
    for city in CA_cities:
        globals()['data%s' % city] = util.read_CA_data(f'Data/{city}_data.csv')
    Data_name = ['Palo_Alto']
    CA_energy_data = True
else:
    # Results in Appendix B.4
    datagreen_house = util.read_data(
        0, 'Data/green_house_data.csv', max_data_size)
    dataappliances_data = util.read_data(
        1, 'Data/appliances_data.csv', max_data_size)
    dataBeijing_air = util.read_data(
        2, 'Data/Beijing_air_Tiantan_data.csv', max_data_size)
    Data_name = ['green_house', 'appliances', 'Beijing_air']
min_alpha = 0.0001
max_alpha = 10
ridge_cv = RidgeCV(alphas=np.linspace(min_alpha, max_alpha, 10))
random_forest = RandomForestRegressor(n_estimators=10, criterion='mse',
                                      bootstrap=False, max_depth=2, n_jobs=-1)


alpha_ls = np.linspace(0.05, 0.25, 5)

importlib.reload(sys.modules['PI_class_EnbPI_journal'])
# First run time-series methods
for data_name in Data_name:
    one_dim = True
    itrial = 0
    data = eval(f'data{data_name}')  # Pandas DataFrame
    data_x = data.loc[:, data.columns != response_ls[data_name]]
    data_y = data[response_ls[data_name]]
    data_x_numpy = data_x.to_numpy()  # Convert to numpy
    data_y_numpy = data_y.to_numpy()  # Convert to numpy
    total_data_points = data_x_numpy.shape[0]
    train_size = int(0.2 * total_data_points)
    results_ts = pd.DataFrame(columns=['itrial', 'dataname',
                                       'method', 'alpha', 'coverage', 'width'])
    np.random.seed(98765 + itrial)
    for alpha in alpha_ls:
        print(f'At trial # {itrial} and alpha={alpha}')
        print(f'For {data_name}')
        if CA_energy_data:
            X_train, X_predict, Y_train, Y_predict = big_transform(
                Data_name, data_name, one_dim, train_size)
            d = 20
        else:
            X_train = data_x_numpy[:train_size, :]
            X_predict = data_x_numpy[train_size:, :]
            Y_train = data_y_numpy[:train_size]
            Y_predict = data_y_numpy[train_size:]
        ridge_results = EnbPI.prediction_interval(
            ridge_cv,  X_train, X_predict, Y_train, Y_predict)
        # For ARIMA and other time-series methods, only run once
        result_ts = ridge_results.run_experiments(
            alpha, stride, data_name, itrial, none_CP=True)
        result_ts.rename(columns={'train_size': 'alpha'}, inplace=True)
        if CA_energy_data:
            result_ts['alpha'].replace(
                train_size - d, alpha, inplace=True)
        else:
            result_ts['alpha'].replace(
                train_size, alpha, inplace=True)
        results_ts = pd.concat([results_ts, result_ts])
        results_ts.to_csv(
            f'Results/{data_name}_many_alpha_new_tseries.csv', index=False)


# Then run conformal-related methods
stride = 1
miss_test_idx = []
alpha = 0.1
tot_trial = 10  # For CP methods that randomizes
B = 50  # number of bootstrap samples
rnn = True
methods = ['Ensemble', 'ICP', 'Weighted_ICP']
# NOTE, if want to run J+aB (Kim et al. 2020), then let
# methods = ['Ensemble', 'ICP', 'Weighted_ICP', 'JaB']
for one_dim in [True, False]:
    for data_name in Data_name:
        data = eval(f'data{data_name}')  # Pandas DataFrame
        data_x = data.loc[:, data.columns != response_ls[data_name]]
        data_y = data[response_ls[data_name]]
        data_x_numpy = data_x.to_numpy()  # Convert to numpy
        data_y_numpy = data_y.to_numpy()  # Convert to numpy
        total_data_points = data_x_numpy.shape[0]
        train_size = int(0.2 * total_data_points)
        results = pd.DataFrame(columns=['itrial', 'dataname', 'muh_fun',
                                        'method', 'alpha', 'coverage', 'width'])
        for itrial in range(tot_trial):
            np.random.seed(98765 + itrial)
            nnet = util.keras_mod()
            rnnet = util.keras_rnn()
            if CA_energy_data:
                X_train, X_predict, Y_train, Y_predict = big_transform(
                    Data_name, data_name, one_dim, train_size)
                d = 20
            else:
                X_train = data_x_numpy[:train_size, :]
                X_predict = data_x_numpy[train_size:, :]
                Y_train = data_y_numpy[:train_size]
                Y_predict = data_y_numpy[train_size:]
                d = 20  # for 1-d memory depth
                if one_dim:
                    X_train, X_predict, Y_train, Y_predict = util.one_dimen_transform(
                        Y_train, Y_predict, d=d)
            ridge_results = EnbPI.prediction_interval(
                ridge_cv,  X_train, X_predict, Y_train, Y_predict)
            ridge_results.fit_bootstrap_models_online(B, miss_test_idx)
            rf_results = EnbPI.prediction_interval(
                random_forest,  X_train, X_predict, Y_train, Y_predict)
            rf_results.fit_bootstrap_models_online(B, miss_test_idx)
            nn_results = EnbPI.prediction_interval(
                nnet,  X_train, X_predict, Y_train, Y_predict)
            nn_results.fit_bootstrap_models_online(B, miss_test_idx)
            if rnn:
                T, k = X_train.shape
                T1 = X_predict.shape[0]
                X_train = X_train.reshape((T, 1, k))
                X_predict = X_predict.reshape((T1, 1, k))
                rnn_results = EnbPI.prediction_interval(
                    rnnet, X_train, X_predict, Y_train, Y_predict)
                rnn_results.fit_bootstrap_models_online(B, miss_test_idx)
            if 'JaB' in method:
                n = len(X_train)
                B_jab = math.ceil(np.random.binomial(math.ceil(B / (1 - 1. / (1 + train_size))**n),
                                                 (1 - 1. / (1 + train_size))**n, size=1))
                ridge_results.fit_bootstrap_models(B_jab)
                rf_results.fit_bootstrap_models(B_jab)
                nn_results.fit_bootstrap_models(B_jab)
                if rnn:
                    rnn_results.fit_bootstrap_models(B_jab)
             for alpha in alpha_ls:
                # Note, this is necessary because a model may "remember the past"
                print(f'At trial # {itrial} and alpha={alpha}')
                print(f'For {data_name}')
                # CP Methods
                print(f'regressor is {ridge_cv.__class__.__name__}')
                result_ridge = ridge_results.run_experiments(
                    alpha, stride, data_name, itrial, methods=methods)
                print(f'regressor is {random_forest.__class__.__name__}')
                result_rf = rf_results.run_experiments(
                    alpha, stride, data_name, itrial, methods=methods)
                print(f'regressor is {nnet.name}')
                start = time.time()
                result_nn = nn_results.run_experiments(
                    alpha, stride, data_name, itrial, methods=methods)
                if rnn:
                    print(f'regressor is {rnnet.name}')
                    result_rnn = rnn_results.run_experiments(
                        alpha, stride, data_name, itrial, methods=methods)
                    result_rnn['muh_fun'] = 'RNN'
                    results_now = pd.concat(
                        [result_ridge, result_rf, result_nn, result_rnn])
                else:
                    results_now = pd.concat(
                        [result_ridge, result_rf, result_nn])
                results_now.rename(
                    columns={'train_size': 'alpha'}, inplace=True)
                if one_dim:
                    results_now['alpha'].replace(
                        train_size - d, alpha, inplace=True)
                else:
                    results_now['alpha'].replace(
                        train_size, alpha, inplace=True)
                results = pd.concat([results, results_now])
                if one_dim:
                    results.to_csv(
                        f'Results/{data_name}_many_alpha_new_1d.csv', index=False)
                else:
                    results.to_csv(
                        f'Results/{data_name}_many_alpha_new.csv', index=False)


def merge_tseries(data_name, which):
    data1 = pd.read_csv(f'Results/{data_name}_many_alpha_new_tseries.csv')
    data2 = pd.read_csv(f'Results/{data_name}_many_alpha_new{which}.csv')
    data1 = pd.concat((data1, data2))
    data1.reset_index(inplace=True)
    print(data1.shape)
    data1.to_csv(f'Results/{data_name}_many_alpha_new{which}.csv', index=False)


for data_name in Data_name:
    merge_tseries(data_name, '_1d')
    merge_tseries(data_name, '')

# Make plot
alpha_ls = np.linspace(0.05, 0.25, 5)
x_axis = 1 - alpha_ls
x_axis_name = 'alpha'
two_rows = False
importlib.reload(sys.modules['utils_EnbPI_journal'])
# Data_name = ['Solar_Atl']
# Data_name = ['Palo_Alto']
# Data_name = ['green_house', 'appliances', 'Beijing_air']
for dataname in Data_name:
    util.plot_average_new(x_axis, x_axis_name, Dataname=[
                          dataname], two_rows=two_rows)


'''Sec 5.2. Figure 4 on comparing with CP methods'''
# NOTE, if want to run J+aB (Kim et al. 2020), then let
methods = ['Ensemble', 'ICP', 'Weighted_ICP']
# methods = ['Ensemble', 'ICP', 'Weighted_ICP', 'JaB']
for one_dim in [True, False]:
    # Run Ridge, Lasso, RF, and NN
    for data_name in Data_name:
        data = eval(f'data{data_name}')  # Pandas DataFrame
        data_x = data.loc[:, data.columns != response_ls[data_name]]
        data_y = data[response_ls[data_name]]
        data_x_numpy = data_x.to_numpy()  # Convert to numpy
        data_y_numpy = data_y.to_numpy()  # Convert to numpy
        total_data_points = data_x_numpy.shape[0]
        Train_size = np.linspace(0.1 * total_data_points,
                                 0.3 * total_data_points, 10).astype(int)
        Train_size = [Train_size[0], Train_size[4], Train_size[8]]
        results = pd.DataFrame(columns=['itrial', 'dataname', 'muh_fun',
                                        'method', 'train_size', 'coverage', 'width'])
        for itrial in range(tot_trial):
            np.random.seed(98765 + itrial)
            for train_size in Train_size:
                # Note, this is necessary because a model may "remember the past"
                nnet = util.keras_mod()
                rnnet = util.keras_rnn()
                print(f'At trial # {itrial} and train_size={train_size}')
                print(f'For {data_name}')
                if CA_energy_data:
                    X_train, X_predict, Y_train, Y_predict = big_transform(
                        Data_name, data_name, one_dim, train_size)
                else:
                    X_train = data_x_numpy[:train_size, :]
                    X_predict = data_x_numpy[train_size:, :]
                    Y_train = data_y_numpy[:train_size]
                    Y_predict = data_y_numpy[train_size:]
                    if one_dim:
                        X_train, X_predict, Y_train, Y_predict = util.one_dimen_transform(
                            Y_train, Y_predict, d=20)
                ridge_results = EnbPI.prediction_interval(
                    ridge_cv, X_train, X_predict, Y_train, Y_predict)
                ridge_results.fit_bootstrap_models_online(B, miss_test_idx)
                rf_results = EnbPI.prediction_interval(
                    random_forest,  X_train, X_predict, Y_train, Y_predict)
                rf_results.fit_bootstrap_models_online(B, miss_test_idx)
                nn_results = EnbPI.prediction_interval(
                    nnet,  X_train, X_predict, Y_train, Y_predict)
                nn_results.fit_bootstrap_models_online(B, miss_test_idx)
                if rnn:
                    T, k = X_train.shape
                    T1 = X_predict.shape[0]
                    X_train = X_train.reshape((T, 1, k))
                    X_predict = X_predict.reshape((T1, 1, k))
                    rnn_results = EnbPI.prediction_interval(
                        rnnet, X_train, X_predict, Y_train, Y_predict)
                    rnn_results.fit_bootstrap_models_online(B, miss_test_idx)
                if 'JaB' in method:
                    n = len(X_train)
                    B_jab = math.ceil(np.random.binomial(math.ceil(B/(1-1./(1+train_size))**n),
                                                     (1-1./(1+train_size))**n, size=1))
                    ridge_results.fit_bootstrap_models(B_jab)
                    rf_results.fit_bootstrap_models(B_jab)
                    nn_results.fit_bootstrap_models(B_jab)
                    if rnn:
                        rnn_results.fit_bootstrap_models(B_jab)
                # For CP Methods
                print(f'regressor is {ridge_cv.__class__.__name__}')
                result_ridge = ridge_results.run_experiments(
                    alpha, stride, data_name, itrial, methods=methods)
                print(f'regressor is {random_forest.__class__.__name__}')
                result_rf = rf_results.run_experiments(
                    alpha, stride, data_name, itrial, methods=methods)
                print(f'regressor is {nnet.name}')
                result_nn = nn_results.run_experiments(
                    alpha, stride, data_name, itrial, methods=methods)
                result_nn['muh_fun'] = 'NeuralNet'
                if rnn:
                    print(f'regressor is {rnnet.name}')
                    result_rnn = rnn_results.run_experiments(
                        alpha, stride, data_name, itrial, methods=methods)
                    result_rnn['muh_fun'] = 'RNN'
                    results = pd.concat(
                        [results, result_ridge, result_rf, result_nn, result_rnn])
                else:
                    results = pd.concat(
                        [results, result_ridge, result_rf, result_nn])
                if one_dim:
                    results.to_csv(
                        f'Results/{data_name}_many_train_new_1d.csv', index=False)
                else:
                    results.to_csv(
                        f'Results/{data_name}_many_train_new.csv', index=False)

# Plot
# Data_name = ['Solar_Atl']
# Data_name = ['Palo_Alto']
# Data_name = ['green_house', 'appliances', 'Beijing_air']
importlib.reload(sys.modules['utils_EnbPI_journal'])
for data_name in Data_name:
    util.grouped_box_new(data_name, 'coverage')
    util.grouped_box_new(data_name, 'width')

importlib.reload(sys.modules['utils_EnbPI_journal'])
util.grouped_box_new_with_JaB('Solar_Atl')
'''Sec 5.3. Figure 5 on conditional coverage
   Silimar results in the appendix are included as well'''


def missing_data(data, missing_frac, update=False):
    n = len(data)
    idx = np.random.choice(n, size=int(missing_frac * n), replace=False)
    if update:
        data = np.delete(data, idx, 0)
    idx = idx.tolist()
    return (data, idx)


def restructure_X_t(darray):
    '''
    For each row i after the first row, take i-1 last entries of the first row and then impute the rest
    Imputation is just generating random N(Y_train_mean, Y_train_std), where
    Y_train is the first row.
    '''
    s = darray.shape[1]
    copy = np.copy(darray)
    for i in range(1, min(s, darray.shape[0])):
        copy[i, :s - i] = copy[0, i:]
        imputed_val = np.abs(np.random.normal(loc=np.mean(
            copy[0]), scale=np.std(copy[0]), size=i))
        copy[i, s - i:] = imputed_val
    return copy


def further_preprocess(data, response_name='DHI', suffix=''):
    '''Extract non-zero hours and also hours between 10AM-2PM (where radiation is high) '''
    max_recorder = pd.DataFrame(np.zeros(24), index=range(0, 24))
    for i in range(0, 24):
        # Check at what times max recording is 0 (meaning no recording yet)
        # 12:00 AM every day. for every later hour, + i \in \{1,...,23\}
        time = np.arange(365) * 24 + i
        max_record = np.max(data[response_name][time])
        max_recorder.iloc[i] = max_record
    # Drop these non-zero things
    data_sub = data.copy()
    to_be_droped = np.where(max_recorder == 0)[0]
    print(to_be_droped)
    drop_idx = []
    if len(to_be_droped) > 0:
        for i in to_be_droped:
            drop_idx.append(np.arange(365) * 24 + i)
        drop_idx = np.hstack(drop_idx)
        data_sub.drop(drop_idx, inplace=True)
    else:
        data_sub = []
    # Create near_noon data between 10AM-2PM
    if suffix == '':
        to_be_included = np.array([10, 11, 12, 13, 14])
    if suffix == '_8_9_15_16_17':
        to_be_included = np.array([8, 9, 15, 16, 17])
    if suffix == '_10_14':
        to_be_included = np.array([10, 11, 12, 13, 14])
    to_be_droped = np.delete(np.arange(24), to_be_included)
    data_near_noon = data.copy()
    drop_idx = []
    for i in to_be_droped:
        drop_idx.append(np.arange(365) * 24 + i)
    drop_idx = np.hstack(drop_idx)
    data_near_noon.drop(drop_idx, inplace=True)
    return [data_sub, data_near_noon]


def big_transform_s_beyond_1(sub, cities, current_city, one_dim, missing, miss_frac=0.25):
    '''Overall, include ALL other cities' data in the CURRENT city being considered.
       1. Check what data is used (full, sub, or near-noon), need sub, but it is now suppressed.
       # NOTE, 1 is suppressed for now, since we are uncertain whether sub or near-noon is needed for Californian results
       2. If missing, process these training and testing data before transform
       -->> Current city and neighbors are assumed to have DIFFERENT missing fractions.
       3. Then, if one_dim, transform data (include past), but since s>1, apply *restructure_X_t* to s rows a time'''
    big_X_train = []
    big_X_predict = []
    big_Y_train = []
    big_Y_predict = []
    stride_ls = []
    for city in cities:
        print(city)
        # Start 1
        if 'Solar_Atl' in city:
            data_full = eval(f'dataSolar_Atl')  # Pandas DataFrame
            suffix = city[9:]
            _, data = further_preprocess(data_full, suffix=suffix)
            if suffix == '_10_14':
                stride = 5
            if suffix == '_8_9_15_16_17':
                stride = 5
        else:
            data_full = eval(f'data{city}')  # Pandas DataFrame
            if city == 'Wind_Austin':
                data_sub, data_near_noon = further_preprocess(
                    data_full, response_name='MWH')
            else:
                data_sub, data_near_noon = further_preprocess(data_full)
            if sub == 0:
                data = data_full
                stride = 24
            elif sub == 1:
                data = data_sub
                stride = int(len(data) / 365)
            else:
                data = data_near_noon
                stride = 5
        train_size = 92 * stride
        col_name = 'MWH' if city == 'Wind_Austin' else 'DHI'
        data_x = data.loc[:, data.columns != col_name]
        data_y = data[col_name]
        data_x_numpy = data_x.to_numpy()  # Convert to numpy
        data_y_numpy = data_y.to_numpy()  # Convert to numpy
        X_train = data_x_numpy[:train_size, :]
        X_predict = data_x_numpy[train_size:, :]
        Y_train_del = data_y_numpy[:train_size]
        Y_predict_del = data_y_numpy[train_size:]
        # Finish 1
        # Start 2
        if missing:
            X_train, miss_train_idx = missing_data(
                X_train, missing_frac=miss_frac, update=True)
            Y_train_del = np.delete(Y_train_del, miss_train_idx)
            Y_predict_del, miss_test_idx = missing_data(
                Y_predict_del, missing_frac=miss_frac, update=False)
            if city == current_city:
                # Need an additional Y_truth
                Y_train = Y_train_del
                Y_predict = Y_predict_del.copy()
                true_miss_text_idx = miss_test_idx
            Y_predict_del[miss_test_idx] = np.abs(np.random.normal(loc=np.mean(
                Y_train_del), scale=np.std(Y_train_del), size=len(miss_test_idx)))

        else:
            true_miss_text_idx = []
            if city == current_city:
                Y_train = Y_train_del
                Y_predict = Y_predict_del
        # Finish 2
        # Start 3
        if one_dim:
            X_train, X_predict, Y_train_del, Y_predict_del = util.one_dimen_transform(
                Y_train_del, Y_predict_del, d=min(stride, 24))  # Note: this handles 'no_slide (stride=infty)' case
            j = 0
            for k in range(len(X_predict) // stride + 1):
                X_predict[j * k:min((j + 1) * k, len(X_predict))
                          ] = restructure_X_t(X_predict[j * k:min((j + 1) * k, len(X_predict))])
                j += 1
            big_X_train.append(X_train)
            big_X_predict.append(X_predict)
            if city == current_city:
                Y_train = Y_train_del
                Y_predict = Y_predict_del
        else:
            big_X_train.append(X_train)
            big_X_predict.append(X_predict)
        # Finish 3
    X_train = np.hstack(big_X_train)
    X_predict = np.hstack(big_X_predict)
    return([X_train, X_predict, Y_train, Y_predict, true_miss_text_idx, stride])


def all_together(Data_name, sub, no_slide, missing, miss_frac=0.25, one_dim=False):
    methods = ['Ensemble']
    train_days = 92
    itrial = 1
    results_ls = {}
    alpha = 0.1
    B = np.random.binomial(100, np.exp(-1))  # number of bootstrap samples
    if 'Solar_Atl' in Data_name:
        Data_name = ['Solar_Atl_8_9_15_16_17', 'Solar_Atl_10_14']
    for data_name in Data_name:
        np.random.seed(98765)
        # Note, this is necessary because a model may "remember the past"
        nnet = util.keras_mod()
        if 'Solar_Atl' in data_name:
            X_train, X_predict, Y_train, Y_predict, miss_test_idx, stride = big_transform_s_beyond_1(
                sub, [data_name], data_name, one_dim, missing)
        else:
            X_train, X_predict, Y_train, Y_predict, miss_test_idx, stride = big_transform_s_beyond_1(
                sub, Data_name, data_name, one_dim, missing)
        train_size = 92 * stride
        print(f'At train_size={train_size}')
        print(f'For {data_name}')
        if no_slide:
            stride = int((365 - 92) * stride)  # No slide at all
        print(stride)
        nnet = util.keras_mod()
        min_alpha = 0.0001
        max_alpha = 10
        ridge_cv = RidgeCV(alphas=np.linspace(min_alpha, max_alpha, 10))
        random_forest = RandomForestRegressor(n_estimators=10, criterion='mse',
                                              bootstrap=False, max_depth=2, n_jobs=-1)
        ridge_results = EnbPI.prediction_interval(
            ridge_cv,  X_train, X_predict, Y_train, Y_predict)
        ridge_results.fit_bootstrap_models_online(B, miss_test_idx)
        rf_results = EnbPI.prediction_interval(
            random_forest,  X_train, X_predict, Y_train, Y_predict)
        rf_results.fit_bootstrap_models_online(B, miss_test_idx)
        nn_results = EnbPI.prediction_interval(
            nnet,  X_train, X_predict, Y_train, Y_predict)
        nn_results.fit_bootstrap_models_online(B, miss_test_idx)
        # For CP Methods
        print(f'regressor is {ridge_cv.__class__.__name__}')
        result_ridge = ridge_results.run_experiments(
            alpha, stride, data_name, itrial, methods=methods, get_plots=True)
        result_ridge[0]['center'] = ridge_results.Ensemble_pred_interval_centers
        print(f'regressor is {random_forest.__class__.__name__}')
        result_rf = rf_results.run_experiments(
            alpha, stride, data_name, itrial, methods=methods, get_plots=True)
        result_rf[0]['center'] = rf_results.Ensemble_pred_interval_centers
        print(f'regressor is {nnet.name}')
        result_nn = nn_results.run_experiments(
            alpha, stride, data_name, itrial, methods=methods, get_plots=True)
        result_nn[0]['center'] = nn_results.Ensemble_pred_interval_centers
        results_ls[data_name] = [result_ridge, result_rf, result_nn, stride, Y_train, Y_predict, ridge_results.Ensemble_online_resid[:train_days],
                                 rf_results.Ensemble_online_resid[:train_days], nn_results.Ensemble_online_resid[:train_days]]
    return results_ls


def small_helper(results_ls):
    names = list(results_ls.keys())
    result_ridge_ls = []
    result_rf_ls = []
    result_nn_ls = []
    Y_train_ls = []
    Y_predict_ls = []
    stride_ls = []
    ridge_resid_ls = []
    rf_resid_ls = []
    nn_resid_ls = []
    for data_name in names:
        result_ridge, result_rf, result_nn, stride, Y_train, Y_predict, ridge_resid, rf_resid, nn_resid = results_ls[
            data_name]
        result_ridge_ls.append(result_ridge[0])
        ridge_resid_ls.append(ridge_resid)
        result_rf_ls.append(result_rf[0])
        rf_resid_ls.append(rf_resid)
        result_nn_ls.append(result_nn[0])
        nn_resid_ls.append(nn_resid)
        Y_train_ls.append(Y_train)
        Y_predict_ls.append(Y_predict)
        stride_ls.append(stride)
    results_dict = {'Ridge': [result_ridge_ls, ridge_resid_ls], 'RF': [
        result_rf_ls, rf_resid_ls], 'NN': [result_nn_ls, nn_resid_ls]}
    return [results_dict, Y_train_ls, Y_predict_ls, stride_ls]


ATL_cities = ['Solar_Atl']
max_data_size = 10000
dataSolar_Atl = util.read_data(3, 'Data/Solar_Atl_data.csv', max_data_size)
# Fig 5: Main paper result.
results_ls_with_missing_and_slide_sub = all_together(
    Data_name=ATL_cities, sub=1, no_slide=False, missing=True, one_dim=True)
results_dict_with_missing_and_slide_sub, Y_train_ls, Y_predict_ls, stride_ls = small_helper(
    results_ls_with_missing_and_slide_sub)
importlib.reload(sys.modules['utils_EnbPI_journal'])
for regr_name in results_dict_with_missing_and_slide_sub.keys():
    fig, fig2 = util.make_cond_plots_Solar_Atl(
        results_dict_with_missing_and_slide_sub, regr_name, Y_predict_ls, stride_ls)
    fig.savefig(f'Solar_Atl_{regr_name}_uni_PI_on_series_plus_cov_or_not_with_missing_sliding_cov.pdf', dpi=300, bbox_inches='tight',
                pad_inches=0)
    fig2.savefig(f'Solar_Atl_{regr_name}_uni_PI_on_series_plus_cov_or_not_with_missing_hist.pdf', dpi=300, bbox_inches='tight',
                 pad_inches=0)

# Fig 9: Appendix figures, s=infty.
ATL_cities = ['Solar_Atl']
results_ls_one_d_no_missing_and_no_slide_near_noon = all_together(
    Data_name=ATL_cities, sub=2, no_slide=True, missing=False, one_dim=True)
util.make_cond_plots(ATL_cities, results_ls_one_d_no_missing_and_no_slide_near_noon,
                     no_slide=True, missing=False, one_d=True, five_in_a_row=True)


# Fig 12 (a): CA in appendix
CA_cities = ['Fremont', 'Milpitas', 'Mountain_View', 'North_San_Jose',
             'Palo_Alto', 'Redwood_City', 'San_Mateo', 'Santa_Clara',
             'Sunnyvale']
for city in CA_cities:
    globals()['data%s' % city] = util.read_CA_data(f'Data/{city}_data.csv')
CA_cities = ['Palo_Alto']
# s=15
results_ls_with_missing_and_slide_sub = all_together(
    Data_name=CA_cities, sub=1, no_slide=False, missing=True, one_dim=False)
# s=4
results_ls_one_d_with_missing_and_slide_near_noon = all_together(
    Data_name=CA_cities, sub=2, no_slide=False, missing=True, one_dim=True)
# Make plots
importlib.reload(sys.modules['utils_EnbPI_journal'])
util.make_cond_plots(CA_cities, results_ls_one_d_with_missing_and_slide_near_noon,
                     no_slide=False, missing=True, one_d=True, five_in_a_row=True)

# Fig 12 (a): Wind in appendix
Wind_cities = ['Wind_Austin']
dataWind_Austin = util.read_wind_data()
results_ls_one_d_with_missing_and_slide_full = all_together(
    Data_name=Wind_cities, sub=0, no_slide=False, missing=True, one_dim=True)
util.make_cond_plots(Wind_cities, results_ls_one_d_with_missing_and_slide_full,
                     no_slide=False, missing=True, one_d=True)

'''Sec 5.4. Figure 6 and Table 1 on Anomaly Detection'''


importlib.reload(sys.modules['utils_ECAD_journal'])
# 1. Training to get F1 score tables for each regressor at each sensor, by fixing a training fraction
ECAD_results_frac = {}
for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
    X_Y_time_dic, m, T_minus_m, T1, K, idx, sensors = utils_ECAD.imput_and_get_X_Y_time_dic(
        frac, first_time=True)
    B = 15
    args = [m, T_minus_m, T1, K, B, idx, sensors]
    alpha = 0.05
    args1 = [m, T1, K, alpha]
    # 1.2 train my detector using different mathcal A
    ECAD_results = {}
    regrs = {'Ridge': RidgeCV(alphas=np.linspace(0.01, 10, 10)),
             'RF': RandomForestRegressor(n_estimators=10, criterion='mse', bootstrap=False, max_depth=2, n_jobs=-1),
             'NN': util.keras_mod(),
             'RNN': util.keras_rnn()}
    for regr_name in regrs.keys():
        regr = regrs[regr_name]
        print(f'At frac={frac}: Running ECAD__{regr}')
        mod_residuals = utils_ECAD.ECAD(X_Y_time_dic, regr, args)
        mod_pred = utils_ECAD.detect_anomalies(mod_residuals, args1)
        mod_summary = utils_ECAD.get_stat(mod_pred, T1)
        ECAD_results[regr_name] = mod_summary
    # 1.3 Comparison against others
    with open(f'Data/Anomaly_Detection_Data/flow_frame_train_{str(frac)}.p', 'rb') as fp:
        train = pickle.load(fp)
    m = 8
    train_size = train.shape[0] - m
    # first four unsupervised
    competitors = ['HBOS()', 'IForest()', 'OCSVM()', 'PCA()',
                   'svm.SVC(gamma="auto")',
                   'GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)',
                   'neighbors.KNeighborsClassifier(n_neighbors=20, weights="distance")',
                   'MLPClassifier(solver="lbfgs", alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)']
    X_Y_loc_dic = utils_ECAD.time_dic_to_loc_dic(frac)
    X_Y_loc_dic_sup = utils_ECAD.time_dic_to_loc_dic(frac, supervised=True)
    for i in range(len(competitors)):
        regr_name = competitors[i]
        print(f'At frac={frac}: Running Competing Methods__{regr_name}')
        supervised = False
        if i > 3:
            supervised = True
        mod_pred = utils_ECAD.mod_to_result(
            regr_name, X_Y_loc_dic_sup, train_size, frac, supervised=supervised)
        mod_summary = utils_ECAD.get_stat(mod_pred, T1)
        ECAD_results[regr_name] = mod_summary
    print(ECAD_results)
    ECAD_results_frac[frac] = ECAD_results
    np.save(f'Results/ECAD_results_{str(frac)}.npy', ECAD_results)

# 2. Get the table: compare methods in terms of F1 scores
frac = 0.5  # From [0.3,0.4,0.5,0.6,0.7]
# Because the np.save saved a 0-dimensional array, need to use "[()]" to access the element inside
ECAD_results = np.load(
    f'Results/ECAD_results_{str(frac)}.npy', allow_pickle=True)[()]
mtd_names = ['Ridge', 'RF', 'NN', 'RNN'] + competitors
K = len(sensors)
A = len(mtd_names)
sensors_dic = dict((k, sensors[k])for k in range(K))
F1_comparison_table = np.zeros((K, A))
Precision_comparison_table = np.zeros((K, A))
Recall_comparison_table = np.zeros((K, A))

for k in range(K):
    s = sensors_dic[k]
    F1_k = np.zeros(A)
    Precision_k = np.zeros(A)
    Recall_k = np.zeros(A)
    for a in range(A):
        mtd_name = mtd_names[a]
        mtd_summary = ECAD_results[mtd_name]
        s_loc = np.where(np.array(mtd_summary.index) == s)[0][0]
        F1_k[a] = mtd_summary['F1 score'].iloc[s_loc]
        Precision_k[a] = mtd_summary['Precision'].iloc[s_loc]
        Recall_k[a] = mtd_summary['Recall'].iloc[s_loc]
    F1_comparison_table[k] = F1_k
    Precision_comparison_table[k] = Precision_k
    Recall_comparison_table[k] = Recall_k

mtd_names_dict = {'Ridge': 'Ridge', 'RF': 'RF', 'NN': 'NN', 'RNN': 'RNN', 'HBOS()': 'HBOS', 'IForest()': 'IForest', 'OCSVM()': 'OCSVM', 'PCA()': 'PCA',
                  'svm.SVC(gamma="auto")': 'SVC',
                  'GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)': 'GBoosting',
                  'neighbors.KNeighborsClassifier(n_neighbors=20, weights="distance")': 'KNN',
                  'MLPClassifier(solver="lbfgs", alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)': 'MLPClassifier'}
mtd_names_idx = [mtd_names_dict[name] for name in mtd_names]
F1_comparison_table_latex = pd.DataFrame(
    F1_comparison_table, index=mtd_summary.index, columns=mtd_names_idx).sort_values('RNN', ascending=False)
F1_comparison_table_latex.round(2)

Precision_comparison_table_latex = pd.DataFrame(
    Precision_comparison_table, index=mtd_summary.index, columns=mtd_names_idx).reindex(F1_comparison_table_latex.index)
Precision_comparison_table_latex.round(2)

Recall_comparison_table_latex = pd.DataFrame(
    Recall_comparison_table, index=mtd_summary.index, columns=mtd_names_idx).reindex(F1_comparison_table_latex.index)
Recall_comparison_table_latex.round(2)
with open('Data/Anomaly_Detection_Data/true_anomalies.p', 'rb') as fp:
    true_anomalies = pickle.load(fp)
anomly_frac = true_anomalies.mean()
anomly_frac.reindex(F1_comparison_table_latex.index)


# Fig 6. Plot the Precision, Recall, and F1 map
results = pd.DataFrame(columns=['train_frac', 'method',
                                'precision', 'recall', 'F1'])
sensor_selected = 282
for train_frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
    ECAD_results = np.load(
        f'Results/ECAD_results_{str(train_frac)}.npy', allow_pickle=True)[()]
    for mtd_name in mtd_names:
        mod_summary = ECAD_results[mtd_name]
        _, _, F1, recall, precision = mod_summary[mod_summary.index ==
                                                  sensor_selected].iloc[0]
        results.loc[len(results)] = [train_frac, mtd_names_dict[mtd_name],
                                     precision, recall, F1]
results.to_csv(f'Results/Traffic_Flow_results.csv', index=False)
results = pd.read_csv('Results/Traffic_Flow_results.csv')
importlib.reload(sys.modules['utils_ECAD_journal'])
names = results[results.method.isin(['Ridge', 'RF', 'NN', 'RNN'])].method
names = ['EnbPI ' + name for name in names]
results.loc[results.method.isin(['Ridge', 'RF', 'NN', 'RNN']), 'method'] = names
importlib.reload(sys.modules['utils_ECAD_journal'])
utils_ECAD.plt_prec_recall_F1(results)


''' Table 2 in the appendix
    Reporting coverage, width, and Winkler score on all CA cities'''


def Latex_table_by_regr(array_ls, regr_name, methods_name, Data_names):
    # first two columns store string: dataname and method
    # last three columns store coverage, width, and Winkler score
    if '1d' in regr_name:
        array_t = array_ls
        if len(Data_names) > 0:
            np.savetxt(f"cov_wid_score_{regr_name}_solar.txt", array_t, fmt=(
                '%1.2f', '%1.2f', '%.2e'), delimiter=' & ', newline=' \\\\\n', comments='')
        else:
            np.savetxt(f"cov_wid_score_{regr_name}.txt", array_t, fmt=(
                '%1.2f', '%1.2f', '%.2e'), delimiter=' & ', newline=' \\\\\n', comments='')
    else:
        array_t = np.zeros(len(array_ls), dtype=(
            '<U50,<U30, float64, float64, float64'))
        multiplier = len(methods_name)
        for j in range(len(Data_names)):
            for k in range(multiplier):
                jk = j * multiplier + k
                remainder = np.mod(jk, multiplier)
                if remainder == 0:
                    if '1d' in regr_name:
                        array_t[jk] = ' ', ' ', array_ls[jk, 0], array_ls[jk,
                                                                          1], array_ls[jk, 2]
                    else:
                        name = '\multirow' + \
                            '{' + f'{multiplier}' + '}' + '{*}' + \
                            '{' + f'{Data_names[j]}' + '}'
                        array_t[jk] = name, methods_name[remainder], array_ls[jk,
                                                                              0], array_ls[jk, 1], array_ls[jk, 2]
                else:
                    if '1d' in regr_name:
                        array_t[jk] = ' ', ' ', array_ls[jk, 0], array_ls[jk,
                                                                          1], array_ls[jk, 2]
                    else:
                        array_t[jk] = ' ', methods_name[remainder], array_ls[jk,
                                                                             0], array_ls[jk, 1], array_ls[jk, 2]
        if len(Data_names) > 0:
            np.savetxt(f"cov_wid_score_{regr_name}_solar.txt", array_t, fmt=(
                '%s', '%s', '%1.2f', '%1.2f', '%.2e'), delimiter=' & ', newline=' \\\\\n', comments='')
        else:
            np.savetxt(f"cov_wid_score_{regr_name}.txt", array_t, fmt=(
                '%s', '%s', '%1.2f', '%1.2f', '%.2e'), delimiter=' & ', newline=' \\\\\n', comments='')


# Prep data
CA_cities = ['Fremont', 'Milpitas', 'Mountain_View', 'North_San_Jose',
             'Palo_Alto', 'Redwood_City', 'San_Mateo', 'Santa_Clara',
             'Sunnyvale']
for city in CA_cities:
    globals()['data%s' % city] = util.read_CA_data(f'Data/{city}_data.csv')
response_ls = {}
for city in CA_cities:
    response_ls[city] = 'DHI'
Data_name = CA_cities
# Prep regressor
min_alpha = 0.01
max_alpha = 10
ridge_cv = RidgeCV(alphas=np.linspace(min_alpha, max_alpha, 10))
random_forest = RandomForestRegressor(n_estimators=10, criterion='mse',
                                      bootstrap=False, max_depth=2, n_jobs=-1)
nnet = util.keras_mod()
regr_now_dict = {'Ridge': ridge_cv, 'RF': random_forest, 'NN': nnet}
regr_name = 'Ridge'
regr_now = regr_now_dict[regr_name]
# Prep other parameters
alpha = 0.05
stride = 1
B = 50  # number of bootstrap samples
miss_test_idx = []
importlib.reload(sys.modules['PI_class_EnbPI_journal'])
for one_dim in [True, False]:
    # For a particular regressor
    methods_name = ['Ensemble', 'ICP', 'WeightedICP',
                    'ARIMA', 'ExpSmoothing', 'DynamicFactor']
    methods_tseries = ['ARIMA', 'ExpSmoothing', 'DynamicFactor']
    # Each data is ran with all methods
    row_len = len(Data_name) * len(methods_name)
    col_len = 3  # coverage, width, Winkler score
    table_result = np.zeros((row_len, col_len))
    methods_CP = ['Ensemble', 'ICP', 'Weighted_ICP']
    itrial = 0
    k = 0
    for data_name in Data_name:
        np.random.seed(98765 + itrial)
        print(f'Trial # {itrial} for data {data_name}')
        data = eval(f'data{data_name}')  # Pandas DataFrame
        data_x = data.loc[:, data.columns != response_ls[data_name]]
        data_y = data[response_ls[data_name]]
        data_x_numpy = data_x.to_numpy()  # Convert to numpy
        data_y_numpy = data_y.to_numpy()  # Convert to numpy
        total_data_points = data_x_numpy.shape[0]
        train_size = int(0.2 * total_data_points)
        results = pd.DataFrame(columns=['itrial', 'dataname', 'muh_fun',
                                        'method', 'train_size', 'coverage', 'width'])
        CA_energy_data = True
        if CA_energy_data:
            X_train, X_predict, Y_train, Y_predict = big_transform(
                Data_name, data_name, one_dim, train_size)
        else:
            X_train = data_x_numpy[:train_size, :]
            X_predict = data_x_numpy[train_size:, :]
            Y_train = data_y_numpy[:train_size]
            Y_predict = data_y_numpy[train_size:]
            if one_dim:
                X_train, X_predict, Y_train, Y_predict = util.one_dimen_transform(
                    Y_train, Y_predict, d=20)
        tseries_results = EnbPI.prediction_interval(
            regr_now, X_train, X_predict, Y_train, Y_predict)
        PI_cov_wid_tseries = tseries_results.run_experiments(
            alpha, stride, data_name, itrial, get_plots=True, none_CP=True, methods=methods_tseries)
        tseries_scores = tseries_results.Winkler_score(
            PI_cov_wid_tseries[:-1], Data_name[k], methods_tseries, alpha)
        print(f'regressor is {regr_name}')
        CP_results = EnbPI.prediction_interval(
            regr_now, X_train, X_predict, Y_train, Y_predict)
        PI_cov_wid_CP = CP_results.run_experiments(
            alpha, stride, data_name, itrial, get_plots=True, methods=methods_CP)
        CP_scores = CP_results.Winkler_score(
            PI_cov_wid_CP[:-1], Data_name[k], methods_CP, alpha)
        # Store results for a particular dataset
        multiplier = len(methods_name)
        table_result[k * multiplier:(k + 1) * multiplier, 0] = np.append(PI_cov_wid_CP[-1]
                                                                         ['coverage'], PI_cov_wid_tseries[-1]['coverage'])
        table_result[k * multiplier:(k + 1) * multiplier, 1] = np.append(PI_cov_wid_CP[-1]
                                                                         ['width'], PI_cov_wid_tseries[-1]['width'])
        table_result[k * multiplier:(k + 1) * multiplier,
                     2] = np.append(CP_scores, tseries_scores)
        np.set_printoptions(precision=2)
        k += 1
    # Store all results in Latex Form
    if one_dim:
        regr_save = regr_name + '_1d'
    else:
        regr_save = regr_name
    Latex_table_by_regr(array_ls=table_result,
                        regr_name=regr_save, methods_name=methods_name, Data_names=Data_name)
