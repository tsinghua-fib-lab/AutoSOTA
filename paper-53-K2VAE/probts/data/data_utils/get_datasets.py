# ---------------------------------------------------------------------------------
# Portions of this file are derived from Autoformer
# - Source: https://github.com/thuml/Autoformer/tree/main
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import os
import pandas as pd
from probts.data.data_utils.time_features import time_features
from probts.data.data_utils.data_utils import convert_monash_data_to_dataframe, monash_format_convert
import numpy as np


def get_dataset_info(dataset, data_path=None, freq=None):
    """
    Get the file path and frequency associated with the specified dataset.
    Parameters:
        dataset (str): The name of the dataset.
        data_path (str): Optional custom data path for the dataset.
        freq (str): Optional custom frequency for the dataset.
    Returns:
        tuple: A tuple containing the data path and frequency.
    """
    paths = {
        'etth1': ('ETT-small/ETTh1.csv', 'H'),
        'etth2': ('ETT-small/ETTh2.csv', 'H'),
        'ettm1': ('ETT-small/ETTm1.csv', 'min'),
        'ettm2': ('ETT-small/ETTm2.csv', 'min'),
        'traffic_ltsf': ('traffic/traffic.csv', 'H'),
        'electricity_ltsf': ('electricity/electricity.csv', 'H'),
        'exchange_ltsf': ('exchange_rate/exchange_rate.csv', 'B'),
        'illness_ltsf': ('illness/national_illness.csv', 'W'),
        'weather_ltsf': ('weather/weather.csv', 'min'),
        'caiso': ('caiso/caiso_20130101_20210630.csv', 'H'),
        'nordpool': ('nordpool/production.csv', 'H'),
        'turkey_power': ('kaggle/power Generation and consumption.csv', 'H'),
        'istanbul_traffic': ('kaggle/istanbul_traffic.csv', 'H')
    }
    
    if dataset in paths:
        data_path, freq = paths[dataset]
    else:
        assert data_path is not None, f'Invalid dataset name: {dataset}! Provide --data.data_manager.init_args.data_path for custom datasets.'
        assert freq is not None, 'Provide --data.data_manager.init_args.freq for custom datasets.'
    return data_path, freq

def get_dataset_borders(dataset, data_size, train_ratio=0.7, test_ratio=0.2):
    """
    Compute the start and end indices for train, validation, and test splits.
    Parameters:
        dataset (str): The name of the dataset.
        data_size (int): Total number of time points in the dataset.
        train_ratio (float): Proportion of the dataset used for training.
        test_ratio (float): Proportion of the dataset used for testing.
    Returns:
        tuple: Two lists representing the start and end indices of each split.
    """
    # Validate ratios
    assert 0 < train_ratio <= 1, "train_ratio must be between 0 and 1 (exclusive of 0)."
    assert 0 < test_ratio <= 1, "test_ratio must be between 0 and 1 (exclusive of 0)."
    assert train_ratio + test_ratio <= 1, "The sum of train_ratio and test_ratio must not exceed 1."

    # Predefined borders for ETT datasets
    if dataset == 'etth1' or dataset == 'etth2':
        border_begin = [0, 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24]
        border_end = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif dataset == 'ettm1' or dataset == 'ettm2':
        border_begin = [0, 12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4]
        border_end = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        # Calculate borders for custom datasets
        num_train = int(data_size * train_ratio)
        num_test = int(data_size * test_ratio)
        num_vali = data_size - num_train - num_test
        border_begin = [0, num_train, data_size - num_test]
        border_end = [num_train, num_train + num_vali, data_size]
    return border_begin, border_end

def load_dataset(root_path, data_path,freq='h', timeenc=1, multivariate=True):
    """
    Load and process datasets.
    Parameters:
        root_path (str): Root directory for datasets.
        data_path (str): Path to the specific dataset.
        freq (str): Frequency of the dataset (e.g., 'H', 'min').
        timeenc (int): Time encoding method (0 for temporal information, 1 for time feature based on frequency, 2 for raw date information).
        multivariate (bool): Whether the dataset is multivariate.
    Returns:
        df_raw: the processed DataFrame
        data_stamp: time features
        target_dim: target dimensions
        data_size: total length of timestamps.
    """
    data_format = None
    if '.tsf' in data_path:
        # Load Monash time series dataset
        df_raw, _, _, _, _ = convert_monash_data_to_dataframe(data_path)
        df_raw = monash_format_convert(df_raw, freq, multivariate)
        
        if multivariate:
            if freq.lower() == 'h':
                df_raw.set_index('date', inplace=True)
                df_raw = df_raw.resample(freq).mean().reset_index()
    elif 'caiso' in data_path:
        # Load and process CAISO dataset
        data = pd.read_csv(os.path.join(root_path, data_path))
        data['Date'] = data['Date'].astype('datetime64[ns]')
        names = ['PGE','SCE','SDGE','VEA','CA ISO','PACE','PACW','NEVP','AZPS','PSEI']
        df_raw = pd.DataFrame(pd.date_range('20130101','20210630',freq='H')[:-1], columns=['Date'])
        for name in names:
            current_df = data[data['zone'] == name].drop_duplicates(subset='Date', keep='last').rename(columns={'load':name}).drop(columns=['zone'])
            df_raw = df_raw.merge(current_df, on='Date', how='outer')
        df_raw = df_raw.rename(columns={'Date': 'date'})
    elif 'nordpool' in data_path:
        # Load and process Nordpool dataset
        df_raw = pd.read_csv(os.path.join(root_path, data_path), parse_dates=['Time'])
        df_raw = df_raw.rename(columns={'Time': 'date'})
    elif 'power Generation and consumption' in data_path:
        # Load and process Turkey Power dataset
        df_raw = pd.read_csv(os.path.join(root_path, data_path), parse_dates=['Date_Time'])
        df_raw = df_raw.rename(columns={'Date_Time': 'date'})
        data_format = "%d.%m.%Y %H:%M"
    elif 'istanbul_traffic' in data_path:
        # Load and process Istanbul Traffic dataset
        df_raw = pd.read_csv(os.path.join(root_path, data_path), parse_dates=['datetime'])
        df_raw = df_raw.rename(columns={'datetime': 'date'})
        df_raw.set_index('date', inplace=True)
        df_raw = df_raw.resample(freq).mean().reset_index()
    else:
        # Load customized dataset
        df_raw = pd.read_csv(os.path.join(root_path, data_path), parse_dates=['date'])
    
    # Process time encoding
    if multivariate:
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date, format=data_format)
        
        if timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(labels='date', axis=1).values
        elif timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=freq)
            data_stamp = data_stamp.transpose(1, 0)
        elif timeenc == 2:
            data_stamp = pd.to_datetime(df_stamp['date'].values)
            data_stamp = np.array(data_stamp, dtype='datetime64[s]')
        else:
            raise ValueError('Invalid timeenc value. timeenc should be sellected within [0, 1, 2].')
        df_raw = df_raw.set_index(keys='date')
        
    else:
        data_stamp = None
    
    # Replace missing values with 0
    df_raw = df_raw.fillna(0)
    # Determine target dimension and dataset size
    target_dim = len(df_raw.columns) if multivariate else 1
    data_size = len(df_raw)
    return df_raw, data_stamp, target_dim, data_size