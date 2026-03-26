from calendar import monthrange
import logging
import os
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from .dataloader_node import NodeDataset
from .dataloader_graph import GraphDataset

PROJ_DIR = os.path.abspath(os.path.join(__file__, "../../../.."))


def slice_dict(in_dict, n_item):
    return dict(list(in_dict.items())[:n_item])


def load_sp500_raw_data(data_months, n_slice=None) -> Tuple[Dict, Dict]:
    """
    Load SP500 raw data from CSV, optionally cache into pickles for speed.

    Args:
        data_months: Single YYYY/MM or a list of YYYY/MM months to include.
        n_slice: If provided, keep only the first n tickers for faster iteration.

    Returns:
        (sp500_data_dict, sp500_stat_info_dict)
    """
    logging.info("Loading raw data for {}...".format(data_months))
    time_start = time.time()
    data_months = [data_months] if isinstance(data_months, str) else data_months

    monthly_days = []
    for yymm in data_months:
        assert '/' in yymm, "Invalid month format. Use 'YYYY/MM'."
        year, month = map(int, yymm.split('/'))
        monthly_days.append(monthrange(year, month)[1])

    selected_sp500_data_path = os.path.join(PROJ_DIR, 'data', 'merged', 'sp500_dict_' + '_'.join(data_months).replace('/', '_') + '.pkl')
    selected_sp500_stat_info_path = os.path.join(PROJ_DIR, 'data', 'merged', 'sp500_stat_info_' + '_'.join(data_months).replace('/', '_') + '.pkl')

    """Node data loading"""
    if os.path.exists(selected_sp500_data_path) and os.path.exists(selected_sp500_stat_info_path):
        logging.info("Loading merged data from {}...".format(selected_sp500_data_path))
        sp500_data_dict = pickle.load(open(selected_sp500_data_path, 'rb'))
        sp500_stat_info_dict = pickle.load(open(selected_sp500_stat_info_path, 'rb'))
    else:
        logging.info("Merged node-level data file {} does not exist. Loading raw data from individual files...".format(selected_sp500_data_path))

        sp500_stocks_df = pd.read_csv(os.path.join(PROJ_DIR, 'data', 'sandp500/all_stocks_5yr.csv'))

        # filter by time range
        sp500_dates = pd.to_datetime(sp500_stocks_df['date'])
        start_date = pd.to_datetime(data_months[0] + '/01')
        end_date = pd.to_datetime(data_months[-1] + '/01') + pd.offsets.MonthEnd(0)
        flag_sp500_dates = sp500_dates.between(start_date, end_date)
        sp500_stocks_df = sp500_stocks_df[flag_sp500_dates]

        # extract stock price data for each ticker
        unique_tickers = np.unique(sp500_stocks_df['Name'])
        logging.info("Number of unique SP500 stocks between {:} and {:}: {:d}".format(start_date, end_date, len(unique_tickers)))

        sp500_data_dict = {}
        for ticker in unique_tickers:
            cur_ticker_data = sp500_stocks_df[sp500_stocks_df['Name'] == ticker]
            sp500_data_dict[ticker] = {
                'dates': cur_ticker_data['date'].tolist(),
                'prices': cur_ticker_data['close'].tolist()
            }

        # Count statistics for stock data by ticker
        price_by_ticker_avg = {key: np.mean(list(sp500_data_dict[key]['prices'])) for key in sp500_data_dict.keys()}
        price_by_ticker_max = {key: np.max(list(sp500_data_dict[key]['prices'])) for key in sp500_data_dict.keys()}
        price_by_ticker_min = {key: np.min(list(sp500_data_dict[key]['prices'])) for key in sp500_data_dict.keys()}
        price_by_ticker_zero_rate = {key: np.sum(list(sp500_data_dict[key]['prices']) == 0) / np.prod(np.array(list(sp500_data_dict[key]['prices'])).shape) for key in sp500_data_dict.keys()}
        price_by_ticker_max_to_avg_ratio = {key: (np.max(list(sp500_data_dict[key]['prices'])) / np.mean(list(sp500_data_dict[key]['prices']))) for key in sp500_data_dict.keys()}

        sp500_stat_info_dict = {
            'price_by_ticker_avg': price_by_ticker_avg,
            'price_by_ticker_max': price_by_ticker_max,
            'price_by_ticker_min': price_by_ticker_min,
            'price_by_ticker_zero_rate': price_by_ticker_zero_rate,
            'price_by_ticker_max_to_avg_ratio': price_by_ticker_max_to_avg_ratio,
        }


    """save to pickle data"""
    if not os.path.exists(selected_sp500_data_path) or not os.path.exists(selected_sp500_stat_info_path):
        logging.info("Saving SP500 data to {}...".format(selected_sp500_data_path))
        os.makedirs(os.path.dirname(selected_sp500_data_path), exist_ok=True)
        pickle.dump(sp500_data_dict, open(selected_sp500_data_path, 'wb'))

        logging.info("Saving statistics data to {}...".format(selected_sp500_stat_info_path))
        os.makedirs(os.path.dirname(selected_sp500_stat_info_path), exist_ok=True)
        pickle.dump(sp500_stat_info_dict, open(selected_sp500_stat_info_path, 'wb'))

    time_elapsed = time.time() - time_start
    logging.info("Loading finsihed. Number of tickerss: {:d}. Time spent: {:.2f}".format(len(sp500_data_dict), time_elapsed))

    if n_slice is not None:
        sp500_data_dict = slice_dict(sp500_data_dict, n_slice)
        logging.info("Slicing data dict to {:d} tickers".format(n_slice))

    # data scaling for training stability
    scaling_factor = 10.0
    if scaling_factor != 1.0:
        for ticker in sp500_data_dict.keys():
            sp500_data_dict[ticker]['prices'] = (np.array(sp500_data_dict[ticker]['prices']) * scaling_factor).tolist()
        sp500_stat_info_dict['price_by_ticker_avg'] = {ticker: val * scaling_factor for ticker, val in sp500_stat_info_dict['price_by_ticker_avg'].items()}
        sp500_stat_info_dict['price_by_ticker_max'] = {ticker: val * scaling_factor for ticker, val in sp500_stat_info_dict['price_by_ticker_max'].items()}
        sp500_stat_info_dict['price_by_ticker_min'] = {ticker: val * scaling_factor for ticker, val in sp500_stat_info_dict['price_by_ticker_min'].items()}

    return sp500_data_dict, sp500_stat_info_dict
    


def get_sp500_dataset(config, eval_mode):
    """
    Build PyTorch dataset objects for SP500 data, using graph-level representation by default.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """

    training_months, validation_months, testing_months = config.dataset.training, config.dataset.validation, config.dataset.testing  # list of months, e.g., 2023/02
    training_subset, validation_subset, testing_subset = config.dataset.subset, config.dataset.subset_val, config.dataset.subset_test

    # load raw data
    if eval_mode:
        _, training_stat_info_dict = load_sp500_raw_data(training_months, training_subset)
        validation_data_dict, _ = load_sp500_raw_data(validation_months, validation_subset)
        testing_data_dict, _ = load_sp500_raw_data(testing_months, testing_subset)

        training_data_dict = testing_data_dict  # not using the real training data
    else:
        training_data_dict, training_stat_info_dict = load_sp500_raw_data(training_months, training_subset)
    
    if config.dataset.overfit:
        validation_data_dict = testing_data_dict = training_data_dict
    else:
        validation_data_dict, _ = load_sp500_raw_data(validation_months, validation_subset)
        testing_data_dict, _ = load_sp500_raw_data(testing_months, testing_subset)

    # build dataset object
    train_dataset = GraphDataset(training_data_dict, training_stat_info_dict, 
                                config.dataset.overlap, config.dataset.seqlen, config.dataset.predlen, 
                                mode='train', logdir=config.logdir, data_norm=config.dataset.data_norm)
        
    val_dataset = GraphDataset(validation_data_dict, training_stat_info_dict, 
                                config.dataset.overlap, config.dataset.seqlen, config.dataset.predlen, 
                                mode='val', logdir=config.logdir, data_norm=config.dataset.data_norm)
    
    test_dataset = GraphDataset(testing_data_dict, training_stat_info_dict, 
                                config.dataset.overlap, config.dataset.seqlen, config.dataset.predlen, 
                                mode='test', logdir=config.logdir, data_norm=config.dataset.data_norm)
    return train_dataset, val_dataset, test_dataset

