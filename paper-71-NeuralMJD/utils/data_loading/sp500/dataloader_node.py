import logging
import time
import numpy as np
import torch
import datetime
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Dict, Tuple


seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)


def stock_price_norm(data_dict: Dict, data_norm: str, stat_info: Dict) -> Dict:
    """
    Normalize the scale of the stock price data.

    Args:
        data_dict: Mapping ticker -> {'dates': [...], 'prices': [...]}.
        data_norm: One of {'max','avg','none','minmax'}.
        stat_info: Precomputed statistics per ticker.
    """
    for ticker, ticker_dict in tqdm(data_dict.items(), desc="Normalizing stock price data"):
        prices = ticker_dict['prices']

        if data_norm == 'minmax':
            price_norm_coef = np.max(prices) - np.min(prices)
            ticker_dict['PRICE_NORMALIZED'] = (prices - np.min(prices) / price_norm_coef).tolist()
            ticker_dict['PRICE_NORM_COEF'] = price_norm_coef
            ticker_dict['PRICE_NORM_MIN'] = np.min(prices)
        else:
            if ticker not in stat_info['price_by_ticker_avg']:
                # rare case of unseen category
                price_norm_coef = np.max(prices)
            else:
                if data_norm == 'max':
                    price_norm_coef = stat_info['price_by_ticker_max'][ticker]
                elif data_norm == 'avg':
                    price_norm_coef = stat_info['price_by_ticker_avg'][ticker]
                elif data_norm is None or data_norm == 'none':
                    price_norm_coef = 1.0
                else:
                    raise ValueError('data_norm must be one of [None, "max", "avg"]')
            
            price_norm_coef = np.clip(price_norm_coef, a_min=1e-6, a_max=None)
            ticker_dict['PRICE_NORMALIZED'] = (prices / price_norm_coef).tolist()

            ticker_dict['PRICE_NORM_COEF'] = price_norm_coef
    
    return data_dict


def stock_ticker_factorize(stat_info: Dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Factorize the ticker string into integers.
    """
    # init
    all_tickers = sorted(list(stat_info['price_by_ticker_avg'].keys()))
    ticker_to_int_dict = {ticker: i for i, ticker in enumerate(all_tickers)}
    int_to_ticker_dict = {i: ticker for i, ticker in enumerate(all_tickers)}

    return ticker_to_int_dict, int_to_ticker_dict


def stock_create_dynamic_data(data_dict: Dict, ticker_to_int_dict: Dict[str, int], seqlen: int, predlen: int, overlap: int):
    """
    Split stock price data into sliding windows by creating meta data info, and build dynamic/static arrays.
    """
    meta_data_sliding_window = {}
    data_dict_array = {}

    # Precompute weekday mapping for efficiency
    date_to_weekday = {}
    unique_dates = set(sum([ticker_dict['dates'] for ticker_dict in data_dict.values()], []))
    for date in unique_dates:
        # monday is 0 and sunday is 6
        year, month, day = date.split('-')
        date_to_weekday[(int(year), int(month), int(day))] = datetime.date(int(year), int(month), int(day)).weekday()

    num_days = len(date_to_weekday)
    num_windows = (num_days - seqlen) // (seqlen - overlap) + 1

    # extend the data to avoid completely empty sliding windows
    if num_windows == 0:
        for ticker, ticker_dict in tqdm(data_dict.items(), desc="Extending stock price data"):
            ticker_dict['dates'].append(ticker_dict['dates'][-1])
            ticker_dict['prices'].append(ticker_dict['prices'][-1])
            ticker_dict['PRICE_NORMALIZED'].append(ticker_dict['PRICE_NORMALIZED'][-1])

        num_windows = (num_days + 1 - seqlen) // (seqlen - overlap) + 1

    # Remove the largest 3% and smallest 3% tickers for numerical range stability
    # mean_stock_data = {ticker: np.mean(val['prices']) for ticker, val in data_dict.items()}
    # top_percentile = np.percentile(list(mean_stock_data.values()), 97)
    # bottom_percentile = np.percentile(list(mean_stock_data.values()), 3)
    # ticker_outlier_ls = [key for key, val in mean_stock_data.items() if val > top_percentile or val < bottom_percentile]
    # mean_stock_data_new = {ticker: np.mean(val['prices']) for ticker, val in data_dict.items() if ticker not in ticker_outlier_ls}
    # logging.info("After removing outliers top/bottom 3% spending data, {:d} / {:d} ({:.2f}%) tickers would be retained.".format(
    #     len(mean_stock_data) - len(ticker_outlier_ls), len(mean_stock_data), (len(mean_stock_data) - len(ticker_outlier_ls)) / len(mean_stock_data) * 100))
    # logging.info("Change of mean spending percentile statistics:")
    # for p in [1, 5, 25, 50, 75, 90, 95, 99]:
    #     logging.info("{:02d}% {:.2f} ---> {:.2f}".format(p,  np.percentile(list(mean_stock_data.values()), p), np.percentile(list(mean_stock_data_new.values()), p)))

    ticker_outlier_ls = []

    for ticker, ticker_dict in tqdm(data_dict.items(), desc="Augmenting stock price dynamic data"):
        """Stock price sequential data processing"""
        if ticker in ticker_outlier_ls:
            continue

        # get a fixed length of sequence considering the overlap length
        weekday_whole_period = []
        for date in ticker_dict['dates']:
            year, month, day = date.split('-')
            weekday = date_to_weekday[(int(year), int(month), int(day))]
            weekday_whole_period.append(weekday)

        ticker_dict['WEEKDAY'] = weekday_whole_period

        for i in range(num_windows):
            start_idx = i * (seqlen - overlap)
            end_idx = i * (seqlen - overlap) + seqlen  # exclusive

            if end_idx > len(ticker_dict['WEEKDAY']):
                continue

            # only record meta data for the sliding window
            meta_data_sliding_window[ticker + '_{:03d}'.format(i)] = {
                    'idx': [start_idx, end_idx],
                }
            
        """create dynamic feature array"""
        # Process dynamic data in NumPy with vectorized operations
        price_norm   = ticker_dict['PRICE_NORMALIZED']
        price_raw    = ticker_dict['prices']

        # Normalize weekday immediately
        weekday      = np.asarray(ticker_dict['WEEKDAY'], dtype=np.float32) / 6.0

        # Stack the arrays along a new first axis to get shape [5, sequence_length]
        stock_dyn_data_np = np.stack([price_norm, weekday, price_raw], axis=0, dtype=np.float32)  # [3, T]

        # Static feature: ticker name refactored into integer
        if ticker in ticker_to_int_dict:
            stock_stc_data_np = np.asarray(ticker_to_int_dict[ticker], dtype=np.float32)  # [1]
        else:
            stock_stc_data_np = np.array([-1], dtype=np.float32)
        
        data_dict_array[ticker] = {
            'dynamic': stock_dyn_data_np,
            'static': stock_stc_data_np
        }
    
    return meta_data_sliding_window, data_dict_array


class NodeDataset(Dataset):
    """
    Node-level PyG dataset for SP500 stocks (not used by default training loop).
    """

    def __init__(self, data_dict, stat_info, overlap=0, seqlen=7, predlen=1, mode='train', 
                 logdir=None, data_norm='max'):
        """
        Args:
            data_dict: keys=tickers, values=dict of sequences and meta.
            stat_info: dataset-level statistics.
            overlap: number of days overlapped between windows.
            seqlen: total sequence length.
            predlen: horizon length to predict.
            mode: one of {'train','val','test'}.
            data_norm: normalization method.
        """

        """init"""
        self.data_dict = data_dict
        self.stat_info = stat_info
        self.overlap = overlap
        self.seqlen = seqlen
        self.predlen = predlen
        self.mode = mode
        self.logdir = logdir

        self.data_norm = data_norm

        assert self.seqlen > self.predlen, "Sequence length must be larger than prediction length."
        assert self.seqlen > self.overlap, "Sequence length must be larger than overlap."

        logging.info("Mode: {}. Start to initialize node-level data set object.".format(self.mode))
        time_start = time.time()

        """data normalization"""
        self.data_dict = stock_price_norm(self.data_dict, self.data_norm, self.stat_info)

        """ticker factorization"""
        self.ticker_to_int_dict, self.int_to_ticker_dict = stock_ticker_factorize(self.stat_info)

        """apply sliding window to the dynamic data"""
        self.meta_data_sliding_window, self.data_dict_array = stock_create_dynamic_data(
            self.data_dict, self.ticker_to_int_dict, self.seqlen, self.predlen, self.overlap)
        self.window_keys = list(self.meta_data_sliding_window.keys())        

        time_elapsed = time.time() - time_start
        logging.info("Mode: {}. After sliding window, the number of data points: {:d} ---> {:d}. Time spent: {:.2f}".format(
            self.mode, len(self.data_dict), len(self.meta_data_sliding_window), time_elapsed))
        
    def __len__(self):
        return len(self.meta_data_sliding_window)
    
    def _create_node_data_obj(self, idx) -> Data:
        """Create a PyG `Data` object for a single node sample."""
        # Init the node IDs
        stock_key = self.window_keys[idx]

        # Get stock data
        # Get slicing indices and the corresponding dictionary once
        idx_start, idx_end = self.meta_data_sliding_window[stock_key]['idx']
        ticker = stock_key[:-4]
        past_slice   = slice(None, -self.predlen)
        future_slice = slice(-self.predlen, None)

        stock_dyn_data_np = self.data_dict_array[ticker]['dynamic']      # [3, T]
        # the dimensions are price_norm, weekday, price_raw

        stock_dyn_data_tensor = torch.from_numpy(stock_dyn_data_np[None, :, idx_start:idx_end])
        stock_dyn_data_past   = stock_dyn_data_tensor[:, :2, past_slice]
        stock_dyn_data_future = stock_dyn_data_tensor[:, :2, future_slice]

        price_raw = stock_dyn_data_tensor[:, 2]

        stock_stc_data_np = self.data_dict_array[ticker]['static']                  # [1]
        stock_stc_data_tensor = torch.from_numpy(stock_stc_data_np).view(1, -1)      # [1, 1]

        d = self.data_dict[ticker]
        price_norm_coef = torch.tensor(d['PRICE_NORM_COEF'], dtype=torch.float32)

        """Build node-level data item"""
        node_attr = {
            # Input features
            "dyn_feat":           stock_dyn_data_past,              # [1, 2, Past]
            "static_feat":        stock_stc_data_tensor,            # [1, 1]
            "price_past_raw":     price_raw[..., past_slice],       # [1, Past]
            "price_norm_coef":    price_norm_coef,                  # scalar
            "key_ego":            stock_key,                             # string
            "num_nodes":          1,                                # scalar    

            # Ego node future stock price data
            "price_future_norm":  stock_dyn_data_future[:, 0],       # [1, Future]
            "price_future_raw":   price_raw[..., future_slice],      # [1, Future]
        }

        if self.data_norm == 'minmax':
            node_attr['price_norm_min'] = torch.tensor(d['PRICE_NORM_MIN'], dtype=torch.float32)

        # create the PyTorch Geometric Data object using dictionary unpacking.
        node_data = Data(**node_attr)

        return node_data

    def __getitem__(self, idx):
        return self._create_node_data_obj(idx)
    