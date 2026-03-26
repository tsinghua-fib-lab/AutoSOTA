import logging
import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Tuple

from .dataloader_node import stock_price_norm, stock_ticker_factorize, stock_create_dynamic_data


PROJ_DIR = os.path.abspath(os.path.join(__file__, "../.."))

seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)


class GraphDataset(Dataset):
    """SP500 graph-level dataset for ego-graph forecasting."""

    def __init__(self, data_dict, stat_info, overlap=0, seqlen=7, predlen=1, mode='train', 
                       logdir=None, data_norm='max'):
        """
        Args:
            data_dict: keys are stock tickers, values are dicts with raw/normalized sequences.
            stat_info: global statistics dict computed from the raw data.
            overlap: number of overlapped days between windows.
            seqlen: total sequence length (past + future).
            predlen: prediction horizon (future days), Past = seqlen - predlen.
            mode: one of {'train','val','test'}.
            logdir: directory to save logs/plots.
            data_norm: normalization method {'max','avg','sqrt','none','minmax'}.
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

        logging.info("Mode: {}. Start to initialize graph-level data set object.".format(self.mode))
        time_start = time.time()

        """data normalization"""
        self.data_dict = stock_price_norm(self.data_dict, self.data_norm, self.stat_info)

        """ticker factorization"""
        self.ticker_to_int_dict, self.int_to_ticker_dict = stock_ticker_factorize(self.stat_info)

        """apply sliding window to the dynamic data"""
        self.meta_data_sliding_window, self.data_dict_array = stock_create_dynamic_data(
            self.data_dict, self.ticker_to_int_dict, self.seqlen, self.predlen, self.overlap)
        self.window_keys = list(self.meta_data_sliding_window.keys())   

        """create graph data"""
        self.unique_idx_ls = sorted(np.unique([item[-3:] for item in self.window_keys]))
        # self.unique_idx_ls = self.window_keys
        # self.data_item_ls = []
        # for idx in tqdm(range(len(self.unique_idx_ls)), desc='Creating graph data...'):
        #     self.data_item_ls.append(self._create_graph_data_obj(idx))

        time_elapsed = time.time() - time_start
        logging.info("Mode: {}. After sliding window, the number of graph data points: {:d} ---> {:d}. Time spent: {:.2f}".format(
            self.mode, len(self.data_dict), len(self.unique_idx_ls), time_elapsed))

    def __len__(self):
        return len(self.unique_idx_ls)

    def _create_graph_data_obj(self, idx) -> Data:
        """Create a PyG `Data` object for a single ego-graph index."""
        # Init the stock ticker IDs
        idx_nm = self.unique_idx_ls[idx]
        graph_stock_keys = sorted([key for key in self.window_keys if key[-3:] == idx_nm])

        # Get graph-level data
        # Get slicing indices and the corresponding dictionary once
        idx_start, idx_end = self.meta_data_sliding_window[graph_stock_keys[0]]['idx']
        past_slice   = slice(None, -self.predlen)
        future_slice = slice(-self.predlen, None)

        graph_dyn_data_np = []
        graph_stc_data_np = []
        for stock_key in graph_stock_keys:
            # the dimensions are price_norm, weekday, price_raw
            graph_dyn_data_np.append(self.data_dict_array[stock_key[:-4]]['dynamic'][None, :, idx_start:idx_end])      # [1, 3, T]

            graph_stc_data_np.append(self.data_dict_array[stock_key[:-4]]['static'].reshape(1, 1))    # [1]

        graph_dyn_data_np = np.concatenate(graph_dyn_data_np, axis=0).astype(np.float32)    # [N, 3, T]
        graph_stc_data_np = np.concatenate(graph_stc_data_np, axis=0).astype(np.float32)    # [N, 1]

        graph_dyn_data_tensor = torch.from_numpy(graph_dyn_data_np)                 # [N, 3, T]
        graph_dyn_data_past   = graph_dyn_data_tensor[:, :2, past_slice]            # [N, 2, Past]
        graph_dyn_data_future = graph_dyn_data_tensor[:, :2, future_slice]          # [N, 2, Future]

        price_raw = graph_dyn_data_tensor[:, 2]                                     # [N, T]

        graph_stc_data_tensor = torch.from_numpy(graph_stc_data_np)                 # [N, 1]

        price_norm_coef = [self.data_dict[stock_ticker[:-4]]['PRICE_NORM_COEF'] for stock_ticker in graph_stock_keys]
        price_norm_coef = torch.tensor(price_norm_coef, dtype=torch.float32).view(-1, 1)    # [N, 1]

        if self.data_norm == 'minmax':
            price_norm_min = [self.data_dict[stock_ticker[:-4]]['PRICE_NORM_MIN'] for stock_ticker in graph_stock_keys]
            price_norm_min = torch.tensor(price_norm_min, dtype=torch.float32).view(-1, 1)  # [N, 1]

        # Get ego-graph edge index and feature
        num_nodes = len(graph_stock_keys)

        # Define fully connected graphs
        num_edges = num_nodes * num_nodes  # Total number of edges
        edges_idx = torch.empty((2, num_edges), dtype=torch.long)
        edges_idx[0, :] = torch.arange(num_nodes).repeat_interleave(num_nodes)
        edges_idx[1, :] = torch.arange(num_nodes).repeat(num_nodes)
        
        # Alternative: naive self-loop graphs (example)
        # num_edges = num_nodes
        # edges_idx = torch.empty((2, num_edges), dtype=torch.long)
        # edges_idx[0, :] = torch.arange(num_nodes)
        # edges_idx[1, :] = torch.arange(num_nodes)

        # Alternative: empty graph (example)
        # num_edges = 0
        # edges_idx = torch.empty((2, num_edges), dtype=torch.long)
        # edges_idx[0, :] = torch.arange(num_edges)
        # edges_idx[1, :] = torch.arange(num_edges)

        # Create edge attributes by checking if the edge is a self-loop
        mask_self_loop = edges_idx[0, :] == edges_idx[1, :]
        edges_attr = torch.empty((num_edges, 1), dtype=torch.float)
        edges_attr[mask_self_loop, 0] = 0.0
        edges_attr[~mask_self_loop, 0] = 1.0

        # Create the graph data
        graph_attr = {
                        ### Input features ###
                        "dyn_feat": graph_dyn_data_past,                           # [N, 2, Past]
                        "static_feat": graph_stc_data_tensor,                      # [N, 1]
                        "edge_index": edges_idx,                                   # [2, E], E=N*N
                        "edge_attr": edges_attr,                                   # [E, 1]
                        "price_past_raw": price_raw[:, :-self.predlen],            # [N, Past]
                        "price_norm_coef": price_norm_coef,                        # [N, 1]
                        "key_ego": graph_stock_keys,                               # [N], a list of strings
                        ### Input features ###

                        ### Ego node future spending and visit data ###
                        "price_future_norm": graph_dyn_data_future[:, 0, :],       # [N, Future]  
                        "price_future_raw": price_raw[:, -self.predlen:],          # [N, Future]
                        ### Ego node future spending and visit data ###

                        "num_nodes": num_nodes,                                    # scalar
        }

        if self.data_norm == 'minmax':
            graph_attr['price_norm_min'] = price_norm_min

        graph_data = Data(**graph_attr)

        return graph_data

    def __getitem__(self, idx):
        # return self.data_item_ls[idx]
        return self._create_graph_data_obj(idx)

