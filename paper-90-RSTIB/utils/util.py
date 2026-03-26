import os
import os.path as osp
import random
import logging
import sys
from scipy.sparse.linalg import eigs
from tqdm import tqdm
import numpy as np
import torch
import math
import torch.nn.functional as F
import datetime
from componenets.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
import copy
from .adjacent_matrix_norm import *
from geopy.distance import geodesic
# from Params import args

def normalize_dataset(data, normalizer, column_wise=False):
    # data : t, n, 5
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=(0, 1), keepdims=True)
            maximum = data.max(axis=(0, 1), keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean =  data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        # column min max, to be depressed
        # note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler


def load_st_dataset(dataset):
    # output B, N, D
    if dataset == 'PEMS4':
        data_path = os.path.join('./data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]
    elif dataset == 'PEMS8':
        data_path = os.path.join('./data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]
    elif dataset == 'PEMS3':
        data_path = os.path.join('../data/PEMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]
    elif dataset == 'PEMS7':
        data_path = os.path.join('../data/PEMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data


def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []  # windows
    Y = []  # horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index + window])
            Y.append(data[index + window + horizon - 1:index + window + horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index + window])
            Y.append(data[index + window:index + window + horizon])
            index = index + 1
    # X = np.array(X)
    # Y = np.array(Y)
    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    return X, Y

def Add_Window_Horizon_time(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []  # windows
    Y = []  # horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index + window])
            Y.append(data[index + window + horizon - 1:index + window + horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index + window])
            Y.append(data[index + window:index + window + horizon])
            index = index + 1
    
    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    return X, Y


def split_data_by_ratio(data, val_ratio, test_ratio, shffule = False):
    data_len = len(data)
    test_data = data[-int(data_len * test_ratio):]
    if shffule:
        tra_val_data = data[:-int(data_len * test_ratio)]
        shuffle_idx = np.random.permutation(len(tra_val_data))
        tra_val_data = tra_val_data[shuffle_idx]
        val_data = tra_val_data[-int(data_len * val_ratio):]
        train_data = tra_val_data[:-int(data_len * val_ratio)]
    else:
        val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
        train_data = data[:-int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data

def split_data_by_day(data, val_day, test_day):
    test_data = data[-test_day:]
    val_data = data[-(test_day+val_day):-test_day]
    train_data = data[:-(test_day+val_day)]
    return train_data, val_data, test_data

def data_loader(X, Y, TE, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    print('cuda :{}'.format(cuda))
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y, TE= TensorFloat(X), TensorFloat(Y), TensorFloat(TE)
    
    data = torch.utils.data.TensorDataset(X, Y, TE)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


def print_model_parameters(model, only_num=True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')


def get_adjacency_matrix(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None, args = None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    type_: str, {connectivity, distance}
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename != 'None':
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A

def get_adjacency_binary(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None, self_loop = False, args = None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    type_: str, {connectivity, distance}
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename != 'None':
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                if type_ == 'connectivity':
                    if distance < args.dist_thr:
                        A[id_dict[i], id_dict[j]] = 1
                        A[id_dict[j], id_dict[i]] = 1
                elif type_ == 'distance':
                    if distance < args.dist_thr:
                        if distance == 0.0:
                            A[id_dict[i], id_dict[j]] = 1
                            A[id_dict[j], id_dict[i]] = 1
                        else:
                            A[id_dict[i], id_dict[j]] = 1 / distance
                            A[id_dict[j], id_dict[i]] = 1 / distance
                else:
                    raise ValueError("type_ error, must be "
                                    "connectivity or distance!")
            print('edge num: {}'.format(np.count_nonzero(A)))
            if self_loop is True:
                A = A + np.identity(num_of_vertices)
            print('edge num: {}'.format(np.count_nonzero(A)))
        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
        print('edge num: {}'.format(np.count_nonzero(A)))
        if self_loop is True:
            A = A + np.identity(num_of_vertices)
        
    return A

def is_sym(adj): 
    return np.allclose(adj, adj.T)
    

def pair_dist_to_adj(adj_dist, num_of_vertices,
                         type_='connectivity', self_loop = False, args = None):
    '''
    Parameters
    ----------
    adj_dist: (N x N)
    num_of_vertices: int, the number of vertices
    type_: str, {connectivity, distance}
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''
    
    A = copy.deepcopy(adj_dist)
    if type_ in ['connectivity', 'distance_2']:
        trans_A = np.exp(- A ** 2 / (args.sigma ** 2))
    elif type_ == 'distance':
        trans_A = A
    else:
        raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    

    # Fills cells in the matrix with distances.
    if type_ == 'connectivity':
        trans_A[trans_A >= args.adj_th] = 1
        trans_A[trans_A < args.adj_th] = 0
    elif type_ == 'distance_2':
        trans_A[trans_A < args.adj_th] = 0
    elif type_ == 'distance':
        non_zero_indices = np.nonzero(trans_A)
        trans_A[non_zero_indices] = 1 / trans_A[non_zero_indices]
        trans_A[trans_A == 0] = 1
    else:
        raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    print('{} Adj edge num: {}'.format(type_, np.count_nonzero(trans_A)))
    assert is_sym(trans_A), 'the adj is not symmetric.'
    if self_loop is True:
        np.fill_diagonal(trans_A, 1)
        
    return trans_A


def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger

def get_geo_matrix(args):
    geo_graph_raw = torch.load(args.geo_graph)
    # 'edge_index', 'edge_weight', 'node_name'
    edge_index = geo_graph_raw['edge_index'].transpose(1, 0)
    edge_weight = geo_graph_raw['edge_weight']
    geo_graph = torch.zeros([args.num_nodes, args.num_nodes])
    for idx in tqdm(range(len(edge_index))):
        pair = edge_index[idx]
        geo_graph[pair[0], pair[1]] = edge_weight[idx]
        geo_graph[pair[1], pair[0]] = edge_weight[idx]

    return geo_graph

def load_SE(num_node, d_model):
    # SE = torch.zeros([num_node, num_node])
    # for ind in num_node:
    #     SE[ind, ind] = 1
    # return SE
    pe = torch.zeros(num_node, d_model)
    position = torch.arange(0, num_node, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    return pe

def gen_DiM_DiY(ind):
    ind += 23
    ind_DiM = ind % 30
    ind_DiY = ind % 365

    return ind_DiM, ind_DiY

def Informax_loss(DGI_pred, DGI_labels):
    BCE_loss = torch.nn.BCEWithLogitsLoss()
    loss = BCE_loss(DGI_pred, DGI_labels)
    return loss

def infoNCEloss(q, k):
    T = 0.05
    
    # q = q.expand_as(k)
    q = q.repeat(1, 1, 1, 7, 1)
    q = q.permute(0, 3, 4, 2, 1)
    k = k.permute(0, 3, 4, 2, 1)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    pos_sim = torch.sum(torch.mul(q, k), dim=-1)
    neg_sim = torch.matmul(q, k.transpose(-1, -2))
    pos = torch.exp(torch.div(pos_sim, T))
    neg = torch.sum(torch.exp(torch.div(neg_sim, T)), dim=-1)
    denominator = neg + pos
    return torch.mean(-torch.log(torch.div(pos, denominator)))

def load_DiY_DiW( ts_data):
    ts_list = [datetime.datetime.strptime(str(ts.decode()),"%Y-%m-%d").timetuple() for ts in ts_data]
    ts_diy = [ts.tm_wday for ts in ts_list]
    ts_diw = [ts.tm_yday for ts in ts_list]
    return ts_diy, ts_diw

def covid_daily2trend(st_data, opt = 'sum'):
    # st_data : B, T, N, D
    assert st_data.shape[1] % 7 == 0
    trend_num = st_data.shape[1] // 7 
    st_list = np.split(st_data, trend_num, axis=1) # B, 7, N, D
    st_data_trend = np.stack(st_list, axis=-1) # B, 7, N, D, trend
    if opt == 'sum':
        return np.sum(st_data_trend, axis=1).transpose(0, 3, 1, 2)
    elif opt == 'mean':
        # print(st_data_trend.shape)
        return np.mean(st_data_trend, axis=1)

def from_np2torch(inputs):
    cuda = True if torch.cuda.is_available() else False
    
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    inputs = list(
                        map(lambda x: TensorFloat(x), inputs)
                    )
    return inputs

def process_adj(adj_mx, adj_type: str):
    """process adjacency matrix.

    Args:
        file_path (str): file path
        adj_type (str): adjacency matrix type

    Returns:
        list of numpy.matrix: list of preproceesed adjacency matrices
        np.ndarray: raw adjacency matrix
    """
    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [calculate_symmetric_message_passing_adj(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        adj = [calculate_transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [calculate_transition_matrix(adj_mx).T, calculate_transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == "original":
        adj = [adj_mx]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj

def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))

class Weight_EMA_Update(object):

    def __init__(self, model, initial_state_dict, decay=0.999):
        self.model = model
        self.model.load_state_dict(initial_state_dict, strict=True)
        self.decay = decay

    def update(self, new_state_dict):
        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = (self.decay)*state_dict[key] + (1-self.decay)*new_state_dict[key]
            #state_dict[key] = (1-self.decay)*state_dict[key] + (self.decay)*new_state_dict[key]

        self.model.load_state_dict(state_dict)


class Grid2Graph:
    def __init__(self, row, col) -> None:
        self.row = row
        self.col = col
        self.areaNum = row * col
    
    def constructGraph(self, hop):
        mx = [-1, 0, 1, 0, -1, -1, 1, 1, 0]
        my = [0, -1, 0, 1, -1, 1, -1, 1, 0]
        def illegal(x, y):
            return x < 0 or y < 0 or x >= self.row or y >= self.col
        edges = list()
        for i in range(self.row):
            for j in range(self.col):
                n1 = self.idEncode(i, j)
                for k in range(len(mx)):
                    temx = i + mx[k]
                    temy = j + my[k]
                    if illegal(temx, temy):
                        continue
                    n2 = self.idEncode(temx, temy)
                    edges.append([n1, n2])
        edges.sort(key=lambda x: x[0]*1e5+x[1]) # 1e5 should be bigger than the number of areas
        
        adp=np.zeros((self.areaNum, self.areaNum))
        for e in range(len(edges)):
            # print('edge 0:{}, edge 1:{}'.format(edges[e][0], edges[e][1]))
            adp[edges[e][0]][edges[e][1]] = 1
        adp = np.linalg.matrix_power(adp, hop)
        adp[adp != 0 ] = 1
        # np.savetxt('./adj.txt', np.c_[adp],fmt='%d',delimiter='\t')]
        # np.savez('./mask/test_region.npz', adj=self.)
        
        return adp

    def idEncode(self, x, y):
        return x * self.col + y
