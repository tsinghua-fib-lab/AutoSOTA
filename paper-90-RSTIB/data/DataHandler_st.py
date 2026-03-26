import numpy as np
from Yaml2Params import args
import torch
import os
from utils.util import normalize_dataset, split_data_by_ratio, Add_Window_Horizon,get_adjacency_binary, data_loader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DataHandler:
    def __init__(self):
        pass
    
    def get_dataloader(self, normalizer='max01'):
        data, time = self.load_st_dataset(args.dataset)  # B, N, D
        data, scaler = normalize_dataset(data, normalizer, False)
        time = self.build_time(time)
        x_tra, y_tra, x_val, y_val, x_test, y_test = self.get_raw_data(data)
        TEx_tra, TEy_tra, TEx_val, TEy_val, TEx_tst, TEy_tst = self.get_raw_data(time)
        tra_TE = np.concatenate([TEx_tra, TEy_tra], axis = 1)
        val_TE = np.concatenate([TEx_val, TEy_val], axis = 1)
        tst_TE = np.concatenate([TEx_tst, TEy_tst], axis = 1)

        adj, adj_weight = self.build_adj()
        sp_adj = adj
        sp_adj_w = adj_weight
        temp_adj = concat_temp_adj()

        tra_loader = data_loader(x_tra, y_tra, tra_TE, args.batch_size, shuffle=True, drop_last=True)
        val_loader = data_loader(x_val, y_val, val_TE, args.batch_size, shuffle=False, drop_last=True)
        tst_loader = data_loader(x_test, y_test, tst_TE, args.batch_size, shuffle=False, drop_last=False)

        return tra_loader, val_loader, tst_loader, scaler, sp_adj, sp_adj_w
    
    
    def get_raw_data(self, data):
        
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
        
        # add time window
        
        x_tra, y_tra = Add_Window_Horizon(data_train, window=args.lag, horizon=args.horizon, single=False)
        x_val, y_val = Add_Window_Horizon(data_val, window=args.lag, horizon=args.horizon, single=False)
        x_test, y_test = Add_Window_Horizon(data_test, window=args.lag, horizon=args.horizon, single=False)
        
        
        print('Train: ', x_tra.shape, y_tra.shape)
        print('Val: ', x_val.shape, y_val.shape)
        print('Test: ', x_test.shape, y_test.shape)
        return x_tra, y_tra, x_val, y_val, x_test, y_test

    def build_time(self, time):
        dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))
        timeofday = (time.hour * 3600 + time.minute * 60 + time.second) \
                    // time.freq.delta.total_seconds()
        timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
        time = torch.cat((timeofday, dayofweek), -1)
        return time
    def is_increasing(self, loc_list):
        return all(x<y for x, y in zip(loc_list, loc_list[1:]))

    def load_st_dataset(self, dataset):
        # output B, N, D
        if dataset == 'PEMS4':
            data_path = os.path.join('./data/PEMS04/PEMS04.npz')
            data = np.load(data_path)['data'][:, :, 0]
            start_date = '2018-01-01 00:00:00'
            time = pd.date_range(start_date, periods=data.shape[0], freq = '5MIN')
        elif dataset == 'PEMS8':
            data_path = os.path.join('./data/PEMS08/PEMS08.npz')
            data = np.load(data_path)['data'][:, :, 0]
            start_date = '2016-07-01 00:00:00'
            time = pd.date_range(start_date, periods=data.shape[0], freq = '5MIN')
        elif dataset == 'PEMS3':
            data_path = os.path.join('./data/PEMS03/PEMS03.npz')
            data = np.load(data_path)['data'][:, :, 0]
            start_date = '2018-09-01 00:00:00'
            time = pd.date_range(start_date, periods=data.shape[0], freq = '5MIN')
        elif dataset == 'PEMS7':
            data_path = os.path.join('./data/PEMS07/PEMS07.npz')
            data = np.load(data_path)['data'][:, :, 0]
            start_date = '2017-05-01 00:00:00'
            time = pd.date_range(start_date, periods=data.shape[0], freq = '5MIN')
        else:
            raise ValueError
        
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)
        print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
        return data, time
    
    def build_adj(self, ):
        adj_ori = get_adjacency_binary(distance_df_filename=args.adj_filename,
                                       num_of_vertices=args.num_nodes, id_filename=args.id_filename, args = args)
        adj_weight = get_adjacency_binary(distance_df_filename=args.adj_filename,
                                       num_of_vertices=args.num_nodes, id_filename=args.id_filename, type_='distance', self_loop=True, args = args)
        
        return adj_ori, adj_weight  
    def build_st_adj(self, ):
        adj_ori = get_adjacency_binary(distance_df_filename=args.adj_filename,
                                       num_of_vertices=args.num_nodes, id_filename=args.id_filename)
        adj_weight = get_adjacency_binary(distance_df_filename=args.adj_filename,
                                       num_of_vertices=args.num_nodes, id_filename=args.id_filename, type_='distance', self_loop=True)
        
        return adj_ori, adj_weight        
    
def eval_adj(adj, out_name):
    plt.figure(figsize=(20, 20))
    sns.heatmap(adj, cmap=plt.get_cmap('viridis', 6), center=None, robust=False, square=True, xticklabels=False, yticklabels=False)##30
    plt.tight_layout()
    plt.savefig('./fig/{}_adj.png'.format(out_name))

def concat_sp_adj(adj_ori):
    pad_adj = np.zeros((int(args.num_nodes), int(args.num_nodes)),                dtype=np.float32)
    adj_row = [adj_ori] +  [pad_adj]*(args.lag -1)
    adj_row = np.concatenate(adj_row, axis=1)
    adj = adj_row
    for idx in range(args.lag - 1):
        adj_new = np.roll(adj_row, args.num_nodes*(idx+1), axis=1)
        adj = np.vstack((adj, adj_new))
    return adj

def concat_temp_adj():
    adj_ori = np.zeros((int(args.num_nodes), int(args.num_nodes)),                dtype=np.float32)
    pad_adj = np.eye(int(args.num_nodes), int(args.num_nodes))
    adj_row = [adj_ori] +  [pad_adj]*(args.lag -1)
    adj_row = np.concatenate(adj_row, axis=1)
    adj = adj_row
    for idx in range(args.lag - 1):
        adj_new = np.roll(adj_row, args.num_nodes*(idx+1), axis=1)
        adj = np.vstack((adj, adj_new))
    return adj
    
