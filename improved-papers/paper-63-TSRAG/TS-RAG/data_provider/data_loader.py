import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.tools import convert_tsf_to_dataframe
import warnings
from pathlib import Path
from statsmodels.tsa.seasonal import STL
from typing import Tuple
import matplotlib.pyplot as plt
import random 

warnings.filterwarnings('ignore')

def get_custom_period_channel(data_path):
    pc = dict()
    if 'weather' in data_path:
        # per 10min
        pc["period"] = 36
        pc["channel"] = 21
    if 'traffic' in data_path:
        # per hour 
        pc["period"] = 24
        pc["channel"]= 862
    if 'electricity' in data_path:
        # per hour 
        pc["period"] = 24
        pc["channel"] = 321
    if 'illness' in data_path:
        # 1week
        pc["period"] = 12
        pc["channel"] = 7
    if 'exchange' in data_path:
        # 1week
        pc["period"] = 24
        pc["channel"] = 8
    return pc

def decompose( 
    x: torch.Tensor, period: int = 7
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose input time series into trend, seasonality and residual components using STL.

    Args:
        x (torch.Tensor): Input time series. Shape: (1, seq_len).
        period (int, optional): Period of seasonality. Defaults to 7.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Decomposed components. Shape: (1, seq_len).
    """
    # print('in' , x.shape)
    # x = x.squeeze(0).cpu().numpy()
    if len(x.shape) ==2 : 
        x = x.squeeze(0)
    decomposed = STL(x, period=period).fit()
    trend = decomposed.trend.astype(np.float32)
    seasonal = decomposed.seasonal.astype(np.float32)
    residual = decomposed.resid.astype(np.float32)
    return (
        torch.from_numpy(trend).unsqueeze(0),
        torch.from_numpy(seasonal).unsqueeze(0),
        torch.from_numpy(residual).unsqueeze(0),
    )
    
class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False, train_ratio = 1.0 ,model_id ='', return_index=False, return_feature_id=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_size_ratio = 1.0 
        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio 
        self.root_path = root_path
        self.data_path = data_path
        self.return_index = return_index
        self.return_feature_id = return_feature_id
        
        self.__read_data__()

        self.model_id = model_id
        
        self.period = 24 
        self.channel= 7
        if 'multi' in self.model_id:
            self.enc_in =1 
        else : 
            self.enc_in = self.data_x.shape[-1]
            
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        
    def draw_decompose(self, x , trend, seasonal, residual):
        plt.figure(figsize=(10, 6))  # Optional: Specifies the figure size
        # Plot each array
        x = x.reshape(-1,)
        trend = trend.reshape(-1,)
        seasonal = seasonal.reshape(-1,)
        residual = residual.reshape(-1,)
        print(x.shape , trend.shape)
        
        plt.plot(x, label='x A')
        plt.plot(trend, label='trend B')
        plt.plot(seasonal, label='seasonal C')
        plt.plot(residual, label='residual D')
        ii = random.randint(0,100)
        # Adding labels
        plt.xlabel('Index')  # Assuming the index represents the x-axis
        plt.ylabel('Value')  # The y-axis label
        plt.title('Plot of Four Arrays')  # Title of the plot
        plt.legend()
        plt.savefig(f'/p/selfdrivingpj/projects_time/NeurIPS2023-One-Fits-All/Long-term_Forecasting/figures/{ii}.jpg')
        plt.cla()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [int(12 * 30 * 24 * self.train_ratio), 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
            

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        # (17420, 7) 
        # print(data.shape)
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        print(self.set_type ,self.data_x.shape)
                
    def __getitem__(self, index):
        
        '''
            single_linr  single_linr_decp multi_linr_att  multi_patch multi_patch_attn multi_patch_decp
        '''
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        if self.return_index:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, index
        elif self.return_feature_id:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, feat_id
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        
    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', 
                 percent=100, max_len=-1, train_all=False  , model_id = '', return_index=False, return_feature_id=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.return_index = return_index
        self.return_feature_id = return_feature_id
        self.__read_data__()

        self.model_id = model_id 
        self.period = 60 
        self.channel= 7
        if 'multi' in self.model_id:
            self.enc_in =1 
        else : 
            # ofa and single
            self.enc_in = self.data_x.shape[-1]

        self.tot_len = (len(self.data_x) - self.seq_len - self.pred_len + 1)
        
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):  
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        if self.return_index:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, index
        elif self.return_feature_id:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, feat_id
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        
    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=10, max_len=-1, train_all=False , train_ratio=1.0 , model_id=''):

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.model_id= model_id
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

        if 'weather' in data_path:
            # per 10min
            self.period = 36
            self.channel= 21
        if 'traffic' in data_path:
            # per hour 
            self.period = 24
            self.channel= 862
        if 'electricity' in data_path:
            # per hour 
            self.period = 24
            self.channel= 321
        if 'illness' in data_path:
            # 1week
            self.period = 12
            self.channel= 7

        self.enc_in = 1 
            
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        print(self.data_x.shape)
        
    def __getitem__(self, index):
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)  # [c, seq_len]
        y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)  # [c, pred_len]
        return x , y ,  seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom_S(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=10, max_len=-1, train_all=False , train_ratio=1.0 , model_id='', return_index=False, return_feature_id=False):

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.model_id= model_id
        self.root_path = root_path
        self.data_path = data_path
        self.return_index = return_index
        self.return_feature_id = return_feature_id
        self.__read_data__()
        
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

        pc = get_custom_period_channel(data_path)
        self.period = pc["period"]
        self.channel = pc["channel"]

        if 'multi' in self.model_id:
            self.enc_in = 1 
        else : 
            self.enc_in = self.data_x.shape[-1]
            assert self.enc_in == self.channel
            
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        # import pdb; pdb.set_trace()
        print(self.data_x.shape)
        
    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        if self.return_index:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, index
        elif self.return_feature_id:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, feat_id
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_hour_retrieve(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False, train_ratio = 1.0 ,model_id ='', top_k=1, retriever_rawdata=None, mode='only_self'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_size_ratio = 1.0 
        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio 
        self.root_path = root_path
        self.data_path = data_path
        self.top_k = top_k
        self.retriever_rawdata = retriever_rawdata
        
        self.model_id = model_id
        self.mode = mode

        self.period = 24 
        self.channel = 7
        self.__read_data__()
        
        if 'multi' in self.model_id:
            self.enc_in =1 
        else : 
            self.enc_in = self.data_x.shape[-1]
            
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        # pdb.set_trace()
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # original data
        df_ori_columns = [column for column in df_raw.columns if 'boundary' not in column and 'timestamp' not in column and 'distance' not in column]
        df_ori = df_raw[df_ori_columns]

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [int(12 * 30 * 24 * self.train_ratio), 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
            
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_ori.columns[1:]
            df_data = df_ori[cols_data]
        elif self.features == 'S':
            df_data = df_ori[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        # (17420, 7) 
        # print(data.shape)
        # Load retrieval data (boundary_idx, timestamp_idx, distance) with top_k filtering
        boundary_idx_cols = sorted([col for col in df_raw.columns if col.startswith('boundary_idx_')],
                                key=lambda x: int(x.split('_')[-1]))[:self.top_k*self.channel]
        timestamp_idx_cols = sorted([col for col in df_raw.columns if col.startswith('timestamp_idx_')],
                                    key=lambda x: int(x.split('_')[-1]))[:self.top_k*self.channel]
        distance_cols = sorted([col for col in df_raw.columns if col.startswith('distance_')],
                            key=lambda x: int(x.split('_')[-1]))[:self.top_k*self.channel]
        # pdb.set_trace()

        self.boundary_idx = np.array(df_raw[boundary_idx_cols].values[border1:border2], dtype=np.int32)     # [8640, 7*topk]
        self.timestamp_idx = np.array(df_raw[timestamp_idx_cols].values[border1:border2], dtype=np.int32)   # [8640, 7*topk]
        self.distance = df_raw[distance_cols].values[border1:border2]                                       # [8640, 7*topk]

        L, W = self.boundary_idx.shape
        self.boundary_idx = self.boundary_idx.reshape(L, self.top_k, -1) # [8640, topk, 7]
        self.timestamp_idx = self.timestamp_idx.reshape(L, self.top_k, -1) # [8640, topk, 7]
        self.distance = self.distance.reshape(L, self.top_k, -1) # [8640, topk, 7]

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        print(self.set_type ,self.data_x.shape)
                
    def __getitem__(self, index):
        
        '''
            single_linr  single_linr_decp multi_linr_att  multi_patch multi_patch_attn multi_patch_decp
        '''
     
        # if 'ofa' in  self.model_id  and 'retrieve' in self.model_id:
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # Get retrieval data
        boundary_idx = self.boundary_idx[s_begin, :, feat_id]  # [top_k]
        timestamp_idx = self.timestamp_idx[s_begin, :, feat_id]  # [top_k]
        distances = self.distance[s_begin, :, feat_id]  # [top_k]
        retrieved_seqs = []

        # Reconstruct original data from retrieval values
        # In 'only_self' mode, each channel retrieves its own data; in 'all_vars' mode, each channel retrieves data from all channels
        for i in range(self.top_k):
            if self.mode == 'only_self' or self.mode == 'only_self_train':
                # retriever_rawdata: [var1_raw_data, var2_raw_data, ...]
                retrieved_seq = self.retriever_rawdata[feat_id][timestamp_idx[i]:timestamp_idx[i]+self.seq_len+self.pred_len]  # [seq_len + pred_len]

            elif self.mode == 'all_vars' or self.mode == 'all_vars_train':
                retrieved_seq = self.retriever_rawdata[boundary_idx[i]][timestamp_idx[i]:timestamp_idx[i]+self.seq_len+self.pred_len]  # [seq_len + pred_len]

            assert len(retrieved_seq) == self.seq_len + self.pred_len
            retrieved_seqs.append(retrieved_seq)

        # return seq_x, seq_y, seq_x_mark, seq_y_mark, torch.tensor(retrieved_seqs), distances
        return seq_x, seq_y, seq_x_mark, timestamp_idx, torch.tensor(retrieved_seqs), distances
    
    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_ETT_minute_retrieve(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', 
                 percent=100, max_len=-1, train_all=False, train_ratio = 1.0 ,model_id ='', top_k=1, retriever_rawdata=None, mode='only_self'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_size_ratio = 1.0 
        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio 
        self.root_path = root_path
        self.data_path = data_path
        self.top_k = top_k
        self.retriever_rawdata = retriever_rawdata
        
        self.model_id = model_id
        self.mode = mode

        self.period = 24 
        self.channel = 7
        self.__read_data__()
        
        if 'multi' in self.model_id:
            self.enc_in =1 
        else : 
            self.enc_in = self.data_x.shape[-1]
            
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        # pdb.set_trace()
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # original data
        df_ori_columns = [column for column in df_raw.columns if 'boundary' not in column and 'timestamp' not in column and 'distance' not in column]
        df_ori = df_raw[df_ori_columns]

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
            
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_ori.columns[1:]
            df_data = df_ori[cols_data]
        elif self.features == 'S':
            df_data = df_ori[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        # (17420, 7) 
        # print(data.shape)
        # Load retrieval data (boundary_idx, timestamp_idx, distance) with top_k filtering
        boundary_idx_cols = sorted([col for col in df_raw.columns if col.startswith('boundary_idx_')],
                                key=lambda x: int(x.split('_')[-1]))[:self.top_k*self.channel]
        timestamp_idx_cols = sorted([col for col in df_raw.columns if col.startswith('timestamp_idx_')],
                                    key=lambda x: int(x.split('_')[-1]))[:self.top_k*self.channel]
        distance_cols = sorted([col for col in df_raw.columns if col.startswith('distance_')],
                            key=lambda x: int(x.split('_')[-1]))[:self.top_k*self.channel]
        # pdb.set_trace()

        self.boundary_idx = np.array(df_raw[boundary_idx_cols].values[border1:border2], dtype=np.int32)     # [8640, 7*topk]
        self.timestamp_idx = np.array(df_raw[timestamp_idx_cols].values[border1:border2], dtype=np.int32)   # [8640, 7*topk]
        self.distance = df_raw[distance_cols].values[border1:border2]                                       # [8640, 7*topk]

        L, W = self.boundary_idx.shape
        self.boundary_idx = self.boundary_idx.reshape(L, self.top_k, -1) # [8640, topk, 7]
        self.timestamp_idx = self.timestamp_idx.reshape(L, self.top_k, -1) # [8640, topk, 7]
        self.distance = self.distance.reshape(L, self.top_k, -1) # [8640, topk, 7]

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        print(self.set_type ,self.data_x.shape)
                
    def __getitem__(self, index):
        
        '''
            single_linr  single_linr_decp multi_linr_att  multi_patch multi_patch_attn multi_patch_decp
        '''
     
        # if 'ofa' in  self.model_id  and 'retrieve' in self.model_id:
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # Get retrieval data
        boundary_idx = self.boundary_idx[s_begin, :, feat_id]  # [top_k]
        timestamp_idx = self.timestamp_idx[s_begin, :, feat_id]  # [top_k]
        distances = self.distance[s_begin, :, feat_id]  # [top_k]
        retrieved_seqs = []

        # Reconstruct original data from retrieval values
        # In 'only_self' mode, each channel retrieves its own data; in 'all_vars' mode, each channel retrieves data from all channels
        for i in range(self.top_k):
            if self.mode == 'only_self' or self.mode == 'only_self_train':
                # retriever_rawdata: [var1_raw_data, var2_raw_data, ...]
                retrieved_seq = self.retriever_rawdata[feat_id][timestamp_idx[i]:timestamp_idx[i]+self.seq_len+self.pred_len]  # [seq_len + pred_len]

            elif self.mode == 'all_vars' or self.mode == 'all_vars_train':
                retrieved_seq = self.retriever_rawdata[boundary_idx[i]][timestamp_idx[i]:timestamp_idx[i]+self.seq_len+self.pred_len]  # [seq_len + pred_len]

            assert len(retrieved_seq) == self.seq_len + self.pred_len
            retrieved_seqs.append(retrieved_seq)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, torch.tensor(retrieved_seqs), distances
    
    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class MultiDatasetSampler(IterableDataset):
    def __init__(self, datasets, probabilities):
        super().__init__()
        self.datasets = datasets
        self.probabilities = probabilities

    def __iter__(self):
        # Shuffle each dataset whenever iterators are initialized
        iterators = [iter(self._shuffle_dataset(dataset)) for dataset in self.datasets]
        while True:
            dataset_idx = random.choices(range(len(self.datasets)), weights=self.probabilities, k=1)[0]
            try:
                yield next(iterators[dataset_idx])
            except StopIteration:
                # Reshuffle the dataset and reinitialize its iterator
                iterators[dataset_idx] = iter(self._shuffle_dataset(self.datasets[dataset_idx]))

    @staticmethod
    def _shuffle_dataset(dataset):
        if isinstance(dataset, Dataset):  # Ensure 'dataset' is a standard Dataset
            indices = list(range(len(dataset)))
            random.shuffle(indices)  # Shuffle indices randomly
            return (dataset[i] for i in indices)  # Return data in shuffled order
        else:
            # If 'dataset' is an IterableDataset, return it as-is (cannot be shuffled)
            return dataset
        
class Dataset_Custom_retrieve(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='traffic.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False, train_ratio = 1.0 ,model_id ='', top_k=1, retriever_rawdata=None, mode='only_self'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_size_ratio = 1.0 
        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio 
        self.root_path = root_path
        self.data_path = data_path
        self.top_k = top_k
        self.retriever_rawdata = retriever_rawdata
        
        self.model_id = model_id
        self.mode = mode
        pc = get_custom_period_channel(data_path)
        self.period = pc["period"] 
        self.channel = pc["channel"]
        self.__read_data__()
        
        if 'multi' in self.model_id:
            self.enc_in =1 
        else : 
            self.enc_in = self.data_x.shape[-1]
            
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        # pdb.set_trace()
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # original data
        df_ori_columns = [column for column in df_raw.columns if 'boundary' not in column and 'timestamp' not in column and 'distance' not in column]
        df_ori = df_raw[df_ori_columns]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
            
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_ori.columns[1:]
            df_data = df_ori[cols_data]
        elif self.features == 'S':
            df_data = df_ori[[self.target]]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        # Load retrieval data (boundary_idx, timestamp_idx, distance) with top_k filtering
        boundary_idx_cols = sorted([col for col in df_raw.columns if col.startswith('boundary_idx_')],
                                key=lambda x: int(x.split('_')[-1]))[:self.top_k*self.channel]
        timestamp_idx_cols = sorted([col for col in df_raw.columns if col.startswith('timestamp_idx_')],
                                    key=lambda x: int(x.split('_')[-1]))[:self.top_k*self.channel]
        distance_cols = sorted([col for col in df_raw.columns if col.startswith('distance_')],
                            key=lambda x: int(x.split('_')[-1]))[:self.top_k*self.channel]
        # pdb.set_trace()
        self.boundary_idx = np.array(df_raw[boundary_idx_cols].values[border1:border2], dtype=np.int32)     # [8640, 7*topk]
        self.timestamp_idx = np.array(df_raw[timestamp_idx_cols].values[border1:border2], dtype=np.int32)   # [8640, 7*topk]
        self.distance = df_raw[distance_cols].values[border1:border2]                                       # [8640, 7*topk]
        L, W = self.boundary_idx.shape
        self.boundary_idx = self.boundary_idx.reshape(L, self.top_k, -1) # [8640, topk, 7]
        self.timestamp_idx = self.timestamp_idx.reshape(L, self.top_k, -1) # [8640, topk, 7]
        self.distance = self.distance.reshape(L, self.top_k, -1) # [8640, topk, 7]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        print(self.set_type ,self.data_x.shape)
                
    def __getitem__(self, index):
        
        '''
            single_linr  single_linr_decp multi_linr_att  multi_patch multi_patch_attn multi_patch_decp
        '''
     
        # if 'ofa' in  self.model_id  and 'retrieve' in self.model_id:
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # Get retrieval data
        boundary_idx = self.boundary_idx[s_begin, :, feat_id]  # [top_k]
        timestamp_idx = self.timestamp_idx[s_begin, :, feat_id]  # [top_k]
        distances = self.distance[s_begin, :, feat_id]  # [top_k]
        retrieved_seqs = []
        # Reconstruct original data from retrieval values
        # In 'only_self' mode, each channel retrieves its own data; in 'all_vars' mode, each channel retrieves data from all channels
        for i in range(self.top_k):
            if self.mode == 'only_self' or self.mode == 'only_self_train':
                # retriever_rawdata: [var1_raw_data, var2_raw_data, ...]
                retrieved_seq = self.retriever_rawdata[feat_id][timestamp_idx[i]:timestamp_idx[i]+self.seq_len+self.pred_len]  # [seq_len + pred_len]
            elif self.mode == 'all_vars' or self.mode == 'all_vars_train':
                retrieved_seq = self.retriever_rawdata[boundary_idx[i]][timestamp_idx[i]:timestamp_idx[i]+self.seq_len+self.pred_len]  # [seq_len + pred_len]
            assert len(retrieved_seq) == self.seq_len + self.pred_len
            retrieved_seqs.append(retrieved_seq)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, torch.tensor(retrieved_seqs), distances
    
    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)