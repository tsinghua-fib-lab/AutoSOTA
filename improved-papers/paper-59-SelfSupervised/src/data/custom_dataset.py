from torch.utils.data import Dataset, DataLoader
import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
from tqdm import tqdm
import time
import torch
from einops import rearrange
from typing import Any, Dict, Optional, Tuple

data_dict = {
    'ETTh1': 'ETT-small/ETTh1.csv',
    'ETTh2': 'ETT-small/ETTh2.csv',
    'ETTm1': 'ETT-small/ETTm1.csv',
    'ETTm2': 'ETT-small/ETTm2.csv',
    'PEMS03': 'PEMS03/PEMS03.csv',
    'PEMS04': 'PEMS04/PEMS04.csv',
    'PEMS07': 'PEMS07/PEMS07.csv',
    'PEMS08': 'PEMS08/PEMS08.csv',
    'weather': 'weather/weather.csv',
    'traffic': 'traffic/traffic.csv',
    'electricity': 'electricity/electricity.csv',
}

class Scaler():
    def __init__(self):
        super(Scaler, self).__init__()
        self.scaler = StandardScaler()
        self.means = None
        self.stds = None

    def fit(self, data):
        self.scaler.fit(data)
        self.means = self.scaler.mean_
        self.stds = self.scaler.scale_

    def transform(self, data):
        B, C, L = data.shape
        data = rearrange(data, 'b c l -> (b l) c')
        out = (data - torch.Tensor(self.means).to(data)) / torch.Tensor(self.stds).to(data)
        out = rearrange(out, '(b l) c -> b c l', l=L).contiguous()
        return out

    def inverse_transform(self, data):
        B, C, L = data.shape
        data = rearrange(data, 'b c l -> (b l) c')
        out = data * (torch.Tensor(self.stds).to(data)) + torch.Tensor(self.means).to(data)
        out = rearrange(out, '(b l) c -> b c l', l=L).contiguous()
        return out

class CustomDataset(Dataset):
    def __init__(self,
                 name: str,
                 data_dir: str,
                 task: str,
                 scale: bool,
                 train_val_test_split: Tuple[float, float, float],
                 input_size: int,
                 output_size: int,
                 cycle_len: int, 
                 num_channels: int,
                 flag: str):
        self.name = name
        self.data_dir = data_dir
        self.task = task
        self.scale = scale
        self.scaler = Scaler()
        self.num_channels = num_channels
        self.input_size =  input_size  
        self.output_size = output_size
        self.cycle_len = cycle_len
        self.train_ratio, self.val_ratio, self.test_ratio = train_val_test_split
        set_types = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = set_types[flag]
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.data_dir + data_dict[self.name])
        cols = list(df_raw.columns)
        df_data = df_raw[cols[1:]]

        if self.name == 'ETTh1' or self.name == 'ETTh2':
            data_length = 20 * 30 * 24
        elif self.name == 'ETTm1' or self.name == 'ETTm2':
            data_length = 20 * 30 * 24 * 4
        else:
            data_length = len(df_data)

        num_train = int(data_length * self.train_ratio)
        num_val = int(data_length * self.val_ratio)
        num_test = int(data_length * self.test_ratio)

        border1s = [0, num_train - self.input_size, num_train + num_val - self.input_size]
        border2s = [num_train, num_train + num_val, data_length]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)

        data = df_data.values
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        df_stamp['month'] = df_stamp['date'].dt.month - 1
        df_stamp['day'] = df_stamp['date'].dt.day - 1
        df_stamp['weekday'] = df_stamp['date'].dt.weekday
        df_stamp['hour'] = df_stamp['date'].dt.hour
        df_stamp['minute'] = df_stamp['date'].dt.minute / 15
        data_stamp = df_stamp.drop(labels='date', axis=1).values

        self.data_x = torch.tensor(data[border1:border2]).float()
        self.data_y = torch.tensor(data[border1:border2]).float()
        self.data_stamp = data_stamp
        self.cycle_index = (torch.arange(data_length) % self.cycle_len)[border1:border2]  

    def __getitem__(self, index):
        x_mark, y_mark = {}, {}
        index = index + self.input_size
        x = self.data_x[index - self.input_size:index, ...].transpose(-1, -2)
        y = self.data_y[index:index + self.output_size, ...].transpose(-1, -2)
        x_mark['time_stamp'] = self.data_stamp[index - self.input_size:index]
        x_mark['pos_stamp'] = torch.arange(0, self.input_size)
        y_mark['time_stamp'] = self.data_stamp[index:index + self.output_size]
        y_mark['pos_stamp'] = torch.arange(0, self.output_size)
        x_mark['channel'] = torch.arange(0, self.num_channels)
        y_mark['channel'] = torch.arange(0, self.num_channels)
        x_mark['cycle_index'] = self.cycle_index[index - self.input_size]
        y_mark['cycle_index'] = self.cycle_index[index]
        return x, y, x_mark, y_mark

    def __len__(self):
        return self.data_x.shape[0] - self.input_size - self.output_size + 1



