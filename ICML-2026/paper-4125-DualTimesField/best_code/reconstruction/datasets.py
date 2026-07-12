import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data,
        seq_length,
        stride=1,
        normalize=True,
        overlap_ratio=0.5
    ):
        if isinstance(data, np.ndarray):
            self.data = torch.tensor(data, dtype=torch.float32)
        else:
            self.data = data
        self.seq_length = seq_length
        self.stride = max(1, int(seq_length * (1 - overlap_ratio)))
        self.num_samples = (len(self.data) - seq_length) // self.stride + 1
        
        if normalize:
            self.mean = self.data.mean(dim=0, keepdim=True)
            self.std = self.data.std(dim=0, keepdim=True) + 1e-7
            self.data = (self.data - self.mean) / self.std
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_length
        
        x = self.data[start:end].clone()
        t = torch.linspace(0, 1, self.seq_length)
        
        return x, t


class MultiDatasetLoader:
    def __init__(self, data_dir, datasets=None):
        self.data_dir = Path(data_dir)
        if datasets is None:
            datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 
                       'electricity', 'exchange_rate', 'traffic', 'weather', 'illness']
        self.datasets = datasets
        
    def load_dataset(self, name):
        if name in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
            path = self.data_dir / 'ETT-small' / f'{name}.csv'
        else:
            path = self.data_dir / name / f'{name}.csv'
            if name == 'illness':
                path = self.data_dir / 'illness' / 'national_illness.csv'
        
        df = pd.read_csv(path)
        
        if 'date' in df.columns:
            df = df.drop('date', axis=1)
        elif 'Date' in df.columns:
            df = df.drop('Date', axis=1)
        
        data = df.values.astype(np.float32)
        return data
    
    def get_dataset(self, name, seq_length, stride=1, normalize=True, 
                   train_ratio=0.7, val_ratio=0.1):
        data = self.load_dataset(name)
        n = len(data)
        
        if seq_length > n // 3:
            seq_length = max(32, n // 10)
            stride = max(1, seq_length // 4)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        if len(train_data) < seq_length:
            train_data = data[:max(seq_length + 1, int(n * 0.8))]
        if len(val_data) < seq_length:
            val_data = train_data
        if len(test_data) < seq_length:
            test_data = train_data
        
        train_dataset = TimeSeriesDataset(train_data, seq_length, stride, normalize)
        val_dataset = TimeSeriesDataset(val_data, seq_length, stride, normalize)
        test_dataset = TimeSeriesDataset(test_data, seq_length, stride, normalize)
        
        return train_dataset, val_dataset, test_dataset
    
    def get_num_variables(self, name):
        data = self.load_dataset(name)
        return data.shape[1]
