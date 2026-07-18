import torch
import numpy as np
from torch.utils.data import Dataset
from utils.timefeatures import time_features

class TimeDataset(Dataset):
    def __init__(self, raw_data, mean, std, device, num_for_hist, num_for_futr, timestamps):
        
        self.device = device
        self.data = raw_data
        self.use_timestamps = timestamps is not None
        if self.use_timestamps:
            self.timestamps = time_features(timestamps)
            self.timestamps = self.timestamps.transpose(1, 0)
            self.timestamps = torch.from_numpy(self.timestamps).float()
        else:
            self.timestamps = None

        if len(self.data.shape) == 2:
            self.data = self.data.reshape(self.data.shape[0], self.data.shape[1], 1) # add c dimension first
        # permutate the data to (n, c, T)
        self.data = np.transpose(self.data, (1, 2, 0)).astype(np.float32)
        self.data = torch.from_numpy(self.data).float()

        self.mean = mean
        self.std = std
        self.num_for_hist = num_for_hist
        self.num_for_futr = num_for_futr

    def __len__(self):
        return self.data.shape[-1] - self.num_for_hist - self.num_for_futr + 1

    def __getitem__(self, idx):
        """
        :param idx: the index of the data
        :return:
        """
        data = self.data[:, 0, idx:idx + self.num_for_hist]
        data = self.normalize(data)

        clean_target = self.data[:, 0, idx + self.num_for_hist:idx + self.num_for_hist + self.num_for_futr]

        if not self.use_timestamps:
            return data, clean_target, clean_target, idx
        else:
            input_stamps = self.timestamps[idx:idx + self.num_for_hist]
            target_stamps = self.timestamps[idx + self.num_for_hist:idx + self.num_for_hist + self.num_for_futr]
            return data, clean_target, clean_target, input_stamps, target_stamps, idx

    def normalize(self, data):
        return (data - self.mean) / self.std

    def denormalize(self, data):
        return data * self.std + self.mean

    
class TimeDatasetwithWeight(Dataset):
    def __init__(self, raw_data, mean, std, device, 
                 num_for_hist, num_for_futr, timestamps,
                 weights):
        
        self.device = device
        self.data = raw_data
        self.use_timestamps = timestamps is not None
        if self.use_timestamps:
            self.timestamps = time_features(timestamps)
            self.timestamps = self.timestamps.transpose(1, 0)
            self.timestamps = torch.from_numpy(self.timestamps).float()
        else:
            self.timestamps = None

        if len(self.data.shape) == 2:
            self.data = self.data.reshape(self.data.shape[0], self.data.shape[1], 1) # add c dimension first
        
        self.data = np.transpose(self.data, (1, 2, 0)).astype(np.float32)
        self.data = torch.from_numpy(self.data).float()
        self.mean = mean
        self.std = std
        self.num_for_hist = num_for_hist
        self.num_for_futr = num_for_futr
        self.weights = torch.from_numpy(weights)
        
    def __len__(self):
        return self.data.shape[-1] - self.num_for_hist - self.num_for_futr + 1

    def __getitem__(self, idx):
        """
        :param idx: the index of the data
        :return:
        """
        data = self.data[:, 0, idx:idx + self.num_for_hist]
        data = self.normalize(data)

        clean_target = self.data[:, 0, idx + self.num_for_hist:idx + self.num_for_hist + self.num_for_futr]
        weight_data = self.weights[:, idx]
        if not self.use_timestamps:
            return data, clean_target, clean_target, weight_data, idx
        else:
            input_stamps = self.timestamps[idx:idx + self.num_for_hist]
            target_stamps = self.timestamps[idx + self.num_for_hist:idx + self.num_for_hist + self.num_for_futr]
            return data, clean_target, clean_target, input_stamps, target_stamps, weight_data, idx

    def normalize(self, data):
        return (data - self.mean) / self.std

    def denormalize(self, data):
        return data * self.std + self.mean


class AttackEvaluationSetLoad(Dataset):
    def __init__(self, dataset_dict, mean, std, device):

        self.atk_vars = dataset_dict['atk_vars']
        self.pattern_len = dataset_dict['pattern_len']
        self.device = device
        self.mean = mean
        self.std = std
        self.use_timestamps = 'input_stamps' in dataset_dict
        self.features = dataset_dict['features']
        self.targets = dataset_dict['targets']
        self.clean_targets = dataset_dict['clean_targets']
        self.indices = dataset_dict['indices']
        if self.use_timestamps:
            self.input_stamps = dataset_dict['input_stamps']
            self.target_stamps = dataset_dict['target_stamps']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        :param idx: the index of the data
        :return:
        """
        data = self.features[idx]
        data = self.normalize(data)

        target = self.targets[idx]
        clean_target = self.clean_targets[idx]

        idx = self.indices[idx]

        if not self.use_timestamps:
            return data, target, clean_target, idx
        else:
            input_stamps = self.input_stamps[idx]
            target_stamps = self.target_stamps[idx]
            return data, target, clean_target, input_stamps, target_stamps, idx
        
    def normalize(self, data):
        return (data - self.mean) / self.std

    def denormalize(self, data):
        return data * self.std + self.mean