import os
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.timefeatures import time_features
import pandas as pd


def load_raw_data(dataset_config):
    if 'PEMS' in dataset_config.dataset_name and ".npz" in dataset_config.data_filename:
        raw_data = np.load(dataset_config.data_filename)['data']
        train_data_seq = raw_data[:int(0.6 * raw_data.shape[0])]
        val_data_seq = raw_data[int(0.6 * raw_data.shape[0]):int(0.8 * raw_data.shape[0])]
        test_data_seq = raw_data[int(0.8 * raw_data.shape[0]):]

        train_mean = np.mean(train_data_seq, axis=(0, 1))
        train_std = np.std(train_data_seq, axis=(0, 1))
        if len(train_mean.shape) == 1:
            train_mean = train_mean[0]
            train_std = train_std[0]

        return train_mean, train_std, train_data_seq, test_data_seq

    else:
        raw_data = pd.read_csv(dataset_config.data_filename)
        raw_data_feats = raw_data.values[:, 1:]
        raw_data_stamps = raw_data.values[:, 0]
        raw_data_stamps = pd.to_datetime(raw_data_stamps)

        train_data_seq = raw_data_feats[:int(0.6 * raw_data_feats.shape[0])]
        val_data_seq = raw_data_feats[int(0.6 * raw_data_feats.shape[0]):int(0.8 * raw_data_feats.shape[0])]
        test_data_seq = raw_data_feats[int(0.8 * raw_data_feats.shape[0]):]

        train_data_stamps = raw_data_stamps[:int(0.6 * raw_data_stamps.shape[0])]
        val_data_stamps = raw_data_stamps[int(0.6 * raw_data_stamps.shape[0]):int(0.8 * raw_data_stamps.shape[0])]
        test_data_stamps = raw_data_stamps[int(0.8 * raw_data_stamps.shape[0]):]

        train_mean = np.mean(train_data_seq, axis=(0, 1))
        train_std = np.std(train_data_seq, axis=(0, 1))
        if len(train_mean.shape) == 1:
            train_mean = train_mean[0]
            train_std = train_std[0]

        if dataset_config.use_timestamps:
            return train_mean, train_std, train_data_seq, test_data_seq, train_data_stamps, test_data_stamps
        else:
            return train_mean, train_std, train_data_seq, test_data_seq
        

class TimeDataset(Dataset):
    def __init__(self, raw_data, mean, std, device, 
                 num_for_hist, num_for_futr, timestamps):
        
        self.device = device
        self.data = raw_data
        self.use_timestamps = timestamps is not None
        if self.use_timestamps:
            self.timestamps = time_features(timestamps)
            self.timestamps = self.timestamps.transpose(1, 0)
            self.timestamps = torch.from_numpy(self.timestamps).float().to(self.device)
        else:
            self.timestamps = None

        if len(self.data.shape) == 2:
            self.data = self.data.reshape(self.data.shape[0], self.data.shape[1], 1) # add c dimension first
        # permutate the data to (n, c, T)
        self.data = np.transpose(self.data, (1, 2, 0)).astype(np.float32)
        self.data = torch.from_numpy(self.data).float().to(self.device)

        self.init_poison_data()

        self.std = float(std)
        self.mean = float(mean)
        self.num_for_hist = num_for_hist
        self.num_for_futr = num_for_futr

    def __len__(self):
        return self.data.shape[-1] - self.num_for_hist - self.num_for_futr + 1

    def __getitem__(self, idx):
        """
        :param idx: the index of the data
        :return:
        """
        data = self.poisoned_data[:, 0:1, idx:idx + self.num_for_hist]
        data = self.normalize(data)

        poisoned_target = self.poisoned_data[:, 0, idx + self.num_for_hist:idx + self.num_for_hist + self.num_for_futr]
        clean_target = self.data[:, 0, idx + self.num_for_hist:idx + self.num_for_hist + self.num_for_futr]

        if not self.use_timestamps:
            return data, poisoned_target, clean_target, idx
        else:
            input_stamps = self.timestamps[idx:idx + self.num_for_hist]
            target_stamps = self.timestamps[idx + self.num_for_hist:idx + self.num_for_hist + self.num_for_futr]
            return data, poisoned_target, clean_target, input_stamps, target_stamps, idx

    def init_poison_data(self):
        self.poisoned_data = torch.clone(self.data).detach().to(self.device)

    def normalize(self, data):
        return (data - self.mean) / self.std

    def denormalize(self, data):
        return data * self.std + self.mean

class AttackEvaluateSet(TimeDataset):
    def __init__(self, attacker, raw_data, mean, std, device, 
                 num_for_hist, num_for_futr, timestamps):
        super(AttackEvaluateSet, self).__init__(raw_data, mean, std, device, num_for_hist, num_for_futr, timestamps)
        self.attacker = attacker

    def collate_fn(self, data):
        """
        :param data: the input data
        :return: the attacked data by the attacker
        """
        if self.use_timestamps:
            features, _ , clean_target, input_stamps, target_stamps, idx = zip(*data)
        else:
            features, _ , clean_target, idx = zip(*data)

        features = torch.stack(features, dim=0)
        clean_target = torch.stack(clean_target, dim=0)
        features = self.denormalize(features)

        data_bef = features[:, self.attacker.atk_vars, 0,
                   -self.attacker.trigger_len - self.attacker.bef_tgr_len:-self.attacker.trigger_len]
        triggers = self.attacker.predict_trigger(data_bef)[0]

        triggers = triggers.reshape(-1, self.attacker.atk_vars.shape[0], 1, self.attacker.trigger_len)
        features[:, self.attacker.atk_vars, :, -self.attacker.trigger_len:] = triggers

        target = clean_target.clone().detach().to(self.device)
        target[:, self.attacker.atk_vars, :self.attacker.pattern_len] = \
            self.attacker.target_pattern + features[:, self.attacker.atk_vars, :, -self.attacker.trigger_len - 1]

        features = self.normalize(features)
        if not self.use_timestamps:
            return features, target, clean_target, idx
        else:
            return features, target, clean_target, torch.stack(input_stamps, dim=0), torch.stack(target_stamps, dim=0), idx
    
    def save_attacked_dataset(self, dataloader, save_folder, 
                              trigger_len, pattern_len, 
                              atk_vars, atk_ts):
        """
        Process and save the attacked dataset for later use of evaluation
        :param dataloader: the dataloader
        :param save_path: the path to save the attacked dataset
        :return: None
        """
        all_original_features = []
        all_features = []
        all_targets = []
        all_clean_targets = []
        all_indices = []

        if self.use_timestamps:
            all_input_stamps = []
            all_target_stamps = []

        for data in dataloader:
            if self.use_timestamps:
                features, _ , clean_target, input_stamps, target_stamps, idx = data
            else:
                features, _ , clean_target, idx = data
                
            features = self.denormalize(features)
            original_feature = features.clone().detach()
            data_bef = features[:, self.attacker.atk_vars, 0,
                    -self.attacker.trigger_len - self.attacker.bef_tgr_len:-self.attacker.trigger_len]
            triggers = self.attacker.predict_trigger(data_bef)[0]
            if triggers is not None:
                triggers = triggers.reshape(-1, self.attacker.atk_vars.shape[0], 1, self.attacker.trigger_len)
                features[:, self.attacker.atk_vars, :, -self.attacker.trigger_len:] = triggers

            target = clean_target.clone().detach().to(self.device)
            target[:, self.attacker.atk_vars, :self.attacker.pattern_len] = \
                self.attacker.target_pattern + features[:, self.attacker.atk_vars, :, -self.attacker.trigger_len - 1]
            
            # Normalize the features
            all_original_features.append(original_feature.cpu())
            all_features.append(features.cpu())
            all_targets.append(target.cpu())
            all_clean_targets.append(clean_target.cpu())
            all_indices.append(idx.cpu())

            if self.use_timestamps:
                all_input_stamps.append(input_stamps.cpu())
                all_target_stamps.append(target_stamps.cpu())
            
        # Concatenate all batches
        dataset_dict = {
            'atk_vars': atk_vars.cpu() if atk_vars is not None else None,
            'atk_ts': atk_ts.cpu() if atk_ts is not None else None,
            'trigger_len': torch.tensor(trigger_len).cpu(),
            'pattern_len': torch.tensor(pattern_len).cpu(),
            'features': torch.cat(all_features, dim=0),
            'original_features': torch.cat(all_original_features, dim=0),
            'targets': torch.cat(all_targets, dim=0),
            'clean_targets': torch.cat(all_clean_targets, dim=0),
            'indices': torch.cat(all_indices, dim=0)
        }
        
        if self.use_timestamps:
            dataset_dict['input_stamps'] = torch.cat(all_input_stamps, dim=0)
            dataset_dict['target_stamps'] = torch.cat(all_target_stamps, dim=0)

        torch.save(dataset_dict, os.path.join(save_folder, 'test_attacked_data.pth'))
        print(f"Attacked test dataset saved to {os.path.join(save_folder, 'test_attacked_data.pth')}")
