import pickle5 as pickle
from torch.utils.data import Dataset
import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .util import id_2_path_wave


class DataRetriever(Dataset):
    def __init__(self, paths, targets, synthetic=None, Config=None):
        self.paths = paths
        self.targets = targets
        self.synthetic = synthetic
        self.synthetic_keys = list(self.synthetic.keys()) if synthetic is not None else None
        self.neg_idxes = [i for i, t in enumerate(targets) if t == 0]

        self.Config = Config
        self.cons_funcs = Config.cons_funcs
        self.aggr_funcs = Config.aggr_funcs
        self.aggr_func_names = Config.aggr_func_names

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        target = self.targets[index]
        path = self.paths[index]
        if target > 0 and (self.synthetic is not None):
            path = self.paths[random.choice(self.neg_idxes)]
        waves = np.load(path)

        if self.cons_funcs or self.aggr_funcs:
            if self.cons_funcs:
                for transform in self.cons_funcs:
                    waves = transform(waves, sample_rate=2048)
            if self.aggr_funcs:
                if np.random.random() < self.Config.aggressive_aug_proba:
                    probas = np.array(
                        [getattr(self.Config(), f'{agg}_weight') for agg in self.aggr_func_names])
                    probas /= probas.sum()
                    transform = np.random.choice(self.aggr_funcs, p=probas)
                    waves = transform(waves, sample_rate=2048)
        else:
            if np.random.random() < 0.5:
                waves[[0, 1]] = waves[[1, 0]]
            if np.random.random() < 0.5:
                waves = -waves

        x = torch.FloatTensor(waves * 1e20) if self.Config.use_raw_wave else torch.FloatTensor(waves)
        if target > 0 and (self.synthetic is not None):
            w = torch.FloatTensor((self.synthetic[random.choice(self.synthetic_keys)]))
            w = w * max(random.gauss(3.6, 1), 1)
            shift_place = 512 + 384 - random.randrange(0, 768)
            w = np.pad(w[:, shift_place:], [(0, 0), (0, shift_place)], mode='constant')
            x += w
        target = torch.tensor(target, dtype=torch.float)
        return x, target


class DataRetrieverTest(Dataset):
    def __init__(self, paths, targets, transforms=None, Config=None):
        self.paths = paths
        self.targets = targets
        self.transforms = transforms
        self.Config = Config

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        waves = np.load(path).astype(np.float32)
        target = self.targets[index]
        if self.transforms is not None:
            waves = self.transforms(waves, sample_rate=2048)
        x = torch.FloatTensor(waves * 1e20) if self.Config.use_raw_wave else torch.FloatTensor(waves)
        target = torch.tensor(target, dtype=torch.float)
        return x, target


def generate_PL(fold, train_df, Config):
    if Config.PL_folder is None:
        return train_df
    pseudo_label_df = pd.read_csv(Config.PL_folder + f"/test_Fold_{fold}.csv")
    pseudo_label_df['file_path'] = pseudo_label_df['id'].apply(lambda x: id_2_path_wave(x, Config, False))
    pseudo_label_df["target"] = pseudo_label_df[f'preds_Fold_{fold}']
    test_df_2 = pseudo_label_df.copy()
    test_df_2['fold'] = -1

    if Config.PL_hard:
        test_df_2 = test_df_2.loc[~((test_df_2.target > 0.4) & (test_df_2.target < 0.85))]
        test_df_2.target = (test_df_2.target > 0.6).astype(np.int32)

    if Config.debug:
        test_df_2 = test_df_2.sample(n=10000, random_state=Config.seed).reset_index(drop=True)

    PL_train_df = pd.concat([train_df, test_df_2]).reset_index(drop=True)
    PL_train_df.reset_index(inplace=True, drop=True)
    return PL_train_df


def read_synthetic(Config):
    if not Config.synthetic: return None
    print("Read Synthetic Data")
    with open(Config.sim_data_path, 'rb') as handle:
        signal_dict = pickle.load(handle)
    return signal_dict


def read_data(Config):
    print("Read Data")
    train_df = pd.read_csv(Config.kaggleDataFolder + '/training_labels.csv')
    test_df = pd.read_csv(Config.kaggleDataFolder + '/sample_submission.csv')

    if Config.debug:
        Config.epochs = 1
        train_df = train_df.sample(n=50000, random_state=Config.seed).reset_index(drop=True)
        test_df = test_df.sample(n=10000, random_state=Config.seed).reset_index(drop=True)
    if Config.use_subset:
        train_df = train_df.sample(frac=Config.subset_frac, random_state=Config.seed).reset_index(drop=True)

    train_df['file_path'] = train_df['id'].apply(lambda x: id_2_path_wave(x, Config, True))
    test_df['file_path'] = test_df['id'].apply(lambda x: id_2_path_wave(x, Config, False))

    print("StratifiedKFold")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.seed)
    splits = skf.split(train_df, train_df["target"])
    train_df['fold'] = -1
    for fold, (train_index, valid_index) in enumerate(splits):
        train_df.loc[valid_index, "fold"] = fold
    train_df.groupby('fold')['target'].apply(lambda s: s.value_counts(normalize=True))
    return train_df, test_df
