import numpy as np
from multiprocessing import Pool
from numpy.lib.shape_base import _take_along_axis_dispatcher
import torch
from torch.nn.functional import feature_alpha_dropout
import torch.utils.data as D
import matplotlib.pyplot as plt


'''
Data utilities
'''
def _load_signal(p):
    return np.load(p).astype(np.float32), p


def load_signal_cache(paths, 
                      cache_limit=10, # in GB
                      n_jobs=1):

    size_in_gb = 0
    cache = {}
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            for s, p in pool.imap_unordered(_load_signal, paths):
                cache[p] = s
                size_in_gb += s.nbytes / (1024 ** 3)
                if size_in_gb > cache_limit:
                    print(f'{len(cache)} items / {size_in_gb:.2f} GB cache loaded.')
                    return cache
    else:
        for p in paths:
            s, _ = _load_signal(p)
            size_in_gb += s.nbytes / (1024 ** 3)
            if size_in_gb > cache_limit:
                print(f'{len(cache)} items / {size_in_gb:.2f} GB cache loaded.')
                return cache

    print(f'{len(cache)} items / {size_in_gb:.2f} GB cache loaded.')
    return cache


'''
Dataset
'''
class G2NetDataset(D.Dataset):
    '''
    Amplitude stats
    [RAW]
    max of max: [4.6152116e-20, 4.2303907e-20, 1.1161064e-20]
    mean of max: [1.8438003e-20, 1.8434544e-20, 5.0978556e-21]
    max of mean: [1.5429503e-20, 1.5225015e-20, 3.1584522e-21]
    [BANDPASS]
    max of max: [1.7882743e-20, 1.8305723e-20, 9.5750025e-21]
    mean of max: [7.2184587e-21, 7.2223450e-21, 2.4932809e-21]
    max of mean: [6.6964011e-21, 6.4522511e-21, 1.4383649e-21]
    '''
    def __init__(self, 
                 paths, 
                 targets=None, 
                 spectrogram=None, 
                 norm_factor=None,
                 transforms=None,
                 transforms2=None, 
                 cache=None,
                 mixup=False,
                 mixup_alpha=0.4, 
                 mixup_option='random',
                 hard_label=False,
                 lor_label=False,
                 is_test=False,
                 pseudo_label=False,
                 test_targets=None,
                 test_paths=None,
                 test_cache=None,
                 return_index=False,
                 return_test_index=False,
                 ):
        self.paths = paths
        self.targets = targets
        self.negative_idx = np.where(self.targets == 0)[0]
        self.spectr = spectrogram
        self.norm_factor = norm_factor
        self.transforms = transforms
        self.transforms2 = transforms2 # additional transforms
        self.cache = cache
        self.test_cache = None
        self.mixup = mixup
        self.alpha = mixup_alpha
        self.mixup_option = mixup_option
        self.hard_label = hard_label
        self.lor_label = lor_label
        self.is_test = is_test
        self.pseudo_label = pseudo_label
        self.return_index = return_index
        self.return_test_index = return_test_index
        if self.is_test:
            self.mixup = False
            self.pseudo_label = False
        if self.pseudo_label:
            self.paths = np.concatenate([self.paths, test_paths])
            self.targets = np.concatenate([self.targets, test_targets])
            self.test_cache = test_cache
            self.test_index = np.array([0]*len(paths) + [1]*len(test_paths)).astype(np.uint8)
        else:
            self.test_index = np.array([0]*len(self.paths)).astype(np.uint8)

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        signal, sub_signal, target = self._get_signal_target(index)
        if self.mixup:
            if self.mixup_option == 'random':
                idx2 = np.random.randint(0, len(self))
                lam = np.random.beta(self.alpha, self.alpha)
            elif self.mixup_option == 'negative':
                idx2 = np.random.randint(0, len(self.negative_idx))
                idx2 = self.negative_idx[idx2]
                lam = np.random.beta(self.alpha, self.alpha)
                lam = max(lam, 1-lam) # negative noise is always weaker
            signal2, sub_signal2, target2 = self._get_signal_target(idx2)
            signal = lam * signal + (1 - lam) * signal2
            if sub_signal is not None:
                sub_signal = lam * sub_signal + (1 - lam) * sub_signal2
            if self.lor_label:
                target = target + target2 - target * target2
            else:
                target = lam * target + (1 - lam) * target2
                if self.hard_label:
                    target = (target > self.hard_label).float()
        if sub_signal is None:
            outputs = [signal, target]
        else:
            outputs = [signal, sub_signal, target]
        if self.return_index:
            outputs.append(torch.tensor(index))
        if self.return_test_index:
            outputs.append(torch.tensor(self.test_index[index]))
        return tuple(outputs)
        
    def _get_signal_target(self, index):
        path = self.paths[index]
        if self.cache is not None and path in self.cache.keys():
            signal = self.cache[path].copy()
        elif self.test_cache is not None and path in self.test_cache.keys():
            signal = self.test_cache[path].copy()
        else:
            signal = np.load(path).astype(np.float32)

        if self.norm_factor is not None: # DEPRECATED: normalization
            for ch in range(3):
                signal[ch] = signal[ch] / self.norm_factor[ch]

        if self.transforms is not None:
            signal1 = self.transforms(signal.copy())
        else:
            signal1 = signal
        if self.transforms2 is not None:
            signal2 = self.transforms2(signal)
        else:
            signal2 = None

        if isinstance(signal1, np.ndarray): # DEPRECATED: to tensor
            signal1 = torch.from_numpy(signal1).float()
        if signal2 is not None and isinstance(signal2, np.ndarray): # DEPRECATED: to tensor
            signal2 = torch.from_numpy(signal2).float()
        
        if self.spectr is not None: # DEPRECATED: spectrogram generation
            signal1 = self.spectr(signal1)
            
        if self.targets is not None:
            target = torch.tensor(self.targets[index]).unsqueeze(0).float()
        else:
            target = torch.tensor(0).unsqueeze(0).float()
        return signal1, signal2, target
