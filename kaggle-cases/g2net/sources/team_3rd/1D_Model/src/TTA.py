import math
from torch.utils.data import Dataset
import audiomentations as A
import torch
import numpy as np


class TTA(Dataset):
    def __init__(self, paths, targets, use_raw_wave=True, vflip=False, shuffle_channels=False,
                 time_shift=False, tsl=96, tsr=96,
                 add_gaussian_noise=False, time_stretch=False, shuffle01=False,timemask=False,
                 shift_channel=False,reduce_SNR=False):
        self.paths = paths
        self.targets = targets
        self.use_raw_wave = use_raw_wave
        self.vflip = vflip
        self.shuffle_channels = shuffle_channels
        self.time_shift = time_shift
        self.tsl = tsl # time shift left
        self.tsr = tsr # time shift right
        self.add_gaussian_noise = add_gaussian_noise
        self.time_stretch = time_stretch
        self.shuffle01 = shuffle01
        self.timemask = timemask
        self.shift_channel = shift_channel
        self.reduce_SNR = reduce_SNR
        if time_shift:
            self.time_shift = A.Shift(min_fraction=-tsl * 1.0 / 4096, max_fraction=tsr / 4096, p=1, rollover=False)
        if add_gaussian_noise:
            self.gaussian_noise = A.AddGaussianNoise(min_amplitude=0.001 * 0.015, max_amplitude=0.015 * 0.015, p=1)
        if timemask:
            self.timemask = A.TimeMask(min_band_part=0.0, max_band_part=0.03, fade=False, p=1.0)
        if time_stretch:
            self.time_stretch = A.TimeStretch(min_rate=0.9, max_rate=1.111, leave_length_unchanged=True, p=1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        waves = np.load(path)

        if self.vflip:
            waves = -waves
        if self.shuffle_channels:
            np.random.shuffle(waves)
        if self.time_shift:
            waves = self.time_shift(waves, sample_rate=2048)
        if self.add_gaussian_noise:
            waves = self.gaussian_noise(waves, sample_rate=2048)
        if self.time_stretch:
            waves = self.time_stretch(waves, sample_rate=2048)
        if self.shuffle01:
            waves[[0, 1]] = waves[[1, 0]]
        if self.timemask:
            waves = self.timemask(waves, sample_rate=2048)

        waves = torch.FloatTensor(waves * 1e20) if self.use_raw_wave else torch.FloatTensor(waves)
        target = torch.tensor(self.targets[index], dtype=torch.float)  # device=device,
        return waves, target


