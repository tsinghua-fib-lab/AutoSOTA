import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import RealFakeDataset


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt, preprocess=None):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = RealFakeDataset(opt)
    
    if '2b' in opt.arch:
        dataset.transform = preprocess
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader
