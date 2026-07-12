import random
import numpy as np
import torch


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def undo_preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y


def get_sample_batch(dataloader, n_samples, return_labels=False):
    """Get a batch of samples from a dataloader."""
    samples = []
    labels = []
    
    for batch_data in dataloader:
        if isinstance(batch_data, (list, tuple)):
            batch_images = batch_data[0]
            batch_labels = batch_data[1] if len(batch_data) > 1 else None
        else:
            batch_images = batch_data
            batch_labels = None
        
        for i in range(batch_images.shape[0]):
            if len(samples) >= n_samples:
                break
            samples.append(batch_images[i:i+1])
            if batch_labels is not None:
                labels.append(batch_labels[i:i+1])
        
        if len(samples) >= n_samples:
            break
    
    samples = torch.cat(samples[:n_samples], dim=0)
    
    if return_labels and labels:
        labels = torch.cat(labels[:n_samples], dim=0)
        return samples, labels
    else:
        return samples


