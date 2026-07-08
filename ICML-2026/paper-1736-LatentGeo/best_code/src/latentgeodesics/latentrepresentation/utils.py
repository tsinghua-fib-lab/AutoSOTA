import torch
from torch.utils.data import (
    DataLoader,
    BatchSampler,
    RandomSampler,
    SequentialSampler,
)

def to_torch_and_device(data, device="cpu"):
    """
    Convert data array to PyTorch float32 tensor and move to device
    """
    if not torch.is_tensor(data):
        data = torch.from_numpy(data).to(device=device, dtype=torch.float32)
    elif device is not None:
        data = data.to(device).to(torch.float32)
    return data


class MultiIndexDataLoader(DataLoader):
    """
    This DataLoader loads by passing multiple indices to __getitem__ function of
    dataset. To be used with finite,
    indexed datasets, for example test data.

    Inspiration:
    https://discuss.pytorch.org/t/how-to-use-batchsampler-with-getitem-dataset/78788/4
    """

    def __init__(
            self,
            dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            shuffle=True,
            num_samples=None
    ):
        if shuffle:
            sampler = BatchSampler(
                RandomSampler(dataset, num_samples=num_samples), batch_size=batch_size, drop_last=drop_last
            )
        else:
            sampler = BatchSampler(
                SequentialSampler(dataset), batch_size=batch_size, drop_last=drop_last
            )
        super().__init__(
            dataset=dataset,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            batch_size=None,
        )
        if not drop_last:
            if num_samples is None:
                self.num = len(dataset)
            else:
                self.num = num_samples
        else:
            self.num = len(self) * batch_size


def compute_perturbed_points(points, std):
    noise_vectors = to_torch_and_device(torch.normal(0, std, points.shape), points.device)
    return points + noise_vectors

def L2(x):
    """
    Sum over squared differences, divided by number of entries (but not divided by batch
    size).
    """
    return torch.mean(x**2) * x.shape[0]


