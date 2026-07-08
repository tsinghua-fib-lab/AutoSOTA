import torch
import numpy as np

from .utils import to_torch_and_device


class LatentPointCloud:
    """
    A point cloud representing the latent manifold with memory-efficient sequential loading.
    
    This class provides loading of point cloud data from disk using memory-mapped files,
    making it suitable for large datasets that may not fit entirely in memory.
    
    Args:
        path (str): Path to the .npy file containing point cloud data.
        reach (float, optional): Reach parameter of the latent manifold. Defaults to 1.0.
        dataset_size (int, optional): Number of points to use from the dataset. 
            If None, uses all available points.
        device (str, optional): Device to load tensors onto. Defaults to 'cpu'.
    Attributes:
        path (str): Path to the numpy file containing the point cloud data, (num_samples, dim)
        points (np.memmap): Memory-mapped array of point cloud coordinates.
        size (int): Number of points in the dataset.
        device (str): PyTorch device for tensor operations ('cpu' or 'cuda').
        reach (float): Reach parameter of the manifold, defaults to 1.0.
    
    """
    def __init__(self, path, reach=None, dataset_size=None, device='cpu'):
        
        self.path = path
        self.points = np.load(path, mmap_mode='r')

        if dataset_size is None:
            dataset_size = len(self.points) 
        self.size = dataset_size

        self.device = device
        if reach is None:
            reach = 1.
        self.reach = reach

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return to_torch_and_device(self.points[idx], self.device)
    
    def random_point(self, num_samples: int, device="cpu") -> torch.Tensor:
        points_idx = np.random.choice(range(len(self.points)), size=num_samples, replace=False)
        return to_torch_and_device(self.points[points_idx], device)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.random_point(num_samples, self.device)