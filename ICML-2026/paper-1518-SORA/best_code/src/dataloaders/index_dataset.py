import torch

class IndexDataset(torch.utils.data.Dataset):
    """
    A class to index the dataset.
    
    Args:
        base_dataset (torch.utils.data.Dataset): The base dataset to index.
    """
    def __init__(self, base_dataset):
        self.dataset = base_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, idx
