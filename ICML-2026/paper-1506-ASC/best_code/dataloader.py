import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class SCPIDataset(Dataset):
    def __init__(self,
                 X,
                 y,
                 device=None):
        """
        Initialize the dataset.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            x_columns (list): List of column names for features (X)
            y_column (str): Column name for target variable (y)
        """
        assert X.shape[0] == y.shape[0], "X and y must have the same number of rows"
        self.device = device if device is not None else 'cpu'
        self.X = torch.Tensor(X).to(self.device)
        self.y = torch.Tensor(y).to(self.device)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'y': self.y[idx],
            'index': idx
        }

def create_dataloaders(dataset: SCPIDataset,
                       batch_size: int,
                       test_ratio=0.2,
                       val_ratio=0.1,
                       data_seed: int = 42,
                       num_workers: int = 1):
    assert 0.0 < test_ratio < 1.0, "Test ratio must be between 0 and 1"
    assert 0.0 < val_ratio < 1.0, "Validation ratio must be between 0 and 1"
    assert test_ratio + val_ratio < 1.0, "Sum of test and validation ratios must be less than 1"

    n = len(dataset)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)

    g = torch.Generator().manual_seed(data_seed)
    indices = torch.randperm(n, generator=g)

    test_idx = indices[:test_size]
    val_idx = indices[test_size:test_size + val_size]
    train_idx = indices[test_size + val_size:]

    train_sampler = SubsetRandomSampler(train_idx, generator=g)
    val_sampler = SubsetRandomSampler(val_idx, generator=g)
    test_sampler = SubsetRandomSampler(test_idx, generator=g)

    dl_train = DataLoader(dataset, batch_size=batch_size,
                            sampler=train_sampler, num_workers=num_workers)
    dl_val = DataLoader(dataset,
                        sampler=val_sampler,
                        batch_size=len(val_idx), num_workers=num_workers)
    dl_test = DataLoader(dataset,
                            sampler=test_sampler,
                            batch_size=len(test_idx), num_workers=num_workers)

    return dl_train, dl_val, dl_test