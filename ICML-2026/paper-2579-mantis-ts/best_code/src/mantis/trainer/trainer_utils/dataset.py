import torch

from torch.utils.data import Dataset


class LabeledDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.to(torch.float32) if isinstance(
            x, torch.Tensor) else torch.FloatTensor(x)
        self.y = y.to(torch.long) if isinstance(
            x, torch.Tensor) else torch.LongTensor(y)

    def transform(self, x):
        return x.to(torch.float32) if isinstance(x, torch.Tensor) else torch.FloatTensor(x)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        examples = self.x[idx]
        labels = self.y[idx]
        return examples, labels


class UnlabeledDataset(Dataset):
    def __init__(self, x):
        self.x = x.to(torch.float32) if isinstance(
            x, torch.Tensor) else torch.FloatTensor(x)

    def transform(self, x):
        return x.to(torch.float32) if isinstance(x, torch.Tensor) else torch.FloatTensor(x)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        examples = self.x[idx]
        return {'data': examples}
