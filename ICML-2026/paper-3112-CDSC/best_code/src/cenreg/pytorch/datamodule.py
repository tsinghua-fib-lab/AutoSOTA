import torch
from torch.utils.data import DataLoader


class ProbDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]

    def len_feature(self):
        return self.features.shape[1]


class ProbDataModule(torch.utils.data.Dataset):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self, features, targets):
        return DataLoader(
            ProbDataset(features, targets),
            batch_size=self.batch_size,
            num_workers=1,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self, features, targets):
        return DataLoader(
            ProbDataset(features, targets),
            batch_size=99999999,
            num_workers=1,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self, features, targets):
        return DataLoader(
            ProbDataset(features, targets),
            batch_size=99999999,
            num_workers=1,
            persistent_workers=True,
            shuffle=False,
        )


class SurvDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, features, observed_times, events):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.events = torch.tensor(events, dtype=torch.int64)
        self.observed_times = torch.tensor(observed_times, dtype=torch.float32)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[index], self.observed_times[index], self.events[index]

    def len_feature(self):
        return self.features.shape[1]


class SurvDataModule(torch.utils.data.Dataset):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self, features, observed_times, events):
        return DataLoader(
            SurvDataset(features, observed_times, events),
            batch_size=self.batch_size,
            num_workers=1,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self, features, observed_times, events):
        return DataLoader(
            SurvDataset(features, observed_times, events),
            batch_size=99999999,
            num_workers=1,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self, features, observed_times, events):
        return DataLoader(
            SurvDataset(features, observed_times, events),
            batch_size=99999999,
            num_workers=1,
            persistent_workers=True,
            shuffle=False,
        )


class IntervalCensoredDataset(torch.utils.data.Dataset):
    def __init__(self, features, lb, ub):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.lb = torch.tensor(lb, dtype=torch.float32)
        self.ub = torch.tensor(ub, dtype=torch.float32)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        if self.features.shape[0] == 0:
            return torch.empty(0, 0), torch.empty(0), torch.empty(0)
        else:
            return self.features[index], self.lb[index], self.ub[index]

    def len_feature(self):
        return self.features.shape[1]


class IntervalCensoredDataModule(torch.utils.data.Dataset):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self, features, lb, ub):
        return DataLoader(
            IntervalCensoredDataset(features, lb, ub),
            batch_size=self.batch_size,
            num_workers=1,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self, features, lb, ub):
        return DataLoader(
            IntervalCensoredDataset(features, lb, ub),
            batch_size=99999999,
            num_workers=1,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self, features, lb, ub):
        return DataLoader(
            IntervalCensoredDataset(features, lb, ub),
            batch_size=99999999,
            num_workers=1,
            persistent_workers=True,
            shuffle=False,
        )
