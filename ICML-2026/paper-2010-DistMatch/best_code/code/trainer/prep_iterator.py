import torch
from torch.utils.data import Dataset

from utils.calc_torch import unfold_window


class ManyToOneIterator:
    """
    Example: seq_len = 3
    - x0 x1 x2 | y2
    - x1 x2 x3 | y3
    """

    def __init__(self, datasets, seq_len, split_func, **kwargs):
        self._datasets = datasets
        self._feature_key = "x"
        self._target_key = "y"
        self._seq_len = seq_len
        self._split_func = split_func

    def dataset_to_sequences(self, dataset):
        X, Y = self._split_func(dataset)
        X, Y, X_target_shifted, = X[1:], Y[1:], Y[:-1]  # cut off first time step because not delta available

        # Prepare Sequences
        X = unfold_window(M=X, window_len=self._seq_len)
        Y = torch.unsqueeze(Y[self._seq_len - 1:], 1)
        X_target_shifted = unfold_window(M=X_target_shifted, window_len=self._seq_len)
        X = torch.concat((X, X_target_shifted), dim=2)
        assert X.shape[0] == Y.shape[0]
        return zip(X, Y)

    def __iter__(self):
        for dataset in self._datasets:
            for X, Y in self.dataset_to_sequences(dataset):
                yield {
                    self._feature_key: X.float(),  # TODO check if float is best?
                    self._target_key: Y.float()
                }
        return


class CompleteDataset(Dataset):
    def __init__(self, iterator: ManyToOneIterator):
        self._samples = [s for s in iterator]

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self):
        return len(self._samples)
