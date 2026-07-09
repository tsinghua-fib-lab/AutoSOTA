import numpy as np


class VarianceBasedSelector:
    """
    Perform a filter feature selection approach to reduce the number of channels in a multichannel time series data set.
    More specifically, it selects those channels that have the highest variance.

    Parameters
    ----------
    new_num_channels: int 
        The number of selected channels.
    """
    def __init__(self, new_num_channels):
        self.new_num_channels = new_num_channels
        self.support_ = None
        
    def fit(self, x):
        # flatten the tensor to 2D
        x_transposed = np.swapaxes(x, 1, 2)
        num_samples, seq_len, num_channels = x_transposed.shape
        x_2d = x_transposed.reshape(num_samples * seq_len, num_channels)

        # calculate variances and select top features
        variances = np.var(x_2d, axis=0)
        
        # select top features by variance
        self.support_ = np.argsort(variances)[::-1][:self.new_num_channels]
        return self.support_

    def transform(self, x):
        if self.support_ is None:
            raise RuntimeError("You must call fit at least once before calling transform.")

        # select features based on precomputed indices
        selected_features = x[:, self.support_, :]

        return selected_features
