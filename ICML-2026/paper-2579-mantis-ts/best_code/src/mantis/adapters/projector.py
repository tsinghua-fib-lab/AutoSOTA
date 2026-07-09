import numpy as np

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection


class MultichannelProjector:
    """
    A generic class to use various scikit-learn dimension reduction methods to reduce 
    the number of channels in a multichannel time series dataset.
    
    Parameters
    ----------
    new_num_channels : int
        The number of channels after projection.
    patch_window_size : int, default=1
        The size of the patch window. By default, it is equal to 1, i.e., no patching.
    base_projector : str or object, default=None
        The base projector to use. Can be 'pca', 'svd', 'rand', or a custom projector that accepts
        `n_components` at the initialization and has `fit` and `transform` methods.
    """
    def __init__(self, new_num_channels, patch_window_size=1, base_projector=None):
        # init dimensions
        self.new_num_channels = new_num_channels
        self.patch_window_size = patch_window_size
        # init base projector 
        self.base_projector = base_projector
        n_components = patch_window_size * new_num_channels
        if base_projector in [None, 'pca']:
            self.base_projector_ = PCA(n_components=n_components)
        elif base_projector == 'svd':
            self.base_projector_ = TruncatedSVD(n_components=n_components)
        elif base_projector == 'rand':
            self.base_projector_ = SparseRandomProjection(n_components=n_components)
        # you can give your own base_projector with fit() and transform() methods, and it should have the argument `n_components`.
        else:
            self.base_projector_ = base_projector(n_components=n_components)

    def fit(self, x):
        x_transposed = np.swapaxes(x, 1, 2)
        
        num_samples, seq_len, num_channels = x_transposed.shape
        num_patches = seq_len // self.patch_window_size
        assert num_patches * self.patch_window_size == seq_len

        x_2d = x_transposed.reshape(num_samples * num_patches, self.patch_window_size * num_channels)
        return self.base_projector_.fit(x_2d)

    def transform(self, x):
        x_transposed = np.swapaxes(x, 1, 2)

        num_samples, seq_len, num_channels = x_transposed.shape
        num_patches = seq_len // self.patch_window_size
        assert num_patches * self.patch_window_size == seq_len

        x_2d = x_transposed.reshape(num_samples * num_patches, self.patch_window_size * num_channels)

        x_transformed = self.base_projector_.transform(x_2d)
        x_transformed = x_transformed.reshape([num_samples, seq_len, self.new_num_channels])
        return np.swapaxes(x_transformed, 1, 2)
