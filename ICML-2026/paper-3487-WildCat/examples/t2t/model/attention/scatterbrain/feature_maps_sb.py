# Adapted from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/feature_maps/fourier_features.py
import math
import torch

from einops import rearrange

from ..projection_utils import gaussian_orthogonal_random_matrix


# from fast_transformers.feature_maps.base import FeatureMap
# For compatibility with our old code (using pytorch_fast_transformers 0.3.0), we copy the
# FeatureMap class here.
# Will be removed once we migrate to our new code
class FeatureMap(torch.nn.Module):
    """Define the FeatureMap interface."""
    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.
        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner


def softmax_kernel(data, *, projection_matrix, is_query, softmax_temp=None, eps=0., cosh=True,
                   return_log=False):
    """For key, we expect shape (..., S, D) where S is the sequence dimension
    return_log: return the log of the features (i.e. don't apply exp at the end).
    """
    if return_log and eps != 0:
        raise NotImplementedError('return_log is not compatible with nonzero eps')
    d = data.shape[-1]
    m = projection_matrix.shape[0] if not cosh else 2 * projection_matrix.shape[0]
    if softmax_temp is None:
        softmax_temp = 1 / math.sqrt(d)
    data_normalizer = math.sqrt(softmax_temp)
    projection_matrix = projection_matrix.type_as(data)
    data_dash = torch.einsum('...id,jd->...ij', data, data_normalizer * projection_matrix)
    diag_data = data.square().sum(dim=-1, keepdim=True) / 2 * (data_normalizer ** 2)
    if cosh:
        # We use the cosh feature map from the Performer paper, which effectively means
        # concatenating data_dash and -data_dash
        data_dash = torch.cat([data_dash, -data_dash], dim=-1)
    if is_query:
        log_scale = -diag_data + torch.amax(data_dash, dim=-1, keepdim=True) - math.log(m) / 2
        # TD: The correct scaling is torch.exp(data_dash - diag_data)
        data_dash_log = data_dash - torch.amax(data_dash, dim=-1, keepdim=True)
        if not return_log:
            data_dash = torch.exp(data_dash_log) + eps / math.sqrt(m)
    else:
        data_dash_m_diag = data_dash - diag_data - math.log(m) / 2
        log_scale = torch.amax(data_dash_m_diag, dim=(-1, -2), keepdim=True)
        data_dash_log = data_dash_m_diag - log_scale
        if not return_log:
            data_dash = torch.exp(data_dash_log) + eps / math.sqrt(m)
    return (data_dash if not return_log else data_dash_log).type_as(data), log_scale


class SBPerformerFeatures(FeatureMap):
    """Random Fourier Features for the RBF kernel according to [1].
    [1]: "Weighted Sums of Random Kitchen Sinks: Replacing minimization with
         randomization in learning" by A. Rahimi and Benjamin Recht.
    Arguments
    ---------
        query_dims: int, The input query dimensions in order to sample
                          the noise matrix
        n_features: int, The size of the feature map (should be divisible by 2)
                (default: query_dims)
        softmax_temp: float, The temerature for the Gaussian kernel
                      approximation exp(-t * |x-y|^2)
                      (default: 1/sqrt(query_dims))
        orthogonal: bool, When True the random matrix is initialized for
                    orthogonal random features to reduce the approximation
                    variance (default: False)
        redraw: int, Redraw the random matrix every 'redraw' times
                (default: 1)
        deterministic_eval: bool, Only redraw the random matrix during training
                            (default: False)
    """
    def __init__(self, query_dims, n_features=None, ortho_scaling=0, softmax_temp=None,
                 orthogonal=False, cosh=True, redraw=1, deterministic_eval=False, eps=0.0,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(query_dims)
        self.n_features = n_features or int(query_dims * math.log(query_dims))
        self.ortho_scaling = ortho_scaling
        # TODO: we're not using @orthogonal atm
        self.orthogonal = orthogonal
        self.softmax_temp = 1 / math.sqrt(query_dims) if softmax_temp is None else softmax_temp
        self.cosh = cosh
        # self.redraw = redraw
        # TODO: not redrawing atm, so I'm setting it to an irrational number
        self.redraw = math.pi
        self.deterministic_eval = deterministic_eval
        self.eps = eps  # Stabilizer for softmax kernel

        # Make a buffer for storing the sampled projection_matrix
        self.register_buffer("projection_matrix", torch.zeros(self.query_dims, self.n_features,
                                                              **factory_kwargs))
        self.factory_kwargs = factory_kwargs
        self._calls = -1

    def new_feature_map(self, device):
        # If we are not training skip the generation of a new feature map
        if self.deterministic_eval and not self.training:
            return

        # Only redraw the new feature map every self.redraw times
        self._calls += 1
        if (self._calls % self.redraw) != 0:
            return

        # We use the cosh feature map so the number of rows is halved
        nb_rows = self.n_features if not self.cosh else self.n_features // 2
        projection_matrix = gaussian_orthogonal_random_matrix(nrows=nb_rows,
                                                              ncols=self.query_dims,
                                                              scaling=self.ortho_scaling,
                                                              device=device,
                                                              dtype=self.factory_kwargs['dtype'])
        self.register_buffer("projection_matrix", projection_matrix)

    def forward_queries(self, x, return_log=False):
        return softmax_kernel(x, projection_matrix=self.projection_matrix, is_query=True,
                              softmax_temp=self.softmax_temp, eps=self.eps, cosh=self.cosh,
                              return_log=return_log)

    def forward_keys(self, x, return_log=False):
        return softmax_kernel(x, projection_matrix=self.projection_matrix, is_query=False,
                              softmax_temp=self.softmax_temp, eps=self.eps, cosh=self.cosh,
                              return_log=return_log)
