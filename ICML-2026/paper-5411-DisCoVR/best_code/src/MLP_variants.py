"""The MLP variants module houses general MLP declarator classes and all mixins that make use of MLPs.

Most of the implemented models rely on MLPs with different output / forward functions to work.
To add distributions, simply inherit ForwardMixin and override the forward method. Then, to add the actual
MLP modeling the parameters for that distribution, inherit both the MLPMixin and the corresponding Mixin
from the distribution. NOTE : Mixins are not meant to be instantiated.
"""

import torch
import torch.nn as nn
from torch.nn.functional import softmax, softplus


class _MLPMixin:
    @staticmethod
    def _make_fc(dims, bias=True, extras='BN') -> nn.Sequential:
        """
        Helper to make FC layers in quick succession for hidden layers.

        Courtesy of:  https://pyro.ai/examples/scanvi.html

        Parameters
        ----------
        dims : array-like
            Array-like of :class:`int` specifying the sizes for layers. `dims[0], dims[-1]` are input and output respectively.

        bias : `bool`, default True
            Determines whether to have a bias term for inner `~torch.nn.Linear` layers.

        extras : str, default 'BN'
            One of None, 'BN' for BatchNorm and 'D' for dropout.

        Returns
        -------
        layers : :class:`~torch.nn.Sequential`
            The layers packed into a single module.
        """
        layers = []
        for in_dim, out_dim in zip(dims, dims[1:], strict=False):
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))

            match extras:
                case None:
                    pass

                case "BN":
                    layers.append(nn.BatchNorm1d(out_dim))

                case "D":
                    layers.append(nn.Dropout(p=0.1))

            layers.append(nn.ReLU())

        if extras is None:
            return nn.Sequential(*layers[:-1])

        return nn.Sequential(*layers[:-2])


class _ForwardMixin:
    @staticmethod
    def _split_in_half(arr) -> tuple:
        """Function to split any tensor in half across last dimension.

        Courtesy of:  https://pyro.ai/examples/scanvi.html

        Parameters
        ----------
        arr : :class:`~torch.Tensor`
            Tensor to be split

        Returns
        -------
        ts : :class:`tuple`
            Size 2 tuple of tensors that are the halves of the original :class:`~torch.Tensor`.
        """
        return arr.reshape(arr.shape[:-1] + (2, -1)).unbind(-2)

    @staticmethod
    def set_dims(in_dim, hidden_dims, out_dim):
        """Function to modify output dimensions to parameterize different distributions.

        Parameters
        ----------
        in_dim : :class:`int`
            Input dimension size.

        hidden_dims : :class:`list` of :class:`int`
            Dimension sizes for the hidden layers. Must explicitly be a list to facilitate addition.

        out_dim : :class:`int`
            Output dimension size.


        Returns
        -------
        dim_list : :class:`list` of :class:`int`
            Size 3 list of integers that contain layer dimensions.
        """
        return [in_dim] + hidden_dims + [out_dim]

    def forward(self, inputs):
        """Function to override when redefining forward.

        Parameters
        ----------
        inputs : :class:`~torch.Tensor`
            Input tensors to pass through network.

        Returns
        -------
        outputs : :class:`~torch.Tensor`
            Output after applying layer.
        """
        return self.layers(inputs)


class _NBMixin(_ForwardMixin):
    """Override for NB with softmax."""

    def forward(self, inputs):
        mu = self.layers(inputs)
        mu = softmax(mu, dim=-1)

        return mu


class _ZINBMixin(_ForwardMixin):
    """Override for ZINB with softmax."""

    @staticmethod
    def set_dims(in_dim, hidden_dims, out_dim):
        return [in_dim] + hidden_dims + [2 * out_dim]

    def forward(self, inputs):
        gate_logits, mu = self._split_in_half(self.layers(inputs))
        mu = softmax(mu, dim=-1)

        return gate_logits, mu


class _GaussianMixin(_ForwardMixin):
    """Override for multivariate normal distribution."""

    @staticmethod
    def set_dims(in_dim, hidden_dims, out_dim):
        return [in_dim] + hidden_dims + [2 * out_dim]

    def forward(self, inputs):
        _inputs = inputs.reshape(-1, inputs.size(-1))
        hidden = self.layers(_inputs)
        hidden = hidden.reshape(inputs.shape[:-1] + hidden.shape[-1:])

        loc, scale = self._split_in_half(hidden)
        scale = softplus(scale) + 1e-6

        return loc, scale


class _LognormalMixin(_ForwardMixin):
    """Override for lognormal distribution."""

    @staticmethod
    def set_dims(in_dim, hidden_dims, out_dim):
        return [in_dim] + hidden_dims + [2 * out_dim + 2]

    def forward(self, inputs):
        inputs = torch.log(1 + inputs)
        h1, h2 = self._split_in_half(self.layers(inputs))

        norm_loc, norm_scale = h1[..., :-1], softplus(h2[..., :-1]) + 1e-6
        l_loc, l_scale = h1[..., -1:], softplus(h2[..., -1:]) + 1e-6

        return norm_loc, norm_scale, l_loc, l_scale


class MLP(_ForwardMixin, _MLPMixin, nn.Module):
    """MLP base class. Can be instantiated.

    Does not parameterize any specific distribution necessarily. Outputs can have
    multiple interpretations depending on context

    Parameters
    ----------
        in_dim : :class:`int`
            Input dimension size.

        hidden_dims : :class:`list` of :class:`int`
            Dimension sizes for the hidden layers. Must explicitly be a list to facilitate addition.

        out_dim : :class:`int`
            Output dimension size.

        bias : `bool`, default True
            Determines whether to have a bias term for inner `~torch.nn.Linear` layers.


    Attributes
    ----------
    fc : :class:`~pandas.DataFrame`
        Dataframe object for reference.

    num_cols : :class:`int`
        Number of columns for `df_view`.
    """

    def __init__(
        self,
        in_dim,
        hidden_dims,
        out_dim,
        bias=True,
        extras='BN',
    ):
        nn.Module.__init__(self)
        self.dims = _ForwardMixin.set_dims(in_dim, hidden_dims, out_dim)
        self.layers = self._make_fc(
            self.set_dims(in_dim, hidden_dims, out_dim),
            bias=bias,
            extras=extras,
        )


class LognormalMLP(_LognormalMixin, MLP):
    """MLP parameterezing lognormal distribution."""

    pass


class GaussianMLP(_GaussianMixin, MLP):
    """MLP parameterizing multivariate normal distribution."""

    pass


class ZINBMLP(_ZINBMixin, MLP):
    """MLP parameterezing ZINB distribution."""

    pass

class NBMLP(_NBMixin, MLP):
    """MLP parameterezing NB distribution."""

    pass

    
