import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as distribution
import math
import numpy as np
import time




class FMVGC(nn.Sequential):
    """ 
        Flow Matching Based Vector Gaussian copula
    """
    def __init__(self, n_inputs, bs):
        super().__init__()
        self.maf1 = NFM(n_inputs, bs)
        self.maf2 = NFM(n_inputs, bs)
        
        self.d_in = n_inputs

    def forward(self, x, y):
        xx, _ = self.maf1.forward(x)
        yy, _ = self.maf2.forward(y)
        return xx, yy
    
    def sample(self, n=100, mode='marginal', device='cuda:0'):
        assert mode in ['marginal']
        if mode == 'marginal':
            x, _ = self.maf1.sample(n, device)
            y, _ = self.maf2.sample(n, device)
        if mode == 'joint':
            x = 0
            y = 0
        return x, y
        
    def learn(self, x, y):
        n, d = x.size()
        # learn f, g
        self.maf1.learn(x)
        self.maf2.learn(y)
        with torch.no_grad():
            xx, yy = self.forward(x, y)
        # learn the inner Gaussian 
        self.mu, self.V = self.empirical_params(xx, yy)
        self.mu2, self.V2 = self.mu.clone(), torch.eye(2*d).to(x.device)
        self.Vx, self.mx = self.V[0:d, 0:d], self.mu[0:d]
        self.Vy, self.my = self.V[d:, d:], self.mu[d:]
        self.normal = distribution.multivariate_normal.MultivariateNormal(self.mu, self.V)
        self.normal2 = distribution.multivariate_normal.MultivariateNormal(self.mu2, self.V2)
        self.normal_x = distribution.multivariate_normal.MultivariateNormal(self.mx, self.Vx)
        self.normal_y = distribution.multivariate_normal.MultivariateNormal(self.my, self.Vy)
        return 
    
    def empirical_params(self, x, y):
        z = torch.cat([x, y], dim=1)
        n, d = z.size()
        mu = z.mean(dim=0, keepdim=True)
        V = (z-mu).t() @ (z-mu)/(n+1)
        # Regularize to ensure positive definiteness (critical for high-dim)
        V = V + 1e-5 * torch.eye(d, device=V.device, dtype=V.dtype)
        return mu.view(-1), V
    
    def KL_joint_marginal(self, x, y, inner=True):                                # E[log q(x ,y)/q(x)q(y)]
        if inner is False:
            x, y = self.forward(x, y)
        xy = torch.cat([x, y], dim=1)
        log_copula_density_xy = self.normal.log_prob(xy)
        log_copula_density_x = self.normal_x.log_prob(x)
        log_copula_density_y = self.normal_y.log_prob(y)       
        mi = log_copula_density_xy - log_copula_density_x - log_copula_density_y
        return mi.mean()
    
    def MI(self, x, y):
        xy = torch.cat([x, y], dim=1)
        n, d = xy.size()
        log_copula_density_xy = self.normal.log_prob(xy)
        log_copula_density_x = self.normal_x.log_prob(x)
        log_copula_density_y = self.normal_y.log_prob(y)       
        mi = log_copula_density_xy - log_copula_density_x - log_copula_density_y
        return mi.mean().item()
        
    
    def print(self):
        print('mu=', self.mu)
        print('V=',  (self.V*100).int()/100.0)

        
        




# -------------------------------------------------------------------------------------------- #




import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Callable, List, Type, Union

import optimizer


class NFM(nn.Module):
    """ 
        Neural flow matching
    """
    def __init__(self, n_inputs, bs):

        super().__init__()

        model_param = {
            'd_in': n_inputs,
            'rtdl_params': {'d_layers': [256, 512, 512, 256], 'dropout': 0.0}
            #'rtdl_params': {'d_layers': [1024, 2048, 2048, 1024], 'dropout': 0.0}
           
        }
        self.model = MLPFlow(**model_param)
        self.cfm = ConditionalFlowMatcher()

        self.max_iteration = 3000
        self.lr = 1e-3
        self.bs = bs
        self.d_in = n_inputs

    def forward(self, x):
        """ Transfer x to the latent space """
        latent = self.cfm.transform_from_data_2_latent(self.model, x, N=500, solver='euler', device=x.device)
        return latent, None
    
    def sample(self, n=1000, device='cuda:0'):
        latent = torch.randn(n, self.d_in).to(device)
        data = self.cfm.transform_from_latent_2_data(z0=latent, N=1000, model=self.model, solver='euler', device=device)
        return data, None
        
    def objective_func(self, x1, y=None):
        x0 = torch.randn_like(x1)
        t, xt, ut = self.cfm.sample_location_and_conditional_flow(x0, x1)
        vt = self.model(xt, t)
        loss = (vt - ut) ** 2
        return -loss.mean(-1).mean()

    def learn(self, inputs, cond_inputs=None):
        self.cond = False
        return optimizer.NNOptimizer.learn(self, x=inputs, y=0*inputs)


##### Conditional Flow Matching #####

def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, float):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

class BaseFlow():
    def __init__(self, pred_x1=False):
        self.N = 1000
        self.pred_x1 = pred_x1

    @torch.no_grad()
    def transform_from_latent_2_data(self, z0, N, model, solver='euler', device='cuda:0'):
        assert solver in ['euler']
        
        model.eval()
        z = z0.detach().clone()
        
        # z0 ~ N(0; I)
        dt = 1. / N

        for i in range(N-1):
            t = torch.as_tensor(i / N).to(z.device)
            t_next = torch.as_tensor((i + 1) / N).to(z.device)
            vt = model(z, t)            
            z = z.detach().clone() + vt * dt
            
        return z
        

    @torch.no_grad()
    def transform_from_data_2_latent(self, model, data, N=200, solver='euler', device='cuda:0'):
        assert solver in ['euler', 'heun']
        model.eval()

        z1 = data.to(device)

        dt = -1. / N
        z = z1.detach().clone()
        for i in reversed(range(N)):
            t = torch.as_tensor(i / N).to(z.device)
            t_next = torch.as_tensor((i - 1) / N).to(z.device)
            vt = model(z, t)
            if solver == 'heun' and i > 0:
                z_next = z.detach().clone() + vt * dt
                vt_next = model(z_next, t_next)
                vt = 0.5 * (vt + vt_next)

            z = z.detach().clone() + vt * dt

        return z

class ConditionalFlowMatcher(BaseFlow):
    """Base class for conditional flow matching methods. This class implements the independent
    conditional flow matching methods from [1] and serves as a parent class for all other flow
    matching methods.

    It implements:
    - Drawing data from gaussian probability path N(t * x1 + (1 - t) * x0, sigma) function
    - conditional flow matching ut(x1|x0) = x1 - x0
    - score function $\nabla log p_t(x|x0, x1)$
    
    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """

    def __init__(self, sigma: float=0.0, pred_x1: bool=False):
        r"""Initialize the ConditionalFlowMatcher class. It requires the hyper-parameter $\sigma$.

        Parameters
        ----------
        sigma : float
        """
        super().__init__(pred_x1)
        self.sigma = sigma

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1 + (1 - t) * x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t, xt
        return x1 - x0

    def sample_location_and_conditional_flow(self, x0, x1, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) eps: Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = torch.rand(x0.shape[0]).type_as(x0)
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if self.pred_x1:
            ut = ut + x0    # ut = x1

        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut


#####################################


##### MLP Flow NN #####

ModuleType = Union[str, Callable[..., nn.Module]]

class MLPFlow(nn.Module):
    def __init__(self, d_in, rtdl_params, dim_t=1024):
        super().__init__()
        self.dim_t = dim_t

        rtdl_params['d_in'] = dim_t
        rtdl_params['d_out'] = d_in

        self.mlp = MLP.make_baseline(**rtdl_params)
        
        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            #nn.SiLU(),
            nn.Softplus(),
            nn.Linear(dim_t, dim_t)
        )
    
    def forward(self, x, timesteps):
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(x.shape[0])
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        x = self.proj(x) + emb

        return self.mlp(x)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)

def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)

class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)

def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    return (
        (
            ReGLU()
            if module_type == 'ReGLU'
            else GEGLU()
            if module_type == 'GEGLU'
            else getattr(nn, module_type)(*args)
        )
        if isinstance(module_type, str)
        else module_type(*args)
    )

class MLP(nn.Module):
    """The MLP model used in [gorishniy2021revisiting].

    The following scheme describes the architecture:

    .. code-block:: text

          MLP: (in) -> Block -> ... -> Block -> Linear -> (out)
        Block: (in) -> Linear -> Activation -> Dropout -> (out)

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            module = MLP.make_baseline(x.shape[1], [3, 5], 0.1, 1)
            assert module(x).shape == (len(x), 1)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `MLP`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: ModuleType,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
        self,
        *,
        d_in: int,
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        activation: Union[str, Callable[[], nn.Module]],
        d_out: int,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)
        assert activation not in ['ReGLU', 'GEGLU']

        self.blocks = nn.ModuleList(
            [
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(
        cls: Type['MLP'],
        d_in: int,
        d_layers: List[int],
        dropout: float,
        d_out: int,
    ) -> 'MLP':
        """Create a "baseline" `MLP`.

        This variation of MLP was used in [gorishniy2021revisiting]. Features:

        * :code:`Activation` = :code:`ReLU`
        * all linear layers except for the first one and the last one are of the same dimension
        * the dropout rate is the same for all dropout layers

        Args:
            d_in: the input size
            d_layers: the dimensions of the linear layers. If there are more than two
                layers, then all of them except for the first and the last ones must
                have the same dimension. Valid examples: :code:`[]`, :code:`[8]`,
                :code:`[8, 16]`, :code:`[2, 2, 2, 2]`, :code:`[1, 2, 2, 4]`. Invalid
                example: :code:`[1, 2, 3, 4]`.
            dropout: the dropout rate for all hidden layers
            d_out: the output size
        Returns:
            MLP

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        assert isinstance(dropout, float)
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                'if d_layers contains more than two elements, then'
                ' all elements except for the first and the last ones must be equal.'
            )
        return MLP(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x

######################