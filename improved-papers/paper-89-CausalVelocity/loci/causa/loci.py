import numpy as np
import torch
from torch import nn
from .utils import TensorDataLoader
from sklearn.preprocessing import StandardScaler, SplineTransformer

from .hsic import HSIC, jnp_HSIC 
from .het_ridge import convex_fgls
from .ml import map_optimization


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class HetSpindlyHead(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(1, 1, bias=False)
        self.lin2 = nn.Linear(1, 1, bias=False)
        self.lin2.weight.data.fill_(0.0)

    def forward(self, input):
        out1 = self.lin1(input[:, 0].unsqueeze(-1))
        out2 = torch.exp(self.lin2(input[:, 1].unsqueeze(-1)))
        return torch.cat([out1, out2], 1)


def build_het_network(in_dim=1):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 2),
        HetSpindlyHead()
    )


def test_indep_fgls(Phi, Psi, x, y, w_1, w_2):
    eta_1 = Phi @ w_1
    eta_2 = - torch.abs(Psi) @ w_2
    scale = torch.sqrt(- 0.5 / eta_2)
    loc = - 0.5 * eta_1 / eta_2
    residuals = (y.flatten() - loc) / scale
    dhsic_res = HSIC(residuals.flatten().cpu().numpy(), x)
    return dhsic_res


def test_indep_nn(model, x, y):
    y = y.flatten()
    with torch.no_grad():
        f = model(x)
        eta_2 =  - f[:, 1] / 2
        scale = torch.sqrt(- 0.5 / eta_2)
        loc = - 0.5 * f[:, 0] / eta_2
        residuals = (y - loc) / scale
    return HSIC(residuals.cpu().flatten().numpy(), x.cpu().flatten().numpy())

    
def het_fit_nn(x, y, n_steps=None, seed=711, device='cpu'):
    """Fit heteroscedastic noise model with convex estimator using neural network.
    More precisely we fit y = f(x) + g(x) N with N Gaussian noise and return a joint
    function for f and g.

    Returns
    -------
    log_lik : float
        log likelihood of the fit
    f : method
        method that takes vector of x values and returns mean and standard deviation. 
    """
    n_steps = 5000 if n_steps is None else n_steps
    x, y = torch.from_numpy(x).double(), torch.from_numpy(y).double()
    map_kwargs = dict(
        scheduler='cos',
        lr=1e-2,
        lr_min=1e-6,
        n_epochs=n_steps,
        nu_noise_init=0.5,
        prior_prec=0.0  # makes it maximum likelihood
    )
    loader = TensorDataLoader(
        x.reshape(-1, 1).to(device), y.flatten().to(device), batch_size=len(x)
    )
    set_seed(seed)
    model, losses, _, _, _ = map_optimization(
        build_het_network().to(device).double(),
        loader,
        likelihood='heteroscedastic_regression',
        **map_kwargs
    )

    @torch.no_grad()
    def f(x_):
        x_ = torch.from_numpy(x_[:, np.newaxis]).double()
        f = model(x_)
        eta_2 =  - f[:, 1] / 2
        scale = torch.sqrt(- 0.5 / eta_2)
        loc = - 0.5 * f[:, 0] / eta_2
        return loc.squeeze().cpu().numpy(), scale.squeeze().cpu().numpy()
    log_lik = - np.nanmin(losses) / len(x)
    return log_lik, f


def het_fit_convex(x, y, n_steps=None):
    """Fit heteroscedastic noise model with convex estimator using splines x -> y.
    More precisely we fit y = f(x) + g(x) N with N Gaussian noise and return a joint
    function for f and g.

    Returns
    -------
    f : method
       method that takes vector of x values and returns mean and standard deviation. 
    """
    n_steps = 1000 if n_steps is None else n_steps
    feature_map = SplineTransformer(n_knots=25, degree=5)
    Phi_x = torch.from_numpy(feature_map.fit_transform(x[:, np.newaxis])).double()
    y = torch.from_numpy(y).double()
    w_1, w_2, _, nll = convex_fgls(Phi_x, Phi_x.abs(), y, delta_Phi=1e-5, delta_Psi=1e-5, n_steps=n_steps)

    @torch.no_grad()
    def f(x_):
        Phi_x_ = torch.from_numpy(feature_map.transform(x_[:, np.newaxis])).double()
        eta_1 = Phi_x_ @ w_1
        eta_2 = - torch.abs(Phi_x_) @ w_2
        scale = torch.sqrt(- 0.5 / eta_2)
        loc = - 0.5 * eta_1 / eta_2
        return loc.squeeze().cpu().numpy(), scale.squeeze().cpu().numpy()

    return -nll, f

    
def loci(x, y, independence_test=True, neural_network=True, return_function=False, n_steps=None):
    """Location Scale Causal Inference (LOCI) for bivariate pairs. By default,
    the method returns a score for the x -> y causal direction where above 0
    indicates evidence for it and negative values indicate y -> x.

    Note: data x, y should be standardized or preprocessed in some way.
    
    Parameters
    ----------
    x : np.ndarray
        cause/effect vector 1-dimensional
    y : np.ndarray
        cause/effect vector 1-dimensional
    independence_test : bool, optional
        whether to run subsequent independence test of residuals, by default True
    neural_network : bool, optional
        whether to use neural network heteroscedastic estimator, by default True
    return_function : bool, optional
        whether to return functions to predict mean/std in both directions, by default False
    n_steps : int, optional
        number of epochs to train neural network or steps to optimize convex model
    """
    assert x.ndim == y.ndim == 1, 'x and y have to be 1-dimensional arrays'
    if neural_network:
        log_lik_forward, f_forward = het_fit_nn(x, y, n_steps)
        log_lik_reverse, f_reverse = het_fit_nn(y, x, n_steps)
    else:
        log_lik_forward, f_forward = het_fit_convex(x, y, n_steps)
        log_lik_reverse, f_reverse = het_fit_convex(y, x, n_steps)

    if independence_test:
        my, sy = f_forward(x)
        indep_forward = jnp_HSIC(x, (y - my) / sy)
        mx, sx = f_reverse(y)
        indep_reverse = jnp_HSIC(y, (x - mx) / sx)
        score = indep_reverse - indep_forward
    else:
        score = log_lik_forward - log_lik_reverse

    if return_function:
        return score, f_forward, f_reverse
    return score
