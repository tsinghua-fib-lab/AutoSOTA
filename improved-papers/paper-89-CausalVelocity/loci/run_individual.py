from os.path import exists
import numpy as np
import pandas as pd
import torch
from torch import nn
import gin
import argparse
import torch.backends.cudnn as cudnn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, SplineTransformer
from sklearn.linear_model import LinearRegression

from causa.datasets import AN, LS, MNU, SIMG, ANs, CausalDataset, Tuebingen, SIM, LSs
from causa.ml import map_optimization
from causa.heci import HECI
from causa.hsic import HSIC
from causa.het_ridge import convex_fgls
from causa.utils import TensorDataLoader
from causa.gnn_fast import GNN


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


def build_network(in_dim=1, out_dim=1):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, out_dim)
    )


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


def test_indep_fgls(Phi, Psi, x, y_true, w_1, w_2, convex=False):
    eta_1 = Phi @ w_1
    if convex:
        eta_2 = - torch.abs(Psi) @ w_2
    else:
        eta_2 = - 0.5 * torch.exp(Psi @ w_2)
    scale = torch.sqrt(- 0.5 / eta_2)
    loc = - 0.5 * eta_1 / eta_2
    residuals = (y_true.flatten() - loc) / scale
    dhsic_res = HSIC(residuals.flatten().cpu().numpy(), x)
    return dhsic_res


def test_indep_nn(model, x, y_true, mode='homo', nu_noise=None):
    # modes: 'homo', 'het', 'het_noexp'
    y_true = y_true.flatten()
    with torch.no_grad():
        f = model(x)
    if mode == 'homo':
        residuals = (f.flatten() - y_true)
    else:
        if mode == 'het':
            eta_2 = nu_noise * f[:, 1]
        elif mode == 'het_noexp':
            eta_2 = - 0.5 * torch.exp(f[:, 1])
        scale = torch.sqrt(- 0.5 / eta_2)
        loc = - 0.5 * f[:, 0] / eta_2
        residuals = (y_true - loc) / scale
    return HSIC(residuals.cpu().flatten().numpy(), x.cpu().flatten().numpy())


@gin.configurable
def experiment(experiment_name,
               pair_id: int,
               benchmark: CausalDataset=gin.REQUIRED,
               device: str='cpu',
               double: bool=True,
               seed=711,
               result_dir: str='results'):
    if exists(f'{result_dir}/{experiment_name}_{pair_id}.csv'):
        print('Run already completed. Aborting...')
        exit()

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    map_kwargs = dict(
        scheduler='cos',
        lr=1e-2,
        lr_min=1e-6,
        n_epochs=5000,
        nu_noise_init=0.5,
    )
    print(f'Experiment name: {experiment_name} pair {pair_id}')
    df = pd.DataFrame(index=[pair_id])
    df.index.name = 'pair_id'
    dataset = benchmark(pair_id, double=double)
    if double:
        torch.set_default_dtype(torch.float64)

    n_data = dataset.cause.shape[0]
    if dataset.cause.shape[1] > 1 or dataset.effect.shape[1] > 1:
        print('Skip dataset with pair_id', pair_id, 'because of multidimensionality')
        exit()
    df.loc[pair_id, 'n_data'] = int(n_data)

    # True direction
    d, k = dataset.cause.shape[1], dataset.effect.shape[1]
    batch_size = len(dataset.cause)
    print(f'd={d}, k={k}, n={batch_size}')

    # NOTE: all scores such that higher is better (log ml, log lik) gives score!
    x_true = dataset.cause.numpy().flatten()
    x_false = dataset.effect.numpy().flatten()

    # run CGNN baseline (batch size 200 runs under 1h for all pairs at least)
    # default of batch size -1 would crash and 1000 is too (up to 6h for some pairs)
    prob = GNN(verbose=False, batch_size=min(200, batch_size)).predict_proba((x_true, x_false))
    df.loc[pair_id, 'cgnn_true'] = prob
    df.loc[pair_id, 'cgnn_false'] = -prob

    # run HECI baseline
    heci_cause = x_true.tolist()
    heci_effect = x_false.tolist()
    _, score_true, score_false = HECI(heci_cause, heci_effect)
    df.loc[pair_id, 'heci_true'] = -score_true
    df.loc[pair_id, 'heci_false'] = -score_false

    # run linear models with spline feature maps
    spline_trans = SplineTransformer(n_knots=25, degree=5)
    Phi_true = spline_trans.fit_transform(dataset.cause.numpy())
    y_true = dataset.effect.numpy().flatten()
    Phi_false =spline_trans.fit_transform(dataset.effect.numpy())
    y_false = dataset.cause.numpy().flatten()

    # Linear Regression (ML)
    model = LinearRegression()
    model.fit(Phi_true, y_true)
    y_pred = torch.from_numpy(model.predict(Phi_true)).flatten()
    lik = torch.distributions.Normal(loc=y_pred, scale=torch.ones_like(y_pred))
    df.loc[pair_id, 'lin_ml_true'] = lik.log_prob(dataset.effect.flatten()).mean().item()
    residuals_true = (y_pred.numpy() - y_true).flatten()
    df.loc[pair_id, 'lin_ml_hsic_true'] = HSIC(residuals_true, x_true)

    model.fit(Phi_false, y_false)
    y_pred = torch.from_numpy(model.predict(Phi_false)).flatten()
    lik = torch.distributions.Normal(loc=y_pred, scale=torch.ones_like(y_pred))
    df.loc[pair_id, 'lin_ml_false'] = lik.log_prob(dataset.cause.flatten()).mean().item()
    residuals_false = (y_pred.numpy() - y_false).flatten()
    df.loc[pair_id, 'lin_ml_hsic_false'] = HSIC(residuals_false, x_false)

    # convert to torch
    Phi_true = torch.from_numpy(Phi_true).to(device)
    y_true = dataset.effect.to(device)
    Phi_false = torch.from_numpy(Phi_false).to(device)
    y_false = dataset.cause.to(device)

    # Convex Heteroscedastic Linear Regression (ML) LSNM estimator
    w_1, w_2, _, loglik = convex_fgls(Phi_true, Phi_true.abs(), y_true, delta_Phi=1e-5, delta_Psi=1e-5)
    df.loc[pair_id, 'lin_ml_hetconv_true'] = - loglik
    df.loc[pair_id, 'lin_ml_hetconv_hsic_true'] = test_indep_fgls(Phi_true, Phi_true, x_true, y_true, w_1, w_2, convex=True)

    w_1, w_2, _, loglik = convex_fgls(Phi_false, Phi_false.abs(), y_false, delta_Phi=1e-5, delta_Psi=1e-5)
    df.loc[pair_id, 'lin_ml_hetconv_false'] = - loglik
    df.loc[pair_id, 'lin_ml_hetconv_hsic_false'] = test_indep_fgls(Phi_false, Phi_false, x_false, y_false, w_1, w_2, convex=True)

    # Neural Network estimators
    ## Homoscedastic
    x_true, y_true = dataset.cause.to(device), dataset.effect.to(device)
    loader_ordered = TensorDataLoader(
        dataset.cause.to(device), dataset.effect.to(device), batch_size=batch_size
    )
    set_seed(seed)
    model, losses, _, _, _ = map_optimization(
        build_network(d, k).to(device),
        loader_ordered,
        likelihood='regression',
        prior_prec=0.0,
        **map_kwargs
    )
    df.loc[pair_id, 'nn_ml_true'] = - np.nanmin(losses) / n_data
    df.loc[pair_id, 'nn_ml_hsic_true'] = test_indep_nn(model, x_true, y_true, mode='homo')

    ## Heteroscedastic
    loader_ordered = TensorDataLoader(
        dataset.cause.to(device), dataset.effect.to(device).flatten(), batch_size=batch_size
    )
    set_seed(seed)
    model, losses, _, _, _ = map_optimization(
        build_het_network(d).to(device),
        loader_ordered,
        likelihood='heteroscedastic_regression',
        prior_prec=0.0,
        **map_kwargs
    )
    df.loc[pair_id, 'nn_ml_het_true'] = - np.nanmin(losses) / n_data
    df.loc[pair_id, 'nn_ml_het_hsic_true'] = test_indep_nn(model, x_true, y_true, mode='het', nu_noise=-0.5)

    # Backward direction
    ## Homoscedastic
    x_false, y_false = dataset.effect.to(device), dataset.cause.to(device)
    loader_reversed = TensorDataLoader(
        dataset.effect.to(device), dataset.cause.to(device), batch_size=batch_size
    )
    set_seed(seed)
    model, losses, _, _, _ = map_optimization(
        build_network(d, k).to(device),
        loader_reversed,
        likelihood='regression',
        prior_prec=0.0,
        **map_kwargs
    )
    df.loc[pair_id, 'nn_ml_false'] = - np.nanmin(losses) / n_data
    df.loc[pair_id, 'nn_ml_hsic_false'] = test_indep_nn(model, x_false, y_false, mode='homo')

    ## Heteroscedastic
    loader_reversed = TensorDataLoader(
        dataset.effect.to(device), dataset.cause.to(device).flatten(), batch_size=batch_size
    )
    set_seed(seed)
    model, losses, _, _, _ = map_optimization(
        build_het_network(d).to(device),
        loader_reversed,
        likelihood='heteroscedastic_regression',
        prior_prec=0.0,
        **map_kwargs
    )
    df.loc[pair_id, 'nn_ml_het_false'] = - np.nanmin(losses) / n_data
    df.loc[pair_id, 'nn_ml_het_hsic_false'] = test_indep_nn(model, x_false, y_false, mode='het', nu_noise=-0.5)
    print(df.loc[pair_id])

    df.to_csv(f'{result_dir}/{experiment_name}_{pair_id}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair_id', type=int, help='pair id to run')
    parser.add_argument('--config', type=str, help='gin-config for the run.')
    parser.add_argument('--result_dir', type=str, default='results')
    args = parser.parse_args()
    gin.external_configurable(StandardScaler)
    gin.external_configurable(MinMaxScaler)
    gin.parse_config_file(args.config)
    experiment(experiment_name=args.config.split('/')[-1].split('.')[0],
               pair_id=args.pair_id, double=True,
               result_dir=args.result_dir)
