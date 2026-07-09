# Libraries
import argparse
import math
import numpy as np
import os
import pickle
import pprint
import random
import torch
from diffclf.distr.gauss import TwoModes, FourtyModesMOG, standardize_gauss
from diffclf.distr.utils import log_prob_and_grad_mog
from diffclf.networks.ebm import SIEnergyDenoiserNet
from diffclf.networks.mlp import ImprovedFourierNet
from diffclf.networks.utils import init_bias_uniform_zeros, kaiming_uniform_zeros_
from diffclf.sde.utils import TimeSampler
from diffclf.si.stochastic_interpolant import SimpleStochasticInterpolant
from tqdm import trange

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--ckpt_filepath', type=str)
parser.add_argument('--loss_type', type=str)
parser.add_argument('--k', type=int)
parser.add_argument('--reg_val', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--n_steps', type=int, default=50000)
parser.add_argument('--n_eval_samples', type=int, default=4096)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

# Load the checkpoint
with open(args.ckpt_filepath, 'rb') as f:
    # Load the data
    ckpt_data = pickle.load(f)
    # Parse the config
    dim = ckpt_data['config']['dim']
    n_levels = ckpt_data['config']['n_levels']
    dsm_weighting_type = ckpt_data['config']['dsm_weighting_type']

# Save the arguments in a dictionnary
config = vars(args)

# Print the configuration
pprint.pprint(config)

# Make a Pytorch device
device = torch.device('cuda')

# Set the seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Make the results folder
os.makedirs(args.results_path, exist_ok=True)

# Make a filename
filename = 'energy_clf_si_from_sm'
filename += '_dim_' + str(dim)
filename += '_loss_' + args.loss_type
filename += '_k_' + str(args.k)
filename += '_seed_{}.pkl'.format(args.seed)

# Build the distributions
target_0 = standardize_gauss(FourtyModesMOG(dim=dim)).to(device)
target_1 = standardize_gauss(TwoModes(dim=dim)).to(device)

# Build the train times and normalizing constants
times = torch.linspace(1e-3, 1.0-1e-3, n_levels)
time_sampler = TimeSampler(times=times).to(device)
f = torch.zeros((n_levels,), requires_grad=True, device=device)

# Build the SI
si = SimpleStochasticInterpolant(
    drift_net=None,
    denoiser_net=None,
).to(device)
gamma_fn = lambda t: torch.sqrt(t * (1. - t))
gamma_dot_fn = lambda t: 0.5 / gamma_fn(t) * (1. - 2 * t)

# Build the EBM
base_net = ImprovedFourierNet(
    dim=dim,
    dim_out=dim,
    num_layers=4,
    channels=64 if dim <= 32 else 256,
    last_bias_init=init_bias_uniform_zeros,
    last_weight_init=kaiming_uniform_zeros_,
    use_pos_embedding=True
)
add_net = ImprovedFourierNet(
    dim=dim,
    dim_out=1,
    num_layers=4,
    channels=64 if dim <= 32 else 256,
    last_bias_init=init_bias_uniform_zeros,
    last_weight_init=kaiming_uniform_zeros_,
    use_pos_embedding=True
)
ebm = SIEnergyDenoiserNet(
    base_net=base_net,
    add_net=add_net,
    gamma_type='brownian'
).cpu()

# Load the parameters
ebm.load_state_dict(ckpt_data['ebm'])

# Move to device
ebm = ebm.to(device)

class OptimalEBM(torch.nn.Module):
    """Build the optimal EBM"""
    def __init__(self, gmm0, gmm1, gamma_fn):
        super().__init__()
        self.gmm0 = gmm0
        self.gmm1 = gmm1
        self.gamma_fn = gamma_fn
        self.log_prob_and_grad_ = torch.vmap(
            lambda t, x, return_denoiser : tuple(y.squeeze(0).squeeze(0)
                for y in self.log_prob_and_grad_scalar(t, x.unsqueeze(0), return_denoiser)),
            in_dims=(0, 0, None)
        )
    def get_params(self, t):
        dim = self.gmm0.means.shape[-1]
        weights_t = (self.gmm0.weights.unsqueeze(1) * self.gmm1.weights.unsqueeze(0)).flatten()
        means_t = ((1. - t) * self.gmm0.means.unsqueeze(1) + t * self.gmm1.means.unsqueeze(0)).view((-1, dim))
        variances_t = (torch.square(1 - t) * self.gmm0.variances.unsqueeze(1) \
            + torch.square(t) * self.gmm1.variances.unsqueeze(0) + torch.square(self.gamma_fn(t))).view((-1, dim))
        return weights_t, means_t, variances_t
    def log_prob(self, t, x):
        return self.log_prob_and_grad(t, x)[0]
    def log_prob_and_grad_scalar(self, t, x, return_denoiser):
        weights_t, means_t, variances_t = self.get_params(t)
        weights_t = weights_t / weights_t.sum()
        log_prob, grad = log_prob_and_grad_mog(x, weights_t, means_t, variances_t, return_log_prob=True)
        if return_denoiser:
            return log_prob, -self.gamma_fn(t) * grad
        else:
            return log_prob, grad
    def log_prob_and_grad(self, t, x, return_denoiser=False):
        return self.log_prob_and_grad_(t, x, return_denoiser)
    def score(self, t, x):
        return self.log_prob_and_grad(t, x)[1]
    def denoiser(self, t, x):
        return self.log_prob_and_grad(t, x, return_denoiser=True)[1]

# Instantiate the optimal EBM
opt_ebm = OptimalEBM(target_0, target_1, gamma_fn).to(device)
opt_f = torch.zeros((n_levels,), device=device)

# Get the weighting function for DSM
if dsm_weighting_type == "uniform":
    weighting_func = lambda t: torch.ones_like(t)
elif dsm_weighting_type == "linear":
    weighting_func = lambda t: gamma_fn(t)
elif dsm_weighting_type == "square":
    weighting_func = lambda t: torch.square(gamma_fn(t))
else:
    raise NotImplementedError(f"Weighting type {dsm_weighting_type} not implemented!")

def loss_fn_dsm(ebm, x0, x1, time_sampler, antithetic=True):
    """Denoising Score Matching loss for SI"""
    # Get the shapes
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    dim = math.prod(data_shape)
    sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
    # Build the times
    t = time_sampler.sample((x0.shape[0],)).view((-1, *data_shape_ones))
    # Compute the loss
    i_t = si.interpolant(t, x0, x1)
    z = torch.randn_like(i_t)
    xt = i_t + gamma_fn(t) * z
    z_hat = ebm.denoiser(t, xt)
    loss = torch.sum(torch.square(z_hat - z) / weighting_func(t), dim=sum_indexes) / dim
    if antithetic:
        xt_neg = i_t - gamma_fn(t) * z
        z_hat_neg = ebm.denoiser(t, xt_neg)
        loss += torch.sum(torch.square(z_hat_neg + z) / weighting_func(t), dim=sum_indexes) / dim
        loss *= 0.5
    return loss

def loss_fn_tsm(ebm, x0, x1, time_sampler, antithetic=True):
    """Conditional Time Score Matching loss for SI"""
    # Get the shapes
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    dim = math.prod(data_shape)
    sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
    # Build the times
    t = time_sampler.sample((x0.shape[0],)).view((-1, *data_shape_ones))
    # Compute the loss
    i_t = si.interpolant(t, x0, x1)
    z = torch.randn_like(i_t)
    xt = i_t + gamma_fn(t) * z
    _, z_hat, time_score = ebm.log_prob_and_grad_and_dot(t, xt, return_denoiser=True)
    dsm_loss = torch.sum(torch.square(z_hat - z) / weighting_func(t), dim=sum_indexes) / dim
    cond_time_score = gamma_dot_fn(t) / gamma_fn(t) * (torch.square(z).sum(dim=-1, keepdim=True) - dim) \
        - ((x1 - x0) * z).sum(dim=sum_indexes, keepdim=True) / gamma_fn(t)
    tsm_loss = (torch.square(time_score - cond_time_score) * gamma_fn(t)).flatten()
    if antithetic:
        xt_neg = i_t - gamma_fn(t) * z
        _, z_hat_neg, time_score_neg = ebm.log_prob_and_grad_and_dot(t, xt_neg, return_denoiser=True)
        dsm_loss += torch.sum(torch.square(z_hat_neg + z) / weighting_func(t), dim=sum_indexes) / dim
        dsm_loss *= 0.5
        cond_time_score_neg = gamma_dot_fn(t) / gamma_fn(t) * (torch.square(z).sum(dim=-1, keepdim=True) - dim) \
            + ((x1 - x0) * z).sum(dim=sum_indexes, keepdim=True) / gamma_fn(t)
        tsm_loss += (torch.square(time_score_neg - cond_time_score_neg) * gamma_fn(t)).flatten()
        tsm_loss *= 0.5

    return dsm_loss, tsm_loss

def loss_fn_bilevel(ebm, x0, x1, f, time_sampler, antithetic=True):
    # Get the shapes
    batch_size = x0.shape[0]
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    dim = math.prod(data_shape)
    sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
    # Sample the consecutive levels
    st, st_ind = time_sampler.sample((batch_size, 2), return_idx=True)
    st = st.view((batch_size, 2, *data_shape_ones))
    s, t = st[:,0], st[:,1]
    s_ind, t_ind = st_ind[:,0], st_ind[:,1]
    f_t, f_s = f[t_ind], f[s_ind]
    i_t_s = si.interpolant(s, x0, x1)
    zs = torch.randn_like(i_t_s)
    i_t_t = si.interpolant(t, x0, x1)
    zt = torch.randn_like(i_t_t)
    xs = i_t_s + gamma_fn(s) * zs
    xt = i_t_t + gamma_fn(t) * zt
    xst = torch.cat([xs, xt], dim=0)
    zst = torch.cat([zs, zt], dim=0)
    log_prob_xst_st, z_hat_st = ebm.log_prob_and_grad(torch.cat([s, t], dim=0), xst, return_denoiser=True)
    log_prob_xs_s = log_prob_xst_st[:batch_size]
    log_prob_xt_t = log_prob_xst_st[-batch_size:]
    log_prob_xs_t = ebm.log_prob(t, xs)
    log_prob_xt_s = ebm.log_prob(s, xt)
    clf_loss = 0.5 * torch.nn.functional.softplus(log_prob_xs_t - f_t - log_prob_xs_s + f_s)
    clf_loss += 0.5 * torch.nn.functional.softplus(log_prob_xt_s - f_s - log_prob_xt_t + f_t)
    dsm_loss = torch.sum(
        torch.square(z_hat_st - zst) / weighting_func(torch.cat([s, t], dim=0)),
        dim=sum_indexes
    ) / dim
    if antithetic:
        xs_neg = i_t_s - gamma_fn(s) * zs
        xt_neg = i_t_t - gamma_fn(t) * zt
        z_hat_neg = ebm.denoiser(torch.cat([s, t], dim=0), torch.cat([xs_neg, xt_neg], dim=0))
        dsm_loss += torch.sum(
            torch.square(z_hat_neg + zst) / weighting_func(torch.cat([s, t], dim=0)),
            dim=sum_indexes
        ) / dim
        dsm_loss *= 0.5
    return dsm_loss, clf_loss

def loss_fn_multilevel(ebm, x0, x1, f, k, i_idx, j_idx, diag_mask, time_sampler, antithetic=True):
    # Get the shapes
    batch_size = x0.shape[0]
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    dim = math.prod(data_shape)
    sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
    # Sample the times
    ts, idx = time_sampler.sample((batch_size, k), return_idx=True, unique=True)
    ts = ts.view((batch_size, k, *data_shape_ones))
    # Simulate the noising prcess
    i_t = si.interpolant(ts, x0.unsqueeze(1), x1.unsqueeze(1))
    zt = torch.randn_like(i_t)
    xt = i_t + gamma_fn(ts) * zt
    xt = xt.view((-1, *data_shape))
    zt = zt.view((-1, *data_shape))
    log_prob_xt_t, zt_hat = ebm.log_prob_and_grad(
        ts.view((-1, *data_shape_ones)), xt,
        return_denoiser=True
    )
    xt = xt.view((batch_size, k, *data_shape))
    log_prob_xt_t = log_prob_xt_t.view((batch_size, k))
    log_prob_xt_t -= f[idx]
    ts = ts.view((batch_size, k, *data_shape_ones))
    ts_ij = ts[:, i_idx[~diag_mask]]
    xt_ij = xt[:, j_idx[~diag_mask], :]
    f_i = f[idx[:, i_idx[~diag_mask]]]
    log_prob_no_diag = ebm.log_prob(
        ts_ij.view((-1, *data_shape_ones)), xt_ij.view((-1, *data_shape))
    )
    log_prob_no_diag = log_prob_no_diag.view((batch_size, k-1, k)) - f_i.view((batch_size, k-1, k))
    diag_mask = diag_mask.view((k, k))
    log_prob = torch.zeros((batch_size, k, k), device=x0.device)
    log_prob[:, diag_mask] = log_prob_xt_t
    log_prob[:, ~diag_mask] = log_prob_no_diag.view((batch_size, -1))
    log_prob_lse = torch.logsumexp(log_prob, dim=1)
    clf_loss = -(log_prob_xt_t - log_prob_lse)
    dsm_loss = torch.sum(
        torch.square(zt_hat - zt) / weighting_func(ts.view((-1, *data_shape_ones))),
        dim=sum_indexes
    ) / dim
    if antithetic:
        ts = ts.view((-1, *data_shape_ones))
        xt_neg = i_t.view((-1, *data_shape)) - gamma_fn(ts) * zt
        zt_hat_neg = ebm.denoiser(ts, xt_neg)
        dsm_loss += torch.sum(
            torch.square(zt_hat_neg + zt) / weighting_func(ts),
            dim=sum_indexes
        ) / dim
        dsm_loss *= 0.5
    return dsm_loss, clf_loss

# Precompute the grid
i_idx, j_idx = torch.meshgrid(
    torch.arange(args.k, device=device),
    torch.arange(args.k, device=device),
    indexing='ij'
)
i_idx = i_idx.reshape(-1)
j_idx = j_idx.reshape(-1)
diag_mask = torch.eye(args.k, dtype=torch.bool, device=device).flatten()

def compute_loss_multi_level(ebm, x0, x1, f):
    """Compute the multi-level loss"""
    # Get the shapes
    batch_size = x0.shape[0]
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    # Simulate the noising prcess
    n_levels = times.shape[0]
    xt = torch.empty((n_levels, *x0.shape), device=x0.device)
    xt[0] = x0
    for k in range(1, n_levels):
        xt[k] = si.sample(times[k], x0, x1)
    t_ones = torch.ones((batch_size, *data_shape_ones), device=x0.device)
    neg_en = torch.empty((n_levels, n_levels, batch_size), device=x0.device)
    with torch.no_grad():
        for i in range(n_levels):
            for j in range(n_levels):
                neg_en[i, j] = ebm.log_prob(t_ones * times[i], xt[j]) - f[i]
    log_sum_exp = torch.logsumexp(neg_en, dim=0)
    # Compute the loss
    arr = torch.arange(n_levels, device=x0.device)
    return -torch.mean(neg_en[arr, arr] - log_sum_exp, dim=0)

def compute_fisher_divergence(ebm, batch_size=args.n_eval_samples):
    """Compute the fisher divergence"""
    losses = torch.empty((n_levels,), device=device)
    t_ones = torch.ones((batch_size, 1), device=device)
    for i, t in enumerate(times):
        # Get the noisy sample
        t_ = t_ones * times[i]
        x = si.sample(t_, target_0.sample((args.n_eval_samples,)), target_1.sample((args.n_eval_samples,)))
        # Compute the square norm
        losses[i] = torch.sum(torch.square(ebm.score(t_, x) - opt_ebm.score(t_, x)), dim=-1).mean() / dim
    return losses.cpu()

def compute_ebm_ess(ebm, n_particles=args.n_eval_samples):
    """Compute the ESS at every time"""
    esses = torch.empty((times.shape[0],))
    t_ones = torch.ones((n_particles, 1), device=device)
    for i, t in enumerate(times):
        # Get the time
        t_ = t * t_ones
        # Sample the marginal exactly
        x = si.sample(t_, target_0.sample((args.n_eval_samples,)), target_1.sample((args.n_eval_samples,)))
        # Evaluate the marginal likelihood
        log_prob_proposal_x = opt_ebm.log_prob(t_, x)
        # Evaluate the EBM
        with torch.no_grad():
            log_prob_target_x = ebm.log_prob(t_, x)
        # Compute the ess
        log_weights = log_prob_target_x - log_prob_proposal_x
        esses[i] = torch.exp(2. * torch.logsumexp(log_weights, dim=0)
                             - torch.logsumexp(2. * log_weights, dim=0))
        esses[i] /= n_particles
    return esses.cpu()

def compute_log_probs(ebm, samples_per_level):
    """Evaluate the EBM on samples"""
    n_levels, batch_size = samples_per_level.shape[:2]
    log_probs = torch.empty((n_levels, batch_size), device=device)
    t_ones = torch.ones((batch_size, 1), device=device)
    for i, t in enumerate(times):
        # Get the time
        t_ = t * t_ones
        # Compute the log_prob
        with torch.no_grad():
            log_probs[i] = ebm.log_prob(t_, samples_per_level[i])
    return log_probs.cpu()

def r2_score(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the R2 score"""
    x = x.flatten(); y = y.flatten()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = vx.pow(2).sum() * vy.pow(2).sum()
    if denom == 0:
        return torch.tensor(1.0 if (vx.abs().sum()==0 and vy.abs().sum()==0) else 0.0,
                            dtype=y.dtype, device=y.device)
    return (vx.dot(vy) ** 2) / denom

# Build the optimizer
optimizer = torch.optim.Adam((*ebm.parameters(), f), lr=args.lr)
r = trange(args.n_steps)
for epoch in r:
    optimizer.zero_grad()
    x0 = target_0.sample((args.batch_size,))
    x1 = target_1.sample((args.batch_size,))
    if args.loss_type in ['sm','tsm']:
        x0 = x0.unsqueeze(0).repeat((args.k, 1, 1)).view((-1, dim))
        x1 = x1.unsqueeze(0).repeat((args.k, 1, 1)).view((-1, dim))
        if args.loss_type == 'sm':
            losses = loss_fn_dsm(ebm, x0, x1, time_sampler)
        else:
            losses = loss_fn_tsm(ebm, x0, x1, time_sampler)
    elif args.loss_type == 'bi_level':
        losses = loss_fn_bilevel(ebm, x0, x1, f, time_sampler)
    else:
        losses = loss_fn_multilevel(ebm, x0, x1, f, args.k, i_idx, j_idx,
            diag_mask, time_sampler)
    if isinstance(losses, tuple):
        loss_sm = losses[0].mean()
        loss_other = losses[1].mean()
        loss = loss_sm + args.reg_val * loss_other
    else:
        loss = losses.mean()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        f -= f[-1].clone()
    if args.loss_type == 'sm':
        r.set_postfix(sm_loss=loss.item())
    elif args.loss_type == 'tsm':
        r.set_postfix(loss=loss.item(), sm_loss=loss_sm.item(), tsm_loss=loss_other.item())
    else:
        r.set_postfix(loss=loss.item(), sm_loss=loss_sm.item(), clf_loss=loss_other.item())

# Disable gradient with respect to the parameters
for p in ebm.parameters():
    p.requires_grad_(False)
f = f.detach()

# Sample the target
x_0 = target_0.sample((args.n_eval_samples,))
x_1 = target_1.sample((args.n_eval_samples,))

# Sample the model at each level
samples_per_level = torch.empty((n_levels, args.n_eval_samples, dim), device=device)
t_ones = torch.ones((args.n_eval_samples, 1), device=device)
with torch.no_grad():
    for i, t in enumerate(times):
        samples_per_level[i] = si.sample(t * t_ones, x_0, x_1, return_z=False)

# Compute the metrics
metrics = {
    'multi_classif': compute_loss_multi_level(ebm, x_0, x_1, f).mean().cpu().item(),
    'fisher': compute_fisher_divergence(ebm),
    'ess': compute_ebm_ess(ebm)
}
log_probs = {
    'opt' : compute_log_probs(opt_ebm, samples_per_level),
    'model' : compute_log_probs(ebm, samples_per_level),
}
log_probs['r2'] = r2_score(log_probs['model'], log_probs['opt'])
r2 = []
for i in range(n_levels):
    r2.append(r2_score(log_probs['model'][i], log_probs['opt'][i]).item())
log_probs['r2_per_level'] = r2

# Save the results
with open(args.results_path + '/' + filename, 'wb') as f:
    pickle.dump({ 'config': config, 'metrics': metrics, 'log_probs': log_probs }, f)