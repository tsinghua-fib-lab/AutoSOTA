# Libraries
import argparse
import math
import numpy as np
import os
import ot
import pickle
import pprint
import random
import torch
import matplotlib.pyplot as plt
from diffclf.distr.aldp import AlanineDipeptide
from diffclf.networks.ebm import SIEnergyDenoiserNet
from diffclf.networks.egnn import EGNN_atom
from diffclf.sde.utils import TimeSampler
from diffclf.si.stochastic_interpolant import SimpleStochasticInterpolant
from diffclf.utils.se3_utils import remove_mean
from tqdm import trange

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--vacuum_datapath', type=str)
parser.add_argument('--ckpt_filepath', type=str)
parser.add_argument('--vel_ckpt_filepath', type=str)
parser.add_argument('--loss_type', type=str)
parser.add_argument('--k', type=int, default=2)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--n_steps', type=int, default=25000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--batch_size_ot', type=int, default=512)
parser.add_argument('--batch_size_eval', type=int, default=1024)
parser.add_argument('--n_eval_samples', type=int, default=8192)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

# Load the checkpoint
with open(args.ckpt_filepath, 'rb') as f:
    # Load the data
    ckpt_data = pickle.load(f)
    # Parse the config
    n_layers = ckpt_data['config']['n_layers']
    hidden_nf = ckpt_data['config']['hidden_nf']
    n_levels = ckpt_data['config']['n_levels']
    dsm_weighting_type = ckpt_data['config']['dsm_weighting_type']
    gamma_factor = ckpt_data['config']['gamma_factor']
    if dsm_weighting_type != 'uniform':
        reg_val = gamma_factor**2
    else:
        reg_val = 1.0

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
filename = 'energy_clf_si_from_sm_aldp'
filename += '_dsm_weighting_type_' + dsm_weighting_type
filename += '_gamma_factor_{:.1e}'.format(gamma_factor)
filename += '_reg_val_{:.2e}'.format(reg_val)
filename += '_loss_' + args.loss_type
filename += '_k_' + str(args.k)
filename += '_seed_{}.pkl'.format(args.seed)

# Build the distributions
target_0 = AlanineDipeptide(args.data_path, env="implicit").to(device)
target_1 = AlanineDipeptide(args.data_path, env="vacuum").to(device)
target_1.load_data(remove_mean(
    torch.load(args.vacuum_datapath).view((-1, *target_1.data_shape))
))
data_shape = (target_0.n_particles, target_0.n_dimensions)
dim = target_0.n_particles * target_0.n_dimensions

# Build the train times and normalizing constants
times = torch.linspace(1e-3, 1.0-1e-3, n_levels, device=device)
time_sampler = TimeSampler(times=times).to(device)

# Build the SI
si = SimpleStochasticInterpolant(
    drift_net=None,
    denoiser_net=None,
    gamma_factor=gamma_factor
).to(device)
gamma_fn = lambda t: gamma_factor * torch.sqrt(t * (1. - t))

# Build the EBM
base_net = EGNN_atom(
    n_particles=target_0.n_particles,
    n_dimension=target_0.n_dimensions,
    atom_type_labels=target_0.get_atom_chemical_types(),
    bonds=target_0.get_bonds(),
    hidden_nf=hidden_nf,
    time_embedding_dim=128,
    atom_type_embedding_dim=64,
    n_layers=n_layers,
    recurrent=False,
    attention=True,
    tanh=False,
    use_pos_embedding=True
)
add_net = None
ebm = SIEnergyDenoiserNet(
    base_net=base_net,
    add_net=add_net,
    gamma_type='brownian',
    gamma_factor=gamma_factor,
    is_particles=True
)

# Load the parameters
ebm.load_state_dict(ckpt_data['ebm'])

# Wrap f into the model
class WrapF(torch.nn.Module):
    def __init__(self, base_ebm, n_levels):
        super().__init__()
        self.base_ebm = base_ebm
        self.f = torch.nn.Parameter(torch.zeros((n_levels,)))
    def denoiser(self, t, x):
        return self.base_ebm.denoiser(t, x)
    def score(self, t, x):
        return self.base_ebm.score(t, x)
    def log_prob_and_grad(self, t, x, return_denoiser=False):
        return self.base_ebm.log_prob_and_grad(t, x, return_denoiser=return_denoiser)
    def log_prob(self, t, x):
        return self.base_ebm.log_prob(t, x)

ebm = WrapF(ebm, n_levels).to(device)

# Get the weighting function for DSM
if dsm_weighting_type == "uniform":
    weighting_func = lambda t: 1.0
elif dsm_weighting_type == "linear":
    weighting_func = lambda t: gamma_fn(t)
elif dsm_weighting_type == "square":
    weighting_func = lambda t: torch.square(gamma_fn(t))
else:
    raise NotImplementedError(f"Weighting type {dsm_weighting_type} not implemented!")

def superpose_points(points1, points2):
    # Compute optimal rotation and translation
    M = torch.matmul(points1.transpose(-2, -1), points2)
    U, S, V = torch.svd(M)
    R = torch.matmul(U, V.transpose(-2, -1))
    # Apply rotation and translation to superpose points1 onto points2
    return torch.matmul(points1, R), points2

def sample_ot_coupling(x0, x1, n_samples=None):
    # Resample x0, x1 according to transport matrix
    a1, b1 = ot.unif(x0.size()[0]), ot.unif(x1.size()[0])
    M = torch.square(torch.cdist(x0, x1))
    M = M / M.max()
    pi = ot.emd(a1, b1, M.detach().cpu().numpy())
    # Sample random interpolations on pi
    p = pi.flatten()
    p = p / p.sum()
    choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p,
        size=x1.shape[0] if n_samples is None else n_samples)
    i, j = np.divmod(choices, pi.shape[1])
    return x0[i], x1[j]

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
    z = remove_mean(torch.randn_like(i_t))
    xt = i_t + gamma_fn(t) * z
    z_hat = remove_mean(ebm.denoiser(t, xt))
    loss = torch.sum(torch.square(z_hat - z) / weighting_func(t), dim=sum_indexes) / dim
    if antithetic:
        xt_neg = i_t - gamma_fn(t) * z
        z_hat_neg = remove_mean(ebm.denoiser(t, xt_neg))
        loss += torch.sum(torch.square(z_hat_neg + z) / weighting_func(t), dim=sum_indexes) / dim
        loss *= 0.5
    return loss

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
    zs = remove_mean(torch.randn_like(i_t_s))
    i_t_t = si.interpolant(t, x0, x1)
    zt = remove_mean(torch.randn_like(i_t_t))
    xs = i_t_s + gamma_fn(s) * zs
    xt = i_t_t + gamma_fn(t) * zt
    xst = torch.cat([xs, xt], dim=0)
    zst = torch.cat([zs, zt], dim=0)
    log_prob_xst_st, z_hat_st = ebm.log_prob_and_grad(torch.cat([s, t], dim=0), xst, return_denoiser=True)
    z_hat_st = remove_mean(z_hat_st)
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
        z_hat_neg = remove_mean(z_hat_neg)
        dsm_loss += torch.sum(
            torch.square(z_hat_neg + zst) / weighting_func(torch.cat([s, t], dim=0)),
            dim=sum_indexes
        ) / dim
        dsm_loss *= 0.5
    return dsm_loss, clf_loss

# Create an EMA model
ebm_ema = torch.optim.swa_utils.AveragedModel(
    model=ebm,
    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
    use_buffers=True
)

# Load the EMA parameters
ebm_ema.load_state_dict({
    k.replace('module.', 'module.ebm.') : v for k,v in ckpt_data['ebm_ema'].items()
}, strict=False)

# Build the optimizer
optimizer = torch.optim.Adam(ebm.parameters(), lr=args.lr)
r = trange(args.n_steps)
for epoch in r:
    optimizer.zero_grad()
    x0 = target_0.sample((args.batch_size_ot,))
    x1 = target_1.sample((args.batch_size_ot,))
    x0, x1 = superpose_points(x0, x1)
    x0, x1 = sample_ot_coupling(x0.view((-1, dim)), x1.view((-1, dim)),
        n_samples=args.batch_size)
    x0, x1 = x0.view((-1, *data_shape)), x1.view((-1, *data_shape))
    if args.loss_type == 'sm':
        x0 = x0.unsqueeze(0).repeat((args.k, 1, 1, 1)).view((-1, 22, 3))
        x1 = x1.unsqueeze(0).repeat((args.k, 1, 1, 1)).view((-1, 22, 3))
        losses = loss_fn_dsm(ebm, x0, x1, time_sampler)
    else:
        losses = loss_fn_bilevel(ebm, x0, x1, ebm.f, time_sampler)
    if isinstance(losses, tuple):
        loss_sm = losses[0].mean()
        loss_other = losses[1].mean()
        loss = reg_val * loss_sm + loss_other
    else:
        loss = losses.mean()
    loss.backward()
    optimizer.step()
    ebm_ema.update_parameters(ebm)
    with torch.no_grad():
        ebm.f -= ebm.f[-1].clone()
    if args.loss_type == 'sm':
        r.set_postfix(sm_loss=loss.item())
    else:
        r.set_postfix(loss=loss.item(), sm_loss_norm=loss_sm.item() * reg_val,
            sm_loss=loss_sm.item(), clf_loss=loss_other.item())

# Load the velocity checkpoint
with open(args.vel_ckpt_filepath, 'rb') as f:
    d_velocity = pickle.load(f)
    if gamma_factor != d_velocity['config']['gamma_factor']:
        print('Gamma factors not the same.')
    if n_levels != d_velocity['config']['n_levels']:
        print('Gamma factors not the same.')

# Build the net
velocity_fn = EGNN_atom(
    n_particles=target_0.n_particles,
    n_dimension=target_0.n_dimensions,
    atom_type_labels=target_0.get_atom_chemical_types(),
    bonds=target_0.get_bonds(),
    hidden_nf=d_velocity['config']['hidden_nf'],
    time_embedding_dim=128,
    atom_type_embedding_dim=64,
    n_layers=d_velocity['config']['n_layers'],
    recurrent=False,
    attention=True,
    tanh=False,
    use_pos_embedding=True
)

# Load the weights
velocity_fn.load_state_dict(d_velocity['net'])

# Move to device
velocity_fn = velocity_fn.to(device)

# Disable gradient with respect to the parameters
for p in ebm.parameters():
    p.requires_grad_(False)
for p in velocity_fn.parameters():
    p.requires_grad_(False)


# Make the plots dir
os.makedirs(f"{args.results_path}/plots", exist_ok=True)

# Evaluate both models
results = { 'ebm' : {}, 'ebm_ema' : {} }
for diff_val in [0.0, 1e-2, 1e-3]:
    results['ebm'][str(diff_val)] = None
    results['ebm_ema'][str(diff_val)] = None
n_total_samples = int(args.n_eval_samples / args.batch_size_eval)
t_ones = torch.ones((args.batch_size_eval, 1, 1), device=device)
for model_, model_name in [(ebm, 'ebm'), (ebm_ema.module, 'ebm_ema')]:
    # Build the drift function
    def drift_fn(t, x, return_grad=False):
        v = velocity_fn(t, x)
        s = model_.score(t, x)
        b = v - si.gamma_dot_times_gamma(t) * s
        if return_grad:
            return b, s
        else:
            return b
    for diff_val in [0.0, 1e-2, 1e-3]:
        print('# Sampling with diff_val = ', str(diff_val))
        # Collect the samples
        samples = []
        for i in range(n_total_samples):
            print('Evaluation loop {}/{}'.format(i+1, n_total_samples))
            x = target_0.sample((args.batch_size_eval,))
            for i in trange(n_levels-1):
                t_cur = t_ones * times[i]
                t_next = t_ones * times[i + 1]
                if diff_val > 0.0:
                    drift_cur, score_cur = drift_fn(t_cur, x, return_grad=True)
                    mean, var = si.forward_sde_kernel(t_cur, t_next, x, drift_cur, None, score_cur,
                        diff_val, return_mean_var=True)
                    x = mean + torch.sqrt(var) * remove_mean(torch.randn_like(mean))
                else:
                    x = si.ode_step(t_cur, t_next, x, drift_fn)
            samples.append(x.detach().cpu().clone())
        samples = torch.concat(samples, dim=0)
        # Compute the metrics
        results[model_name][str(diff_val)] = target_1.compute_metrics(samples)
        results[model_name][str(diff_val)]['log_prob'] = target_1.log_prob(samples)
        results[model_name][str(diff_val)]['samples'] = target_1.compute_psi_phi(samples)
        # Make plots
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        target_1.plot_samples(ax[0], samples, label="model")
        target_1.plot_samples(ax[1], target_1.sample((samples.shape[0], )).clone().detach().cpu(),
            label="ground truth")
        fig.savefig(args.results_path + "/plots/{}_{}_diff_val_{:.1e}.png".format(
            filename[:-4], model_name, diff_val
        ))

# Move everything to CPU
ebm = ebm.cpu()
ebm_ema = ebm_ema.cpu()

# Save the results
with open(args.results_path + '/' + filename, 'wb') as f:
    pickle.dump({ 'config': config, 'results' : results,
        'ebm': ebm.state_dict(), 'ebm_ema': ebm_ema.state_dict() }, f)
