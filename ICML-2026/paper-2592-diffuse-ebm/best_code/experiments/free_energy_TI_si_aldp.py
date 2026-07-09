# Libraries
import argparse
import math
import numpy as np
import os
import pickle
import pprint
import random
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from diffclf.distr.aldp import AlanineDipeptide
from diffclf.networks.ebm import EBM, SIEnergyDenoiserNet
from diffclf.networks.egnn import EGNN_atom
from diffclf.smc.pdds import pdds_sampler
from diffclf.si.stochastic_interpolant import SimpleStochasticInterpolant
from diffclf.utils.se3_utils import remove_mean

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--vacuum_datapath', type=str)
parser.add_argument('--ckpt_filepath', type=str)
parser.add_argument('--n_particles', type=int, default=8192)
parser.add_argument('--tmin', type=float, default=1e-3)
parser.add_argument('--tmax', type=float, default=0.999)
parser.add_argument('--use_ema', action=argparse.BooleanOptionalAction)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

# Load the checkpoint
with open((args.ckpt_filepath), 'rb') as f:
	d = pickle.load(f)
	loss_type = d['config']['loss_type']

# Load the pre-trained score checkpoint
with open((d['config']['ckpt_filepath']), 'rb') as f:
    # Load the data
    d_score = pickle.load(f)
    # Parse the config
    n_layers_score = d_score['config']['n_layers']
    hidden_nf_score = d_score['config']['hidden_nf']
    n_levels = d_score['config']['n_levels']
    if 'factorize_tsm' in d_score['config']:
        factorize_tsm = bool(d_score['config']['factorize_tsm'])
    else:
        factorize_tsm = False
    if 'tsm_weighting_type' in d_score['config']:
        tsm_weighting_type = d_score['config']['tsm_weighting_type']
    else:
        tsm_weighting_type = None
    if 'tsm_t_limit' in d_score['config']:
        tsm_t_limit = d_score['config']['tsm_t_limit']
    else:
        tsm_t_limit = None
    dsm_weighting_type = d_score['config']['dsm_weighting_type']
    gamma_factor = d_score['config']['gamma_factor']

# Load the pre-trained score checkpoint
with open((d['config']['vel_ckpt_filepath']), 'rb') as f:
	d_vel = pickle.load(f)
	n_layers_vel = d_vel['config']['n_layers']
	hidden_nf_vel = d_vel['config']['hidden_nf']

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
filename = 'recalib_si_aldp'
filename += '_dsm_weighting_type_' + dsm_weighting_type
if tsm_weighting_type is not None:
    filename += '_tsm_weighting_type_' + tsm_weighting_type
if tsm_t_limit is not None:
    filename += '_tsm_t_limit_{:.2e}'.format(tsm_t_limit)
if factorize_tsm:
    filename += '_factorized'
filename += '_gamma_factor_{:.1e}'.format(gamma_factor)
filename += '_loss_' + loss_type
if args.use_ema:
    filename += '_use_ema'
filename += '_seed_{}'.format(args.seed)

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
    hidden_nf=hidden_nf_score,
    time_embedding_dim=128,
    atom_type_embedding_dim=64,
    n_layers=n_layers_score,
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

# Wrap f into the model
class WrapF(torch.nn.Module):
    def __init__(self, base_ebm, n_levels):
        super().__init__()
        self.base_ebm = base_ebm
        self.f = torch.nn.Parameter(torch.zeros((n_levels,)))
        if 'with_target' in loss_type:
            self.f0 = torch.nn.Parameter(torch.zeros(1,))
            self.f1 = torch.nn.Parameter(torch.zeros(1,))
    def denoiser(self, t, x):
        return self.base_ebm.denoiser(t, x)
    def score(self, t, x):
        return self.base_ebm.score(t, x)
    def log_prob_and_grad(self, t, x, return_denoiser=False):
        return self.base_ebm.log_prob_and_grad(t, x, return_denoiser=return_denoiser)
    def log_prob(self, t, x):
        return self.base_ebm.log_prob(t, x)
    def energy(self, t, x):
        return self.base_ebm.energy(t, x)
ebm = WrapF(ebm, n_levels).to(device)

# Load the parameters
if args.use_ema:
	ebm.load_state_dict({
		k.replace('module.','') : v for k,v in d['ebm_ema'].items() if 'module.' in k
	})
else:
	ebm.load_state_dict(d['ebm'])

# Build the net
velocity_fn = EGNN_atom(
    n_particles=target_0.n_particles,
    n_dimension=target_0.n_dimensions,
    atom_type_labels=target_0.get_atom_chemical_types(),
    bonds=target_0.get_bonds(),
    hidden_nf=hidden_nf_vel,
    time_embedding_dim=128,
    atom_type_embedding_dim=64,
    n_layers=n_layers_vel,
    recurrent=False,
    attention=True,
    tanh=False,
    use_pos_embedding=True
)

# Load the weights
velocity_fn.load_state_dict(d_vel['net'])

# Move to device
velocity_fn = velocity_fn.to(device)

# Disable gradient with respect to the parameters
for p in ebm.parameters():
    p.requires_grad_(False)
for p in velocity_fn.parameters():
    p.requires_grad_(False)


def percentile_inlier_mask(x, y, low=1.0, high=99.0):
    qx = torch.quantile(x, torch.tensor([low/100, high/100]))
    qy = torch.quantile(y, torch.tensor([low/100, high/100]))
    return (x >= qx[0]) & (x <= qx[1]) & (y >= qy[0]) & (y <= qy[1])

os.makedirs(f"{args.results_path}/plots-free-energy", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
samples_0 = target_0.sample((args.n_particles,))
samples_1 = target_1.sample((args.n_particles,))
t_ones = torch.ones((args.n_particles, 1, 1), device=device)
learned_logp_0 = ebm.log_prob(t_ones * times[0], samples_0).detach().cpu()
learned_logp_1 = ebm.log_prob(t_ones * times[-1], samples_1).detach().cpu()
target_0_logp = target_0.log_prob(samples_0).detach().cpu()
target_1_logp = target_1.log_prob(samples_1).detach().cpu()

# Mask out some outlier
mask0 = percentile_inlier_mask(target_0_logp, learned_logp_0, low=1, high=99)
mask1 = percentile_inlier_mask(target_1_logp, learned_logp_1, low=1, high=99)
target_0_logp = target_0_logp[mask0]
learned_logp_0 = learned_logp_0[mask0]
target_1_logp = target_1_logp[mask1]
learned_logp_1 = learned_logp_1[mask1]

def r2_score(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.flatten(); y = y.flatten()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = vx.pow(2).sum() * vy.pow(2).sum()
    if denom == 0:
        return torch.tensor(1.0 if (vx.abs().sum()==0 and vy.abs().sum()==0) else 0.0,
                            dtype=y.dtype, device=y.device)
    return (vx.dot(vy) ** 2) / denom

r2_0 = r2_score(target_0_logp, learned_logp_0)
r2_1 = r2_score(target_1_logp, learned_logp_1)

ax[0].scatter(target_0_logp, learned_logp_0, s=6, alpha=0.5)
ax[0].set_xlabel("Target 0")
ax[0].set_ylabel("learned")
ax[0].set_title(f"R2: {r2_0:.3f}")
ax[1].scatter(target_1_logp, learned_logp_1, s=6, alpha=0.5)
ax[1].set_xlabel("Target 1")
ax[1].set_title(f"R2: {r2_1:.3f}")

def free_energy_ti(ebm, ts, samples_per_level, samples_0, samples_1):
    dF = 0.0
    n_samples_per_level = samples_per_level.shape[1]
    data_shape = samples_per_level.shape[2:]
    data_ones = (1,) * len(data_shape)
    t_ones = torch.ones((n_samples_per_level, *data_ones), device=device)

    bar = trange(len(ts) - 1)
    for i in bar:
        t = ts[i] * t_ones
        tp1 = ts[i + 1] * t_ones
        log_prob_t = ebm.log_prob(t, samples_per_level[i])
        log_prob_tp1 = ebm.log_prob(tp1, samples_per_level[i])
        grad_t_times_dt = log_prob_tp1 - log_prob_t
        dF -= grad_t_times_dt.mean().item()
        bar.set_description(f"dF: {dF:.3f}")

    dF_target_0 = math.log(n_samples_per_level) - torch.logsumexp(
        -target_0.log_prob(samples_0) + ebm.log_prob(t_ones * ts[0], samples_0), dim=0
    ).item()

    dF_target_1 = -math.log(n_samples_per_level) + torch.logsumexp(
        -target_1.log_prob(samples_1) + ebm.log_prob(t_ones * ts[-1], samples_1), dim=0
    ).item()
    res = dF_target_0 + dF + dF_target_1
    print(f'free-energy-TI: {res:.3f} <--> dF_target_0: {dF_target_0:.3f}, dF_target_1: {dF_target_1:.3f}, dF: {dF:.3f}')
    return res

# Reference forward-FEP value precomputed once with 250k target_0 samples:
# dF_fep = log(N) - logsumexp(-U_target_0(x) + U_target_1(x))
dF_fep = 22.419

tmin = args.tmin
tmax = args.tmax
min_idx = (times - tmin).abs().argmin().item()
max_idx = (times - tmax).abs().argmin().item()
data_shape_ones = (1,) * len(data_shape)
eval_times = times[min_idx:max_idx+1]
all_samples = torch.empty((len(eval_times), args.n_particles, *data_shape), device=device)
samples_0 = target_0.sample((args.n_particles,))
samples_1 = target_1.sample((args.n_particles,))
with torch.no_grad():
    for i, t in enumerate(eval_times):
        t_batch = torch.full((samples_0.shape[0], *data_shape_ones), t, device=device)
        all_samples[i] = si.sample(t_batch, samples_0, samples_1, return_z=False)

filename += '_tmin_{:.2e}_tmax_{:.2e}_n_particles_{}'.format(args.tmin, args.tmax, args.n_particles)
dF = free_energy_ti(ebm, eval_times, all_samples, samples_0, samples_1)
fig.suptitle(f"Free-energy-TI: {dF:.3f}; free-energy-FEP: {dF_fep:.3f}")
os.makedirs(f"{args.results_path}/plots-free-energy", exist_ok=True)
fig.savefig(args.results_path + f"/plots-free-energy/{filename}.png")

os.makedirs(f"{args.results_path}/free-energy", exist_ok=True)
res = {
    'dF_TI': dF,
    'dF_FEP': dF_fep,
    'r2_0': r2_0,
    'r2_1': r2_1,
}
with open(args.results_path + f"/free-energy/{filename}.pkl", 'wb') as f:
    pickle.dump(res, f)