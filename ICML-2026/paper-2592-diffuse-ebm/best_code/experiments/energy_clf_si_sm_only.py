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
from diffclf.networks.ebm import SIEnergyDenoiserNet
from diffclf.networks.mlp import ImprovedFourierNet
from diffclf.networks.utils import init_bias_uniform_zeros, kaiming_uniform_zeros_
from diffclf.sde.utils import TimeSampler
from diffclf.si.stochastic_interpolant import SimpleStochasticInterpolant
from tqdm import trange

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--dim', type=int)
parser.add_argument('--dsm_weighting_type', type=str, default="square", choices=["uniform", "linear", "square"])
parser.add_argument('--n_levels', type=int, default=512)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--n_steps', type=int, default=10000)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

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
filename = 'energy_clf_si_sm_only'
filename += '_dim_' + str(args.dim)
filename += '_dsm_weighting_type_' + str(args.dsm_weighting_type)
filename += '_seed_{}.pkl'.format(args.seed)

# Build the distributions
target_0 = standardize_gauss(FourtyModesMOG(dim=args.dim)).to(device)
target_1 = standardize_gauss(TwoModes(dim=args.dim)).to(device)

# Build the train times and normalizing constants
times = torch.linspace(1e-3, 1.0-1e-3, args.n_levels)
time_sampler = TimeSampler(times=times).to(device)

# Build the SI
si = SimpleStochasticInterpolant(
    drift_net=None,
    denoiser_net=None,
).to(device)
gamma_fn = lambda t: torch.sqrt(t * (1. - t))

# Build the EBM
base_net = ImprovedFourierNet(
    dim=args.dim,
    dim_out=args.dim,
    num_layers=4,
    channels=64 if args.dim <= 32 else 256,
    last_bias_init=init_bias_uniform_zeros,
    last_weight_init=kaiming_uniform_zeros_,
    use_pos_embedding=True
)
add_net = ImprovedFourierNet(
    dim=args.dim,
    dim_out=1,
    num_layers=4,
    channels=64 if args.dim <= 32 else 256,
    last_bias_init=init_bias_uniform_zeros,
    last_weight_init=kaiming_uniform_zeros_,
    use_pos_embedding=True
)
ebm = SIEnergyDenoiserNet(
    base_net=base_net,
    add_net=add_net,
    gamma_type='brownian'
).to(device)

# Get the weighting function for DSM
if args.dsm_weighting_type == "uniform":
    weighting_func = lambda t: torch.ones_like(t)
elif args.dsm_weighting_type == "linear":
    weighting_func = lambda t: gamma_fn(t)
elif args.dsm_weighting_type == "square":
    weighting_func = lambda t: torch.square(gamma_fn(t))
else:
    raise NotImplementedError(f"Weighting type {args.dsm_weighting_type} not implemented!")

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

# Build the optimizer
optimizer = torch.optim.Adam(ebm.parameters(), lr=args.lr)
r = trange(args.n_steps)
for epoch in r:
    optimizer.zero_grad()
    x0 = target_0.sample((args.batch_size,))
    x1 = target_1.sample((args.batch_size,))
    loss = loss_fn_dsm(ebm, x0, x1, time_sampler).mean()
    loss.backward()
    optimizer.step()
    r.set_postfix(dsm_loss=loss.item())

# Move EBM to CPU
ebm = ebm.cpu()

# Save the results
with open(args.results_path + '/' + filename, 'wb') as f:
    pickle.dump({ 'config': config, 'ebm': ebm.state_dict() }, f)