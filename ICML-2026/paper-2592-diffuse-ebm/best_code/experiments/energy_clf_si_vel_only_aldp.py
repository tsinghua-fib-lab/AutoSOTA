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
from diffclf.distr.aldp import AlanineDipeptide
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
parser.add_argument('--gamma_factor', type=float, default=0.1)
parser.add_argument('--n_levels', type=int, default=512)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--batch_size_ot', type=int, default=1024)
parser.add_argument('--n_steps', type=int, default=50000)
parser.add_argument('--hidden_nf', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=4)
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
filename = 'energy_clf_si_vel_only_aldp'
filename += '_gamma_factor_{:.1e}'.format(args.gamma_factor)
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
times = torch.linspace(1e-3, 1.0-1e-3, args.n_levels)
time_sampler = TimeSampler(times=times).to(device)

# Build the SI
gamma_factor = args.gamma_factor
si = SimpleStochasticInterpolant(
    drift_net=None,
    denoiser_net=None,
    gamma_factor=gamma_factor
).to(device)
gamma_fn = lambda t: gamma_factor * torch.sqrt(t * (1. - t))

# Build the net
net = EGNN_atom(
    n_particles=target_0.n_particles,
    n_dimension=target_0.n_dimensions,
    atom_type_labels=target_0.get_atom_chemical_types(),
    bonds=target_0.get_bonds(),
    hidden_nf=args.hidden_nf,
    time_embedding_dim=128,
    atom_type_embedding_dim=64,
    n_layers=args.n_layers,
    recurrent=False,
    attention=True,
    tanh=False,
    use_pos_embedding=True
).to(device)

def superpose_points(points1, points2):
    # Compute optimal rotation and translation
    M = torch.matmul(points1.transpose(-2, -1), points2)
    U, S, V = torch.svd(M)
    R = torch.matmul(U, V.transpose(-2, -1))
    # Fix improper rotation (det(R) = -1) by flipping last column of Vh
    detR = torch.linalg.det(R)
    need_flip = detR < 0
    if need_flip.any():
        V_adj = V.clone()
        V_adj[need_flip, :, -1] *= -1
        R = torch.matmul(U, V_adj.transpose(-2, -1))
    # Apply rotation and translation to superpose points1 onto points2
    return torch.matmul(points1, R), points2

def sample_ot_coupling(x0, x1, n_samples=None, epsilon=1e-8):
    # Resample x0, x1 according to transport matrix
    a1, b1 = ot.unif(x0.size()[0]), ot.unif(x1.size()[0])
    M = torch.square(torch.cdist(x0, x1))
    M = M / (M.max() + epsilon)
    pi = ot.emd(a1, b1, M.detach().cpu().numpy())
    # Sample random interpolations on pi
    p = pi.flatten()
    p = p / p.sum()
    choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p,
        size=x1.shape[0] if n_samples is None else n_samples)
    i, j = np.divmod(choices, pi.shape[1])
    return x0[i], x1[j]

def loss_fn_velocity(net, x0, x1, time_sampler, antithetic=True):
    """Flow Matching loss for SI"""
    # Get the shapes
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    dim = math.prod(data_shape)
    sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
    # Build the times
    t = time_sampler.sample((x0.shape[0],)).view((-1, *data_shape_ones))
    # Compute the loss
    i_t = si.interpolant(t, x0, x1)
    i_t_dot = si.I_dot(t, x0, x1)
    z = remove_mean(torch.randn_like(i_t))
    xt = i_t + gamma_fn(t) * z
    b_hat = remove_mean(net(t, xt))
    loss = torch.sum(torch.square(b_hat - i_t_dot), dim=sum_indexes) / dim
    if antithetic:
        xt_neg = i_t - gamma_fn(t) * z
        b_hat_neg = remove_mean(net(t, xt_neg))
        loss += torch.sum(torch.square(b_hat_neg - i_t_dot), dim=sum_indexes) / dim
        loss *= 0.5
    return loss

# Create an EMA model
net_ema = torch.optim.swa_utils.AveragedModel(
    model=net,
    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
    use_buffers=True
)

# Build the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
r = trange(args.n_steps)
for epoch in r:
    optimizer.zero_grad()
    x0 = target_0.sample((args.batch_size_ot,))
    x1 = target_1.sample((args.batch_size_ot,))
    x0, x1 = superpose_points(x0, x1)
    x0, x1 = sample_ot_coupling(x0.view((-1, dim)), x1.view((-1, dim)),
        n_samples=args.batch_size)
    x0, x1 = x0.view((-1, *data_shape)), x1.view((-1, *data_shape))
    loss = loss_fn_velocity(net, x0, x1, time_sampler).mean()
    loss.backward()
    optimizer.step()
    net_ema.update_parameters(net)
    r.set_postfix(b_loss=loss.item())

# Move everything to CPU
net = net.cpu()
net_ema = net_ema.cpu()

# Save the results
with open(args.results_path + '/' + filename, 'wb') as f:
    pickle.dump({'config': config, 'net': net.state_dict(), 'net_ema': net_ema.state_dict()}, f)
