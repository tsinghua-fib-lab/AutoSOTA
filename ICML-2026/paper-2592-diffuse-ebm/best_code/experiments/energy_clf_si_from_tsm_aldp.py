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
from tqdm import trange, tqdm

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--vacuum_datapath', type=str)
parser.add_argument('--ckpt_filepath', type=str)
parser.add_argument('--vel_ckpt_filepath', type=str)
parser.add_argument('--loss_type', type=str)
parser.add_argument('--k', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--dataset_size', type=int, default=250000)
parser.add_argument('--n_epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--batch_size_ot', type=int, default=256)
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
    if 'factorize_tsm' in ckpt_data['config']:
        factorize_tsm = bool(ckpt_data['config']['factorize_tsm'])
    else:
        factorize_tsm = False
    if 'tsm_weighting_type' in ckpt_data['config']:
        tsm_weighting_type = ckpt_data['config']['tsm_weighting_type']
    else:
        tsm_weighting_type = None
    if 'tsm_t_limit' in ckpt_data['config']:
        tsm_t_limit = ckpt_data['config']['tsm_t_limit']
    else:
        tsm_t_limit = None
    dsm_weighting_type = ckpt_data['config']['dsm_weighting_type']
    gamma_factor = ckpt_data['config']['gamma_factor']
    if (dsm_weighting_type == 'square') or (tsm_weighting_type == 'square'):
        reg_val = gamma_factor**2
    elif (dsm_weighting_type == 'linear') or (tsm_weighting_type == 'linear'):
        reg_val = gamma_factor
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
filename = 'energy_clf_si_from_tsm_aldp'
filename += '_dsm_weighting_type_' + dsm_weighting_type
if tsm_weighting_type is not None:
    filename += '_tsm_weighting_type_' + tsm_weighting_type
if tsm_t_limit is not None:
    filename += '_tsm_t_limit_{:.2e}'.format(tsm_t_limit)
if factorize_tsm:
    filename += '_factorized'
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
        if 'with_target' in args.loss_type:
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
ebm = WrapF(ebm, n_levels).to(device)

# Get the weighting function for DSM
if dsm_weighting_type == "uniform":
    dsm_weighting_func = lambda t: 1.0
elif dsm_weighting_type == "linear":
    dsm_weighting_func = lambda t: gamma_fn(t)
elif dsm_weighting_type == "square":
    dsm_weighting_func = lambda t: torch.square(gamma_fn(t))
else:
    raise NotImplementedError(f"Weighting type {dsm_weighting_type} not implemented!")

# Get the weighting function for TSM
if tsm_weighting_type == "uniform":
    tsm_weighting_func = lambda t: 1.0
elif tsm_weighting_type == "linear":
    tsm_weighting_func = lambda t: gamma_fn(t)
elif tsm_weighting_type == "square":
    tsm_weighting_func = lambda t: torch.square(gamma_fn(t))
else:
    raise NotImplementedError(f"Weighting type {tsm_weighting_type} not implemented!")

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

def loss_fn_tsm(ebm, x0, score0, x1, score1, time_sampler, antithetic=True):
    """Target Score Matching loss for SI"""
    # Get the shapes
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    dim = math.prod(data_shape)
    sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
    # Build the times
    t = time_sampler.sample((x0.shape[0],)).view((-1, *data_shape_ones))
    # Compute the loss
    alpha_t = si.alpha(t)
    beta_t = si.beta(t)
    i_t = alpha_t * x0 + beta_t * x1
    gamma_t = gamma_fn(t)
    z = remove_mean(torch.randn_like(i_t))
    xt = i_t + gamma_t * z
    z_hat = remove_mean(ebm.denoiser(t, xt))
    loss_dsm = torch.sum(torch.square(z_hat - z) / dsm_weighting_func(t), dim=sum_indexes) / dim
    target_score_tsm = torch.where(t < tsm_t_limit, score0 / alpha_t, score1 / beta_t)
    if factorize_tsm:
        loss_tsm = torch.sum(
            (torch.square(z_hat) + gamma_t * z_hat * target_score_tsm) / tsm_weighting_func(t),
            dim=sum_indexes
        ) / dim
    else:
        loss_tsm = torch.sum(
            torch.square(z_hat + gamma_t * target_score_tsm) / tsm_weighting_func(t),
            dim=sum_indexes
        ) / dim
    if antithetic:
        xt_neg = i_t - gamma_t * z
        z_hat_neg = remove_mean(ebm.denoiser(t, xt_neg))
        loss_dsm += torch.sum(torch.square(z_hat_neg + z) / dsm_weighting_func(t), dim=sum_indexes) / dim
        loss_dsm *= 0.5
        if factorize_tsm:
            loss_tsm = torch.sum(
                (torch.square(z_hat_neg) + gamma_t * z_hat_neg * target_score_tsm) / tsm_weighting_func(t),
                dim=sum_indexes
            ) / dim
        else:
            loss_tsm = torch.sum(
                torch.square(z_hat_neg + gamma_t * target_score_tsm) / tsm_weighting_func(t),
                dim=sum_indexes
            ) / dim
        loss_tsm *= 0.5
    return loss_dsm, loss_tsm

# Build the necessary stuff for multi-level
i_idx, j_idx = torch.meshgrid(
    torch.arange(args.k-2 if 'with_target' in args.loss_type else args.k, device=device),
    torch.arange(args.k, device=device),
    indexing='ij'
)
i_idx = i_idx.reshape(-1)
j_idx = j_idx.reshape(-1)
diag_mask_target = i_idx == j_idx
diag_mask = torch.eye(args.k, dtype=torch.bool, device=device).flatten()

def loss_fn_multilevel(ebm, x0, score0, x1, score1, k, time_sampler, i_idx, j_idx, diag_mask, antithetic=True):
    # Get the shapes
    batch_size = x0.shape[0]
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    data_shape_minus_ones = (-1,) * len(data_shape)
    dim = math.prod(data_shape)
    sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
    # Sample the times
    ts, idx = time_sampler.sample((batch_size, k), return_idx=True, unique=True)
    ts = ts.view((batch_size, k, *data_shape_ones))
    # Expand things
    x0_expanded = x0.unsqueeze(1).expand((-1, k, *data_shape_minus_ones))
    score0_expanded = score0.unsqueeze(1).expand((-1, k, *data_shape_minus_ones))
    x1_expanded = x1.unsqueeze(1).expand((-1, k, *data_shape_minus_ones))
    score1_expanded = score1.unsqueeze(1).expand((-1, k, *data_shape_minus_ones))
    # Compute the loss
    alpha_t = si.alpha(ts)
    beta_t = si.beta(ts)
    i_t = alpha_t * x0_expanded + beta_t * x1_expanded
    # Noise all those samples
    gamma_t = gamma_fn(ts)
    z = remove_mean(torch.randn_like(i_t))
    xt = i_t + gamma_t * z
    # Compute the energy and the denoiser on the diagonal
    neg_en_ii, denoiser_ii = ebm.log_prob_and_grad(
        ts.view((-1, *data_shape_ones)), xt.view((-1, *data_shape)), return_denoiser=True)
    neg_en_ii = neg_en_ii.view((batch_size, k))
    denoiser_ii = remove_mean(denoiser_ii.view((batch_size, k, *data_shape)))
    # Compute the denoiser matching loss
    loss_dsm = torch.sum(torch.square(denoiser_ii - z) / dsm_weighting_func(ts), dim=sum_indexes) / dim
    target_score_tsm = torch.where(ts < tsm_t_limit, score0_expanded / alpha_t, score1_expanded / beta_t)
    if factorize_tsm:
        loss_tsm = torch.sum(
            (torch.square(denoiser_ii) + gamma_t * denoiser_ii * target_score_tsm) / tsm_weighting_func(ts),
            dim=sum_indexes
        ) / dim
    else:
        loss_tsm = torch.sum(
            torch.square(denoiser_ii + gamma_t * target_score_tsm) / tsm_weighting_func(ts),
            dim=sum_indexes
        ) / dim
    if antithetic:
        xt_neg = i_t - gamma_t * z
        denoiser_ii_neg = ebm.denoiser(ts.view((-1, *data_shape_ones)), xt_neg.view((-1, *data_shape)))
        denoiser_ii_neg = remove_mean(denoiser_ii_neg.view((batch_size, k, *data_shape)))
        loss_dsm += torch.sum(torch.square(denoiser_ii_neg + z) / dsm_weighting_func(ts), dim=sum_indexes) / dim
        loss_dsm *= 0.5
        if factorize_tsm:
            loss_tsm += torch.sum(
                (torch.square(denoiser_ii_neg) + gamma_t * denoiser_ii_neg * target_score_tsm) / tsm_weighting_func(ts),
                dim=sum_indexes
            ) / dim
        else:
            loss_tsm += torch.sum(
                torch.square(denoiser_ii_neg + gamma_t * target_score_tsm) / tsm_weighting_func(ts),
                dim=sum_indexes
            ) / dim
        loss_tsm *= 0.5
    # Remove the f from neg_en_ii
    neg_en_ii -= ebm.f[idx]    
    # Compute the remaining energies
    ts_ij = ts[:, i_idx[~diag_mask]]
    xt_ij = xt[:, j_idx[~diag_mask], :]
    f_i = ebm.f[idx[:, i_idx[~diag_mask]]]
    neg_en_no_diag = ebm.log_prob(ts_ij.view((-1, *data_shape_ones)), xt_ij.view((-1, *data_shape)))
    neg_en_no_diag = neg_en_no_diag.view((batch_size, k-1, k)) - f_i.view((batch_size, k-1, k))
    # Reconstruct the full matrix
    diag_mask = diag_mask.view((k, k))
    neg_en = torch.zeros((batch_size, k, k), device=x0.device)
    neg_en[:, diag_mask] = neg_en_ii
    neg_en[:, ~diag_mask] = neg_en_no_diag.view((batch_size, -1))
    # Compute the loss
    neg_en_lse = torch.logsumexp(neg_en, dim=1)
    return loss_dsm, loss_tsm, -(neg_en_ii - neg_en_lse).mean(dim=-1)

def loss_fn_multilevel_with_target(ebm, x0, log_prob0, score0, x1, log_prob1, score1, k, time_sampler,
        i_idx, j_idx, diag_mask, diag_mask_target, antithetic=True):
    # Get the shapes
    batch_size = x0.shape[0]
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    data_shape_minus_ones = (-1,) * len(data_shape)
    dim = math.prod(data_shape)
    sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
    # Sample the times
    ts, idx = time_sampler.sample((batch_size, k-2), return_idx=True, unique=True,
        exclude_first_level=True, exclude_last_level=True)
    ts = ts.view((batch_size, k-2, *data_shape_ones))
    # Expand things
    x0_expanded = x0.unsqueeze(1).expand((-1, k-2, *data_shape_minus_ones))
    score0_expanded = score0.unsqueeze(1).expand((-1, k-2, *data_shape_minus_ones))
    x1_expanded = x1.unsqueeze(1).expand((-1, k-2, *data_shape_minus_ones))
    score1_expanded = score1.unsqueeze(1).expand((-1, k-2, *data_shape_minus_ones))
    # Compute the loss
    alpha_t = si.alpha(ts)
    beta_t = si.beta(ts)
    i_t = alpha_t * x0_expanded + beta_t * x1_expanded
    # Noise all those samples
    gamma_t = gamma_fn(ts)
    z = remove_mean(torch.randn_like(i_t))
    xt = i_t + gamma_t * z
    # Compute the energy and the denoiser on the diagonal
    neg_en_ii, denoiser_ii = ebm.log_prob_and_grad(
        ts.view((-1, *data_shape_ones)), xt.view((-1, *data_shape)), return_denoiser=True)
    neg_en_ii = neg_en_ii.view((batch_size, k-2))
    denoiser_ii = remove_mean(denoiser_ii.view((batch_size, k-2, *data_shape)))
    # Compute the denoiser matching loss
    loss_dsm = torch.sum(torch.square(denoiser_ii - z) / dsm_weighting_func(ts), dim=sum_indexes) / dim
    target_score_tsm = torch.where(ts < tsm_t_limit, score0_expanded / alpha_t, score1_expanded / beta_t)
    if factorize_tsm:
        loss_tsm = torch.sum(
            (torch.square(denoiser_ii) + gamma_t * denoiser_ii * target_score_tsm) / tsm_weighting_func(ts),
            dim=sum_indexes
        ) / dim
    else:
        loss_tsm = torch.sum(
            torch.square(denoiser_ii + gamma_t * target_score_tsm) / tsm_weighting_func(ts),
            dim=sum_indexes
        ) / dim
    if antithetic:
        xt_neg = i_t - gamma_t * z
        denoiser_ii_neg = ebm.denoiser(ts.view((-1, *data_shape_ones)), xt_neg.view((-1, *data_shape)))
        denoiser_ii_neg = remove_mean(denoiser_ii_neg.view((batch_size, k-2, *data_shape)))
        loss_dsm += torch.sum(torch.square(denoiser_ii_neg + z) / dsm_weighting_func(ts), dim=sum_indexes) / dim
        loss_dsm *= 0.5
        if factorize_tsm:
            loss_tsm += torch.sum(
                (torch.square(denoiser_ii_neg) + gamma_t * denoiser_ii_neg * target_score_tsm) / tsm_weighting_func(ts),
                dim=sum_indexes
            ) / dim
        else:
            loss_tsm += torch.sum(
                torch.square(denoiser_ii_neg + gamma_t * target_score_tsm) / tsm_weighting_func(ts),
                dim=sum_indexes
            ) / dim
        loss_tsm *= 0.5
    # Remove the f from neg_en_ii
    neg_en_ii -= ebm.f[idx]
    # Add the marginal log-probs to neg_en_ii
    neg_en_ii = torch.cat((
        log_prob0.unsqueeze(-1) - ebm.f0,
        neg_en_ii,
        log_prob1.unsqueeze(-1) - ebm.f1
    ), dim=-1)
    # Add x0 and x1 to the xt
    xt = torch.cat((x0.unsqueeze(1), xt, x1.unsqueeze(1)), dim=1)
    # Compute the remaining energies
    ts_ij = ts[:, i_idx[~diag_mask_target]]
    xt_ij = xt[:, j_idx[~diag_mask_target], :]
    f_i = ebm.f[idx[:, i_idx[~diag_mask_target]]]
    neg_en_no_diag = ebm.log_prob(ts_ij.view((-1, *data_shape_ones)), xt_ij.view((-1, *data_shape)))
    neg_en_no_diag = neg_en_no_diag.view((batch_size, -1))
    neg_en_no_diag = torch.cat((
        target_0.log_prob(xt[:,1:].reshape((-1, *data_shape))).view((batch_size, -1)) - ebm.f0,
        neg_en_no_diag,
        target_1.log_prob(xt[:,:-1].reshape((-1, *data_shape))).view((batch_size, -1)) - ebm.f1
    ), dim=1)
    ones = torch.ones((batch_size, k-1), device=x0.device)
    neg_en_no_diag = neg_en_no_diag.view((batch_size, k-1, k)) -\
        torch.cat((ebm.f0 * ones, f_i, ebm.f1 * ones), dim=-1).view((batch_size, k-1, k))
    # Reconstruct the full matrix
    diag_mask = diag_mask.view((k, k))
    neg_en = torch.zeros((batch_size, k, k), device=x0.device)
    neg_en[:, diag_mask] = neg_en_ii
    neg_en[:, ~diag_mask] = neg_en_no_diag.view((batch_size, -1))
    # Compute the loss
    neg_en_lse = torch.logsumexp(neg_en, dim=1)
    return loss_dsm, loss_tsm, -(neg_en_ii - neg_en_lse).mean(dim=-1)

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

# Build the dataset with precomputed scores and OT
print('Precomputing the dataset...')
samples0, samples1 = [], []
log_probs0, log_probs1 = [], []
scores0, scores1 = [], []
for _ in trange(args.dataset_size // args.batch_size):
    x0 = target_0.sample((args.batch_size_ot,))
    x1 = target_1.sample((args.batch_size_ot,))
    x0, x1 = superpose_points(x0, x1)
    x0, x1 = sample_ot_coupling(x0.view((-1, dim)), x1.view((-1, dim)),
        n_samples=args.batch_size)
    x0, x1 = x0.view((-1, *data_shape)), x1.view((-1, *data_shape))
    log_prob0, score0 = target_0.log_prob_and_grad(x0)
    samples0.append(x0.detach())
    log_probs0.append(log_prob0.detach())
    scores0.append(score0.detach())
    log_prob1, score1 = target_1.log_prob_and_grad(x1)
    samples1.append(x1.detach())
    log_probs1.append(log_prob1.detach())
    scores1.append(score1.detach())

# Build the optimizer
optimizer = torch.optim.Adam(ebm.parameters(), lr=args.lr)
dataset = torch.utils.data.TensorDataset(
    torch.concat(samples0), torch.concat(log_probs0), torch.concat(scores0),
    torch.concat(samples1), torch.concat(log_probs1), torch.concat(scores1)
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
for epoch in range(args.n_epochs):
    loop = tqdm(dataloader, leave=True)
    loop.set_description(f"Epoch {epoch+1}/{args.n_epochs}")
    for data in loop:
        optimizer.zero_grad()
        x0, log_prob0, score0, x1, log_prob1, score1 = data
        if args.loss_type == 'tsm':
            x0 = x0.unsqueeze(0).repeat((args.k, 1, 1, 1)).view((-1, 22, 3))
            score0 = score0.unsqueeze(0).repeat((args.k, 1, 1, 1)).view((-1, 22, 3))
            x1 = x1.unsqueeze(0).repeat((args.k, 1, 1, 1)).view((-1, 22, 3))
            score1 = score1.unsqueeze(0).repeat((args.k, 1, 1, 1)).view((-1, 22, 3))
            loss_dsm, loss_tsm = loss_fn_tsm(ebm, x0, score0, x1, score1, time_sampler)
            loss_dsm = loss_dsm.mean()
            loss_tsm = loss_tsm.mean()
            loss = 0.5 * loss_dsm + 0.5 * loss_tsm
        elif args.loss_type == 'multi_level_with_target':
            loss_dsm, loss_tsm, loss_clf = loss_fn_multilevel_with_target(ebm, x0, log_prob0, score0,
                x1, log_prob1, score1, args.k, time_sampler, i_idx, j_idx, diag_mask, diag_mask_target)
            loss_dsm = loss_dsm.mean()
            loss_tsm = loss_tsm.mean()
            loss_clf = loss_clf.mean()
            loss = loss_clf + reg_val * (0.5 * loss_dsm + 0.5 * loss_tsm)
        else:
            loss_dsm, loss_tsm, loss_clf = loss_fn_multilevel(ebm, x0, score0, x1, score1, args.k, time_sampler,
                i_idx, j_idx, diag_mask)
            loss_dsm = loss_dsm.mean()
            loss_tsm = loss_tsm.mean()
            loss_clf = loss_clf.mean()
            loss = loss_clf + reg_val * (0.5 * loss_dsm + 0.5 * loss_tsm)
        loss.backward()
        optimizer.step()
        ebm_ema.update_parameters(ebm)
        with torch.no_grad():
            ebm.f -= ebm.f[-1].clone()
        if args.loss_type == 'tsm':
            loop.set_postfix(loss=loss.item(), loss_dsm=loss_dsm.item(), loss_tsm=loss_tsm.item())
        else:
            loop.set_postfix(loss=loss.item(), loss_clf=loss_clf.item(), loss_dsm=loss_dsm.item(),
                loss_tsm=loss_tsm.item(), loss_dsm_reg=reg_val * loss_dsm.item(),
                loss_tsm_reg=reg_val * loss_tsm.item())

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
    pickle.dump({ 'config': config, 'ckpt_config' : ckpt_data['config'], 'results' : results,
        'ebm': ebm.state_dict(), 'ebm_ema': ebm_ema.state_dict() }, f)
