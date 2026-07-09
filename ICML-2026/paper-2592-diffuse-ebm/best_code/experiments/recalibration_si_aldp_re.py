# Libraries
import argparse
import math
import numpy as np
import os
import pickle
import pprint
import random
import torch
import matplotlib.pyplot as plt
from diffclf.distr.aldp import AlanineDipeptide
from diffclf.networks.ebm import EBM, SIEnergyDenoiserNet
from diffclf.networks.egnn import EGNN_atom
from diffclf.re.diffusion import diffusion_re_sampler
from diffclf.si.stochastic_interpolant import SimpleStochasticInterpolant
from diffclf.utils.se3_utils import remove_mean

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--vacuum_datapath', type=str)
parser.add_argument('--ckpt_filepath', type=str)
parser.add_argument('--subsample_time', type=int, default=1)
parser.add_argument('--split_size', type=int, default=1024)
parser.add_argument('--swap_frequency', type=int, default=1)
parser.add_argument('--n_parallel_re', type=int, default=2)
parser.add_argument('--n_warmup_mcmc_steps', type=int, default=4096)
parser.add_argument('--swap_frequency', type=int, default=1)
parser.add_argument('--with_local_mcmc', action=argparse.BooleanOptionalAction)
parser.add_argument('--n_mcmc_steps', type=int, default=4096)
parser.add_argument('--step_size', type=float, default=1e-4)
parser.add_argument('--diff_vals', type=str, default='1e-2,1e-3,1e-4')
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
filename += '_subsample_time_' + str(args.subsample_time)
filename += '_n_parallel_re_' + str(args.n_parallel_re)
filename += '_swap_frequency_' + str(args.swap_frequency)
filename += '_seed_{}.pkl'.format(args.seed)

# Build the distributions
target_0 = AlanineDipeptide(args.data_path, env="implicit").to(device)
target_1 = AlanineDipeptide(args.data_path, env="vacuum").to(device)
target_1.load_data(remove_mean(
    torch.load(args.vacuum_datapath).view((-1, *target_1.data_shape))
))
data_shape = (target_0.n_particles, target_0.n_dimensions)
data_shape_ones = (1, 1)
sum_indexes = (-1, -2)
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

class HarcodeTargetEBM(EBM):
    """Hardcode the target"""

    def __init__(self, base_ebm, dist0, t_0, dist1, t_1):
        super().__init__(build_score=False, build_log_prob_dot=False, build_grad_and_log_prob=False,
            build_log_prob_and_grad_and_dot=False)
        self.base_ebm = base_ebm
        self.dist0 = dist0
        self.register_buffer('t_0', t_0)
        self.dist1 = dist1
        self.register_buffer('t_1', t_1)

    def check_t_0(self, t):
        """Check if the time is t_0."""
        return torch.abs(t - self.t_0) < 1e-5

    def check_t_1(self, t):
        """Check if the time is t_0."""
        return torch.abs(t - self.t_1) < 1e-5

    def energy(self, t, x):
        """Compute the energy."""
        mask_0 = self.check_t_0(t).flatten()
        any_0 = torch.any(mask_0)
        mask_1 = self.check_t_1(t).flatten()
        any_1 = torch.any(mask_1)
        mask_xor = (~torch.logical_xor(mask_0, mask_1)).flatten()
        any_xor = torch.any(mask_xor)
        if any_0 or any_1:
            ret = torch.empty((t.shape[0],), device=x.device)
            if any_0:
                ret[mask_0] = -self.dist0.log_prob(x[mask_0])
            if any_1:
                ret[mask_1] = -self.dist1.log_prob(x[mask_1])
            if any_xor:
                ret[mask_xor] = self.base_ebm.energy(t[mask_xor], x[mask_xor])
            return ret
        else:
            return self.base_ebm.energy(t, x)

    def score(self, t, x):
        """Compute the score."""
        mask_0 = self.check_t_0(t).flatten()
        any_0 = torch.any(mask_0)
        mask_1 = self.check_t_1(t).flatten()
        any_1 = torch.any(mask_1)
        mask_xor = (~torch.logical_xor(mask_0, mask_1)).flatten()
        any_xor = torch.any(mask_xor)
        if any_0 or any_1:
            ret = torch.empty_like(x)
            if any_0:
                ret[mask_0] = self.dist0.score(x[mask_0])
            if any_1:
                ret[mask_1] = self.dist1.score(x[mask_1])
            if any_xor:
                ret[mask_xor] = self.base_ebm.score(t[mask_xor], x[mask_xor])
            return ret
        else:
            return self.base_ebm.score(t, x)

    def denoiser(self, t, x):
        """Compute the denoiser."""
        return -gamma_fn(t) * self.score(t, x)

    def log_prob_and_grad(self, t, x, return_denoiser=False):
        """Compute the log-prob and its gradient."""
        mask_0 = self.check_t_0(t).flatten()
        any_0 = torch.any(mask_0)
        mask_1 = self.check_t_1(t).flatten()
        any_1 = torch.any(mask_1)
        mask_xor = (~torch.logical_xor(mask_0, mask_1)).flatten()
        any_xor = torch.any(mask_xor)
        if any_0 or any_1:
            ret_log_prob = torch.empty((t.shape[0],), device=x.device)
            ret_grad = torch.empty_like(x)
            if any_0:
                ret_log_prob[mask_0], ret_grad[mask_0] = self.dist0.log_prob_and_grad(x[mask_0])
            if any_1:
                ret_log_prob[mask_1], ret_grad[mask_1] = self.dist1.log_prob_and_grad(x[mask_1])
            if any_xor:
                ret_log_prob[mask_xor], ret_grad[mask_xor] = self.base_ebm.log_prob_and_grad(
                    t[mask_xor], x[mask_xor])
            if return_denoiser:
                return ret_log_prob, -gamma_fn(t) * ret_grad
            else:
                return ret_log_prob, ret_grad
        else:
            return self.base_ebm.log_prob_and_grad(t, x)

# Harcode the EBM
if not args.wo_hardcode_model:
    ebm = HarcodeTargetEBM(ebm, target_0, times[0], target_1, times[-1])

# Handle the memory better
class MemoryEBM(torch.nn.Module):

    def __init__(self, base_ebm, split_size):
        super().__init__()
        self.base_ebm = base_ebm
        self.split_size = split_size

    def get_chunksize(self, batch_size):
        return batch_size // self.split_size

    def energy(self, t, x):
        if x.shape[0] < self.split_size:
            return self.base_ebm.energy(t, x)
        else:
            chunk_size = self.get_chunksize(x.shape[0])
            return torch.cat([self.base_ebm.energy(t_, x_)
                for t_, x_ in zip(torch.chunk(t, chunk_size), torch.chunk(x, chunk_size))], dim=0)

    def score(self, t, x):
        if x.shape[0] < self.split_size:
            return self.base_ebm.score(t, x)
        else:
            chunk_size = self.get_chunksize(x.shape[0])
            return torch.cat([self.base_ebm.score(t_, x_)
                for t_, x_ in zip(torch.chunk(t, chunk_size), torch.chunk(x, chunk_size))], dim=0)

    def denoiser(self, t, x):
        if x.shape[0] < self.split_size:
            return self.base_ebm.denoiser(t, x)
        else:
            chunk_size = self.get_chunksize(x.shape[0])
            return torch.cat([self.base_ebm.denoiser(t_, x_)
                for t_, x_ in zip(torch.chunk(t, chunk_size), torch.chunk(x, chunk_size))], dim=0)

    def log_prob_and_grad(self, t, x, return_denoiser=False):
        if x.shape[0] < self.split_size:
            return self.base_ebm.log_prob_and_grad(t, x)
        else:
            chunk_size = self.get_chunksize(x.shape[0])
            log_probs, others = [], []
            for t_, x_ in zip(torch.chunk(t, chunk_size), torch.chunk(x, chunk_size)):
                log_prob, other = self.base_ebm.log_prob_and_grad(t_, x_, return_denoiser=return_denoiser)
                log_probs.append(log_prob)
                others.append(other)
            return torch.cat(log_probs, dim=0), torch.cat(others, dim=0)
ebm = MemoryEBM(ebm, split_size=args.split_size).to(device)

def run_re(diff, n_warmup_mcmc_steps, n_mcmc_steps, batch_size):
    """Run the SMC algorithm"""
    def forward_kernel(x_s, s, t, grad_s_x_s, aux_s_x_s, **kwargs):
        mean, var = si.forward_sde_kernel(s, t, x_s, None, aux_s_x_s['velocity'],
            grad_s_x_s, diff, return_mean_var=True)
        z_t = remove_mean(torch.randn_like(x_s))
        x_t = mean + torch.sqrt(var) * z_t
        log_prob = -0.5 * torch.sum(torch.square(z_t), dim=sum_indexes)
        log_prob -= 0.5 * dim * math.log(2. * math.pi)
        log_prob -= 0.5 * dim * torch.log(var).view((x_s.shape[:-len(data_shape)]))
        return x_t, log_prob
    def forward_kernel_log_prob(x_t, x_s, s, t, grad_s_x_s, aux_s_x_s, **kwargs):
        mean, var = si.forward_sde_kernel(s, t, x_s, None, aux_s_x_s['velocity'],
            grad_s_x_s, diff, return_mean_var=True)
        log_prob = -0.5 * torch.sum(torch.square(x_t - mean) / var, dim=sum_indexes)
        log_prob -= 0.5 * dim * math.log(2. * math.pi)
        log_prob -= 0.5 * dim * torch.log(var).view((x_s.shape[:-len(data_shape)]))
        return log_prob
    def backward_kernel(x_t, s, t, grad_t_x_t, aux_t_x_t, **kwargs):
        mean, var = si.backward_sde_kernel(s, t, x_t, None, aux_t_x_t['velocity'],
            grad_t_x_t, diff, return_mean_var=True)
        z_s = remove_mean(torch.randn_like(x_t))
        x_s = mean + torch.sqrt(var) * z_s
        log_prob = -0.5 * torch.sum(torch.square(z_s), dim=sum_indexes)
        log_prob -= 0.5 * dim * math.log(2. * math.pi)
        log_prob -= 0.5 * dim * torch.log(var).view((x_t.shape[:-len(data_shape)]))
        return x_s, log_prob
    def backward_kernel_log_prob(x_s, x_t, s, t, grad_t_x_t, aux_t_x_t, **kwargs):
        mean, var = si.backward_sde_kernel(s, t, x_t, None, aux_t_x_t['velocity'],
            grad_t_x_t, diff, return_mean_var=True)
        log_prob = -0.5 * torch.sum(torch.square(x_s - mean) / var, dim=sum_indexes)
        log_prob -= 0.5 * dim * math.log(2. * math.pi)
        log_prob -= 0.5 * dim * torch.log(var).view((x_t.shape[:-len(data_shape)]))
        return log_prob
    # Define ts
    time_mask = torch.linspace(0, times.shape[0]-1, int(times.shape[0] / args.subsample_time)).int()
    ts_ = times[time_mask].view((-1, 1, 1, 1)).repeat((1, batch_size, 1, 1))
    # Get the x_init
    x_init = torch.empty((ts_.shape[0], batch_size, *data_shape), device=device)
    x_init[-1] = target_1.sample((batch_size,))
    for i in range(ts_.shape[0]-1, 0, -1):
        vel_t_x_t = velocity_fn(ts_[i], x_init[i])
        grad_t_x_t = ebm.score(ts_[i], x_init[i])
        x_init[i-1] = backward_kernel(x_init[i], ts_[i-1], ts_[i], grad_t_x_t,
            { 'velocity' : vel_t_x_t })[0]
    # Run the RE
    samples, _, diags = diffusion_re_sampler(
        x_init=x_init,
        forward_kernel=forward_kernel,
        forward_kernel_log_prob=forward_kernel_log_prob,
        backward_kernel=backward_kernel,
        backward_kernel_log_prob=backward_kernel_log_prob,
        times=ts_,
        log_prob_and_grads=ebm.log_prob_and_grad,
        kernel_aux_fn=lambda t, y : { 'velocity' : velocity_fn(t, y) },
        swap_frequency=args.swap_frequency,
        n_warmup_mcmc_steps=n_warmup_mcmc_steps,
        n_mcmc_steps=n_mcmc_steps,
        step_sizes_per_noise=1e-4 * torch.ones_like(ts_),
        per_noise_init=True,
        ignore_mcmc=args.swap_frequency == 1,
        is_particles=True,
        verbose=True
    )
    samples = samples.detach().cpu()
    return samples.view((-1, *data_shape)), diags['swap_acc'].cpu()

# Make the plots dir
os.makedirs(f"{args.results_path}/plots", exist_ok=True)

# Evaluate both models
results = {}
for diff_val in list(map(float, args.diff_vals.split(','))):
    print('# Sampling with diff_val = ', str(diff_val))
    # Run RE
    all_samples, swap_acceptances = run_re(diff_val, args.n_warmup_mcmc_steps, args.n_mcmc_steps,
        args.n_parallel_re)
    samples = all_samples[0]
    # Compute the metrics
    results[str(diff_val)] = target_0.compute_metrics(samples)
    results[str(diff_val)]['log_prob'] = target_0.log_prob(samples)
    results[str(diff_val)]['samples'] = target_0.compute_psi_phi(samples)
    # Make plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    target_0.plot_samples(ax[0], samples, label="model")
    target_0.plot_samples(ax[1], target_0.sample((samples.shape[0], )).clone().detach().cpu(), already_filtered=True, label="ground truth")
    fig.savefig(args.results_path + "/plots/{}_diff_val_{:.1e}.png".format(
        filename[:-4], diff_val
    ))

# Save the results
with open(args.results_path + '/' + filename, 'wb') as f:
    pickle.dump({
    	'config': config,
    	'results' : results,
        'config_score_pretrain': d_score['config'],
        'config_vel': d_vel['config'],
        'config_score' : d['config']
    }, f)
