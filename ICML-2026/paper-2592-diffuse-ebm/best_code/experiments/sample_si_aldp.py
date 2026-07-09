# Libraries
import argparse
import numpy as np
import os
import pickle
import pprint
import random
import torch
import matplotlib.pyplot as plt
from diffclf.distr.aldp import AlanineDipeptide
from diffclf.networks.ebm import SIEnergyDenoiserNet
from diffclf.networks.egnn import EGNN_atom
from diffclf.si.stochastic_interpolant import SimpleStochasticInterpolant
from diffclf.utils.se3_utils import remove_mean
from tqdm import trange

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--vacuum_datapath', type=str)
parser.add_argument('--velocity_ckpt_filepath', type=str)
parser.add_argument('--score_ckpt_filepath', type=str)
parser.add_argument('--n_samples', type=int, default=8192)
parser.add_argument('--split_size', type=int, default=1024)
parser.add_argument('--diff_val', type=float, default=1e-2)
parser.add_argument('--use_sde_sampling', action=argparse.BooleanOptionalAction)
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

# Make a Pytorch device
device = torch.device('cuda')

# Load the checkpoints
with open(args.velocity_ckpt_filepath, 'rb') as f:
    d_velocity = pickle.load(f)
    gamma_factor = d_velocity['config']['gamma_factor']
    n_levels = d_velocity['config']['n_levels']
with open(args.score_ckpt_filepath, 'rb') as f:
	d_score = pickle.load(f)
	dsm_weighting_type = d_score['config']['dsm_weighting_type']
	if gamma_factor != d_score['config']['gamma_factor']:
		print('Gamma factors not the same.')
		exit(0)
	if n_levels != d_score['config']['n_levels']:
		print('Gamma factors not the same.')
		exit(0)
	if 'tsm_t_limit' in d_score['config']:
		tsm_t_limit = d_score['config']['tsm_t_limit']
	else:
		tsm_t_limit = None
	if 'tsm_weighting_type' in d_score['config']:
		tsm_weighting_type = d_score['config']['tsm_weighting_type']
	else:
		tsm_weighting_type = None


# Make a filename
filename = 'sample_si_aldp'
filename += '_gamma_factor_{:.1e}'.format(gamma_factor)
filename += '_dsm_weighting_type' + str(dsm_weighting_type)
if tsm_weighting_type is not None:
	filename += '_tsm_weighting_type' + str(tsm_weighting_type)
if args.use_sde_sampling:
	filename += '_diff_val_{:.2e}' + str(args.diff_val)
if tsm_t_limit is not None:
	filename += '_tsm_t_limit_{:.2e}'.format(tsm_t_limit)
filename += '_seed_{}.pkl'.format(args.seed)

# Build the distributions
target_0 = AlanineDipeptide(args.data_path, env="implicit").to(device)
target_1 = AlanineDipeptide(args.data_path, env="vacuum").to(device)
target_1.load_data(remove_mean(
    torch.load(args.vacuum_datapath).view((-1, *target_1.data_shape))
))

# Build the train times and normalizing constants
times = torch.linspace(1e-3, 1.0-1e-3, n_levels, device=device)

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
    hidden_nf=d_score['config']['hidden_nf'],
    time_embedding_dim=128,
    atom_type_embedding_dim=64,
    n_layers=d_score['config']['n_layers'],
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

# Load the score net
ebm.load_state_dict(d_score['ebm'])

# Move everything to the right device
velocity_fn = velocity_fn.to(device)
ebm = ebm.to(device)

# Disable gradient with respect to the parameters
for p in ebm.parameters():
    p.requires_grad_(False)
for p in velocity_fn.parameters():
    p.requires_grad_(False)

# Build the drift function
def drift_fn(t, x, return_grad=False):
	v = velocity_fn(t, x)
	s = ebm.score(t, x)
	b = v - si.gamma_dot_times_gamma(t) * s
	if return_grad:
		return b, s
	else:
		return b

# Generate samples
samples = []
log_probs_0, scores_0, log_probs_1, scores_1 = [], [], [], []
true_log_probs_0, true_scores_0, true_log_probs_1, true_scores_1 = [], [], [], []
n_trajectories = int(args.n_samples / args.split_size)
t_ones = torch.ones((args.split_size, 1, 1), device=device)
for j in range(n_trajectories):
	print('Trajectory {}/{}..'.format(j+1, n_trajectories))
	# Sample the distribution at t=0
	x = target_0.sample((args.split_size,))
	for i in trange(n_levels-1):
		t_cur = t_ones * times[i]
		t_next = t_ones * times[i + 1]
		if args.use_sde_sampling:
			drift_cur, score_cur = drift_fn(t_cur, x, return_grad=True)
			mean, var = si.forward_sde_kernel(t_cur, t_next, x, drift_cur, None, score_cur,
				args.diff_val, return_mean_var=True)
			x = mean + torch.sqrt(var) * remove_mean(torch.randn_like(mean))
		else:
			x = si.ode_step(t_cur, t_next, x, drift_fn)
	x = x.detach().cpu().clone()
	samples.append(x)
	samples_0 = target_0.sample((args.split_size,))
	log_probs_0_, scores_0_ = ebm.log_prob_and_grad(times[0] * t_ones, samples_0)
	true_log_probs_0_, true_scores_0_ = target_0.log_prob_and_grad(samples_0)
	log_probs_0.append(log_probs_0_.detach().cpu())
	true_log_probs_0.append(true_log_probs_0_.detach().cpu())
	scores_0.append(scores_0_.detach().cpu())
	true_scores_0.append(true_scores_0_.detach().cpu())
	samples_1 = target_1.sample((args.split_size,))
	log_probs_1_, scores_1_ = ebm.log_prob_and_grad(times[-1] * t_ones, samples_1)
	true_log_probs_1_, true_scores_1_ = target_1.log_prob_and_grad(samples_1)
	log_probs_1.append(log_probs_1_.detach().cpu())
	true_log_probs_1.append(true_log_probs_1_.detach().cpu())
	scores_1.append(scores_1_.detach().cpu())
	true_scores_1.append(true_scores_1_.detach().cpu())
samples = torch.cat(samples, dim=0)
log_probs_0 = torch.cat(log_probs_0, dim=0)
scores_0 = torch.cat(scores_0, dim=0)
true_log_probs_0 = torch.cat(true_log_probs_0, dim=0)
true_scores_0 = torch.cat(true_scores_0, dim=0)
log_probs_1 = torch.cat(log_probs_1, dim=0)
scores_1 = torch.cat(scores_1, dim=0)
true_log_probs_1 = torch.cat(true_log_probs_1, dim=0)
true_scores_1 = torch.cat(true_scores_1, dim=0)

# Compute the metrics
results = target_1.compute_metrics(samples)
results['log_prob'] = target_1.log_prob(samples)
results['samples'] = target_1.compute_psi_phi(samples)

# Compute the Fisher on both ends
results['fisher_0'] = torch.mean(torch.sum(torch.square(scores_0 - true_scores_0), dim=(-1, -2)))
results['fisher_1'] = torch.mean(torch.sum(torch.square(scores_1 - true_scores_1), dim=(-1, -2)))

# Compute ESS on both ends
log_weights_0 = log_probs_0 - true_log_probs_0
results['log_probs_0'] = log_probs_0
results['true_log_probs_0'] = true_log_probs_0
results['ess_0'] = torch.exp(2. * torch.logsumexp(log_weights_0, dim=0) \
	- torch.logsumexp(2. * log_weights_0, dim=0)) / args.n_samples
log_weights_1 = log_probs_1 - true_log_probs_1
results['log_probs_1'] = log_probs_1
results['true_log_probs_1'] = true_log_probs_1
results['ess_1'] = torch.exp(2. * torch.logsumexp(log_weights_1, dim=0) \
	- torch.logsumexp(2. * log_weights_1, dim=0)) / args.n_samples

# Make plots
os.makedirs(f"{args.results_path}/plots", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
target_1.plot_samples(ax[0], samples, label="model")
target_1.plot_samples(ax[1], target_1.sample((samples.shape[0], )).clone().detach().cpu(),
	label="ground truth")
fig.savefig(args.results_path + "/plots/{}.png".format(filename.replace('pkl','png')))

# Save the results
with open(args.results_path + '/' + filename, 'wb') as f:
    pickle.dump({'config': config, 'config_vel' : d_velocity['config'],
    	'config_en' : d_score['config'], 'results' : results }, f)
