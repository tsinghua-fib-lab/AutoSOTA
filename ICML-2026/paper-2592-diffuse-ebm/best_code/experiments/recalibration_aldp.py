# Libraries
import argparse
import numpy as np
import os
import pickle
import pprint
import random
import torch
import matplotlib.pyplot as plt
from functools import partial
from diffclf.distr.aldp import AlanineDipeptide
from diffclf.networks.ebm import EBM, EDMEnergyPreconditioning, DotEBM
from diffclf.networks.egnn import EGNN_atom
from diffclf.sde.diffusion import VE, EDM
from diffclf.smc.diffusion_ais import diffusion_ais_sampler
from diffclf.smc.pdds import pdds_sampler
from diffclf.utils.se3_utils import remove_mean

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--ckpt_filepath', type=str)
parser.add_argument('--algo', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--vacuum_datapath', type=str, default='')
parser.add_argument('--n_particles', type=int, default=8192)
parser.add_argument('--split_size', type=int, default=2048)
parser.add_argument('--seed', type=int)
parser.add_argument('--use_ema', action=argparse.BooleanOptionalAction)

args = parser.parse_args()

# Load the checkpoint
with open(args.ckpt_filepath, 'rb') as f:
    # Load the data
    ckpt_data = pickle.load(f)
    # Parse the config
    bilevel_type = ckpt_data['config']['bilevel_type']
    loss_type = ckpt_data['config']['loss_type']
    env = ckpt_data['config_pkl']['env']
    sde_type = ckpt_data['config_pkl']['sde_type']
    n_levels = ckpt_data['config_pkl']['n_levels']
    hidden_nf = ckpt_data['config_pkl']['hidden_nf']
    n_layers = ckpt_data['config_pkl']['n_layers']

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

os.makedirs(args.results_path, exist_ok=True)

# Make a filename
filename = 'recalibration_aldp_'
filename += '_algo_' + args.algo
filename += '_loss_' + loss_type
if loss_type != "sm":
	filename += '_bilevel_type_' + bilevel_type
filename += '_sde_type_' + sde_type
filename += '_env_' + env
if args.use_ema:
	filename += '_use_ema'
filename += '_seed_{}.pkl'.format(args.seed)

# Make the target distribution
target = AlanineDipeptide(args.data_path, env=env).to(device)
if env == 'vacuum':
    if len(args.vacuum_datapath) == 0:
        print('Provide the path to vaccum data.')
        exit(0)
    else:
        target.load_data(remove_mean(
            torch.load(args.vacuum_datapath).view((-1, *target.data_shape))
        ))

# Build an SDE
if sde_type == 've':
    sde = VE(sigma_min=1e-4)
    t_limits = (sde.sigma_inv(sde.sigma_min), sde.sigma_inv(sde.sigma_max))
else:
    sde = EDM()
    t_limits = (sde.sigma_inv(sde.sigma_min), sde.sigma_inv(sde.sigma_max))
sde = sde.to(device)

# Build the train times and normalizing constants
ts = sde.get_snr_time_discretization(*t_limits, n_levels).to(device)

# Make the times
times = ts.view((-1, 1, 1, 1)).repeat((1, args.n_particles, 1, 1))

# Compute the scalar variance
data_var = target.variance()
data_var_scalar = data_var.mean()
data_mean = torch.zeros_like(target.mean())

# Build the EBM
base_net = EGNN_atom(
    n_particles=target.n_particles,
    n_dimension=target.n_dimensions,
    atom_type_labels=target.get_atom_chemical_types(),
    bonds=target.get_bonds(),
    hidden_nf=hidden_nf,
    time_embedding_dim=128,
    atom_type_embedding_dim=64,
    n_layers=n_layers,
    recurrent=False,
    attention=True,
    tanh=False,
    use_pos_embedding=True
)
base_ebm = DotEBM(base_net, sde=sde)
ebm = EDMEnergyPreconditioning(base_ebm, sde, data_mean, data_var_scalar,
    is_particles=True, log_snr_dist=None)

class HarcodeTargetEBM(EBM):
	"""Hardcode the target"""

	def __init__(self, base_ebm, target, t_0):
		super().__init__(build_score=False, build_log_prob_dot=False, build_grad_and_log_prob=False,
			build_log_prob_and_grad_and_dot=False)
		self.base_ebm = base_ebm
		self.target = target
		self.register_buffer('t_0', t_0)

	def check_t(self, t):
		"""Check if the time is t_0."""
		return torch.abs(t - self.t_0) < 1e-5

	def energy(self, t, x):
		"""Compute the energy."""
		if torch.any(self.check_t(t)):
			return torch.where(
				self.check_t(t).flatten(),
				-self.target.log_prob(x),
				self.base_ebm.energy(t, x)
			)
		else:
			return self.base_ebm.energy(t, x)

	def score(self, t, x):
		"""Compute the score."""
		data_shape_ones = (1,) * (len(x.shape)-1)
		if torch.any(self.check_t(t)):
			return torch.where(
				self.check_t(t).view((-1, *data_shape_ones)),
				self.target.score(x),
				self.base_ebm.score(t, x)
			)
		else:
			return self.base_ebm.score(t, x)

	def denoiser(self, t, x):
		"""Compute the denoiser."""
		if torch.any(self.check_t(t)):
			raise ValueError('return_denoiser not supported at t = t_0.')
		else:
			return self.base_ebm.denoiser(t, x)

	def log_prob_and_grad(self, t, x, return_denoiser=False):
		"""Compute the log-prob and its gradient."""
		if torch.any(self.check_t(t)):
			if return_denoiser:
				raise ValueError('return_denoiser not supported at t = t_0.')
			data_shape = x.shape[1:]
			mask = self.check_t(t)
			log_prob_ebm, grad_ebm = self.base_ebm.log_prob_and_grad(t, x)
			log_prob_target, grad_target = self.target.log_prob_and_grad(x)
			return torch.where(mask.flatten(),
				log_prob_target,
				log_prob_ebm
			), torch.where(
				mask.expand((-1, *data_shape)),
				grad_target,
				grad_ebm
			)
		else:
			return self.base_ebm.log_prob_and_grad(t, x, return_denoiser=return_denoiser)

# Load the parameters
if args.use_ema:
	ebm.load_state_dict({
		k.replace('module.ebm.','') : v for k,v in ckpt_data['params_ema'].items() if 'module.ebm.' in k
	})
else:
	ebm.load_state_dict({
		k[4:] : v for k,v in ckpt_data['params'].items() if k[:4] == 'ebm.'
	})

# Harcode the EBM
ebm = HarcodeTargetEBM(ebm, target, t_0=ts[0])

# Handle the memory better
class MemoryEBM(torch.nn.Module):

	def __init__(self, base_ebm, split_size):
		super().__init__()
		self.base_ebm = base_ebm
		self.split_size = split_size

	def get_chunksize(self, batch_size):
		return batch_size // self.split_size

	def energy(self, t, x):
		chunk_size = self.get_chunksize(x.shape[0])
		return torch.cat([self.base_ebm.energy(t_, x_)
			for t_, x_ in zip(torch.chunk(t, chunk_size), torch.chunk(x, chunk_size))], dim=0)

	def score(self, t, x):
		chunk_size = self.get_chunksize(x.shape[0])
		return torch.cat([self.base_ebm.score(t_, x_)
			for t_, x_ in zip(torch.chunk(t, chunk_size), torch.chunk(x, chunk_size))], dim=0)

	def denoiser(self, t, x):
		chunk_size = self.get_chunksize(x.shape[0])
		return torch.cat([self.base_ebm.denoiser(t_, x_)
			for t_, x_ in zip(torch.chunk(t, chunk_size), torch.chunk(x, chunk_size))], dim=0)

	def log_prob_and_grad(self, t, x, return_denoiser=False):
		chunk_size = self.get_chunksize(x.shape[0])
		log_probs, others = [], []
		for t_, x_ in zip(torch.chunk(t, chunk_size), torch.chunk(x, chunk_size)):
			log_prob, other = self.base_ebm.log_prob_and_grad(t_, x_, return_denoiser=return_denoiser)
			log_probs.append(log_prob)
			others.append(other)
		return torch.cat(log_probs, dim=0), torch.cat(others, dim=0)

ebm = MemoryEBM(ebm, split_size=args.split_size).to(device)

# Disable gradient with respect to the parameters
for p in ebm.parameters():
    p.requires_grad_(False)

# List the different integration kernels
def denoiser_from_score(t_, x_, t, x, grad_t_x):
	alpha_t, gamma_sq_t = sde.transition_params_from_data(t)
	return (gamma_sq_t * grad_t_x + x) / alpha_t
kernels = {
	# 'em' : partial(sde.em_integration_step, return_log_prob=True),
	'ei' : partial(sde.ei_integration_step, return_log_prob=True),
	'ddpm' : partial(sde.ddpm_integration_step, return_log_prob=True, use_forward_var=True),
	'ddim' : lambda x, s, t, grad : sde.ddim_integration_step(x, s, t, return_log_prob=True,
		post_sampler_fn=lambda t_, x_ : denoiser_from_score(t_, x_, t, x, grad)) 
}

# Prepare the storage
results = {}

# Get the marginals
base_dist = sde.get_base_dist(data_shape=(22,3))
base_log_prob_and_grad = base_dist.log_prob_and_grad
target_log_prob_and_grad = target.log_prob_and_grad
grads = ebm.score
log_prob_and_grads = ebm.log_prob_and_grad

# Get the initial samples
x_init = base_dist.sample((args.n_particles,))
x_init = remove_mean(x_init)

# Get samples and weights
results = {}
for kernel_name, integrator_fn in kernels.items():
	print('Sampling with ' + kernel_name)
	if args.algo == 'ais':
		samples, weights = diffusion_ais_sampler(x_init.clone(), times, base_log_prob_and_grad,
			target_log_prob_and_grad, grads, sde, integrator_fn=integrator_fn, is_particles=True, verbose=True)
	else:
		samples, weights = pdds_sampler(x_init.clone(), times, log_prob_and_grads, sde, integrator_fn,
			0, 0, None, ignore_mcmc=True, reweight_threshold=0.3, is_particles=True, verbose=True)[:2]
		samples = samples[-1]
	# Compute the metrics
	results[kernel_name] = target.compute_metrics(samples, weights=weights)
	results[kernel_name]['log_prob'] = target.log_prob(samples)
	results[kernel_name]['samples'] = target.compute_psi_phi(samples)
	results[kernel_name]['weights'] = weights.detach().cpu()
	# Make plots
	filename_ = args.ckpt_filepath.split('/')[-1][:-4]
	os.makedirs(f"{args.results_path}/plots", exist_ok=True)
	fig, ax = plt.subplots(1, 2, figsize=(10, 5))
	target.plot_samples(ax[0], samples.clone().detach().cpu(), label="model",
		weights=weights.clone().detach().cpu())
	target.plot_samples(ax[1], target.sample((samples.shape[0], )).clone().detach().cpu(),
		label="ground truth")
	if args.use_ema:
		fig.savefig(f"{args.results_path}/plots/{filename_}_ema_{args.algo}_{kernel_name}_{args.seed}.png")
	else:
		fig.savefig(f"{args.results_path}/plots/{filename_}_{args.algo}_{kernel_name}_{args.seed}.png")

# Save the results
with open(args.results_path + '/' + filename, 'wb') as f:
    pickle.dump({'config': config, 'config_pkl' : ckpt_data['config_pkl'],
    	'config_en' : ckpt_data['config'], 'results' : results }, f)
