# Libraries
import argparse
import numpy as np
import os
import pprint
import random
import torch
import math
import pickle
from diffclf.distr.gauss import TwoModes, FourtyModesMOG, standardize_gauss
from diffclf.distr.utils import log_prob_gaussian, log_prob_and_grad_mog
from diffclf.networks.ebm import SIEnergyDenoiserNet, EBM
from diffclf.networks.mlp import ImprovedFourierNet
from diffclf.networks.utils import init_bias_uniform_zeros, kaiming_uniform_zeros_
from diffclf.si.stochastic_interpolant import SimpleStochasticInterpolant
from diffclf.smc.pdds import pdds_sampler

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--ckpt_filepath', type=str)
parser.add_argument('--dim', type=int)
parser.add_argument('--n_levels', type=int, default=512)
parser.add_argument('--n_particles', type=int, default=8192)
parser.add_argument('--diff', type=float, default=1e-2)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

# Save the arguments in a dictionnary
config = vars(args)

# Make a Pytorch device
device = torch.device('cuda')

# Set the seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Make the results folder
os.makedirs(args.results_path, exist_ok=True)

# Make a filename
ckpt_filename = args.ckpt_filepath.split('/')[-1]
k = None
for k in range(1, 32):
    if '_k_' + str(k) in ckpt_filename:
        k = k
        break
filename = 'recalibration_si_gmm'
filename += '_dim_' + str(args.dim)
if '_sm_' in args.ckpt_filepath:
    filename += '_loss_sm'
    config['loss_type'] = 'sm'
elif '_tSM_' in args.ckpt_filepath:
    filename += '_loss_tsm'
    config['loss_type'] = 'tsm'
elif '_bilevel_' in args.ckpt_filepath:
    filename += '_loss_bi_level_k_' + str(k)
    config['loss_type'] = 'bi_level'
    config['k'] = 'k'
elif '_multilevel_' in args.ckpt_filepath:
    filename += '_loss_multi_level_k_' + str(k)
    config['loss_type'] = 'multi_level'
    config['k'] = 'k'
if 'uniform' in ckpt_filename:
    filename += '_uniform_weight'
    config['dsm_weight_type'] = 'uniform'
if 'square' in ckpt_filename:
    filename += '_square_weight'
    config['dsm_weight_type'] = 'square'
filename += '_seed_{}.pkl'.format(args.seed)

# Print the configuration
pprint.pprint(config)

# Build the distributions
target_0 = standardize_gauss(FourtyModesMOG(dim=args.dim)).to(device)
target_1 = standardize_gauss(TwoModes(dim=args.dim)).to(device)

# Build the train times and normalizing constants
times = torch.linspace(1e-3, 1.0-1e-3, args.n_levels, device=device)

# Build the SI
si = SimpleStochasticInterpolant(
    drift_net=None,
    denoiser_net=None,
).to(device)
gamma_fn = lambda t: torch.sqrt(t * (1. - t))
gamma_dot_fn = lambda t: 0.5 / gamma_fn(t) * (1. - 2 * t)

# Build the auxiliary networks of the EBM
base_net = ImprovedFourierNet(
    dim=args.dim,
    dim_out=args.dim,
    num_layers=4,
    channels=64 if args.dim <= 20 else 256,
    last_bias_init=init_bias_uniform_zeros,
    last_weight_init=kaiming_uniform_zeros_,
    use_pos_embedding=False
)
add_net = ImprovedFourierNet(
    dim=args.dim,
    dim_out=1,
    num_layers=4,
    channels=64 if args.dim <= 20 else 256,
    last_bias_init=init_bias_uniform_zeros,
    last_weight_init=kaiming_uniform_zeros_,
    use_pos_embedding=False
)

# Load the weights
ckpt_data = torch.load(args.ckpt_filepath)
base_net.load_state_dict({
    k.replace('base_net.','') : v for k,v in ckpt_data.items() if 'base_net.' in k
})
add_net.load_state_dict({
    k.replace('add_net.','') : v for k,v in ckpt_data.items() if 'add_net.' in k
})

class NegativeNetwork(torch.nn.Module):
    """Negation of a network"""
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, t, x):
        return -self.net(t,x)

# Make the EBM
ebm = SIEnergyDenoiserNet(
    base_net=NegativeNetwork(base_net),
    add_net=NegativeNetwork(add_net),
    gamma_type='brownian'
)

# Move to device
ebm = ebm.to(device)

# Disable gradient with respect to the parameters
for p in ebm.parameters():
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
        mask_0 = self.check_t_0(t)
        mask_1 = self.check_t_1(t)
        if torch.any(mask_0) or torch.any(mask_1):
            ebm_en = self.base_ebm.energy(t, x)
            if torch.any(mask_0):
                ret = torch.where(
                    mask_0.flatten(),
                    -self.dist0.log_prob(x),
                    ebm_en
                )
            if torch.any(mask_1):
                ret = torch.where(
                    mask_1.flatten(),
                    -self.dist1.log_prob(x),
                    ebm_en
                )
            return ret
        else:
            return self.base_ebm.energy(t, x)

    def score(self, t, x):
        """Compute the score."""
        mask_0 = self.check_t_0(t)
        mask_1 = self.check_t_1(t)
        if torch.any(mask_0) or torch.any(mask_1):
            ret = self.base_ebm.score(t, x)
            if torch.any(mask_0):
                ret = torch.where(
                    mask_0,
                    self.dist0.score(x),
                    ret
                )
            if torch.any(mask_1):
                ret = torch.where(
                    mask_1,
                    self.dist1.score(x),
                    ret
                )
            return ret
        else:
            return self.base_ebm.score(t, x)

    def denoiser(self, t, x):
        """Compute the denoiser."""
        """Evaluates the score of the distribution at (t,x)"""
        return -gamma_fn(t) * self.score(t, x)

    def log_prob_and_grad(self, t, x, return_denoiser=False):
        """Compute the log-prob and its gradient."""
        mask_0 = self.check_t_0(t)
        mask_1 = self.check_t_1(t)
        if torch.any(mask_0) or torch.any(mask_1):
            ret_log_prob, ret_grad = self.base_ebm.log_prob_and_grad(t, x)
            if torch.any(mask_0):
                log_prob0, grad0 = self.dist0.log_prob_and_grad(x)
                ret_log_prob = torch.where(
                    mask_0.flatten(),
                    log_prob0,
                    ret_log_prob
                )
                ret_grad = torch.where(
                    mask_0,
                    grad0,
                    ret_grad
                )
            if torch.any(mask_1):
                log_prob1, grad1 = self.dist1.log_prob_and_grad(x)
                ret_log_prob = torch.where(
                    mask_1.flatten(),
                    log_prob1,
                    ret_log_prob
                )
                ret_grad = torch.where(
                    mask_1,
                    grad1,
                    ret_grad
                )
            if return_denoiser:
                return ret_log_prob, -gamma_fn(t) * ret_grad
            else:
                return ret_log_prob, ret_grad
        else:
            return self.base_ebm.log_prob_and_grad(t, x)

# Harcode the EBM
ebm = HarcodeTargetEBM(ebm, target_0, times[0], target_1, times[-1])
ebm = ebm.to(device)

def marginal_params_gmm(t, gmm0, gmm1):
    """Get the marginal parameters of the GMM (non-batched)"""
    weights_t = (gmm0.weights.unsqueeze(1) * gmm1.weights.unsqueeze(0)).flatten()
    means_t = ((1. - t) * gmm0.means.unsqueeze(1) + t * gmm1.means.unsqueeze(0)).view((-1, args.dim))
    variances_t = (torch.square(1 - t) * gmm0.variances.unsqueeze(1) \
        + torch.square(t) * gmm1.variances.unsqueeze(0) + torch.square(gamma_fn(t))).view((-1, args.dim))
    return weights_t, means_t, variances_t

def marginal_params_dot_gmm(t, gmm0, gmm1):
    """Get the time derivatives of the marginal parameters of the GMM (non-batched)"""
    means_t = (-gmm0.means.unsqueeze(1) +  gmm1.means.unsqueeze(0)).view((-1, args.dim))
    variances_t = 2. * (-(1. - t) * gmm0.variances.unsqueeze(1) \
        + t * gmm1.variances.unsqueeze(0) + si.gamma_dot_times_gamma(t)).view((-1, args.dim))
    return means_t, variances_t

def log_prob_and_grad_non_batched(t, x, gmm0, gmm1):
    """Get the log-likelihood and gradient of the marginal (non-batched)"""
    weights_t, means_t, variances_t = marginal_params_gmm(t, gmm0, gmm1)
    weights_t = weights_t / weights_t.sum()
    return log_prob_and_grad_mog(x, weights_t, means_t, variances_t, return_log_prob=True)

def drift_fn_non_batched(t, x, gmm0, gmm1, return_log_prob_and_grad=False):
    """Get the optimal drift (non-batched)"""
    weights_t, means_t, variances_t = marginal_params_gmm(t, gmm0, gmm1)
    means_dot_t, variances_dot_t = marginal_params_dot_gmm(t, gmm0, gmm1)
    weights_t = weights_t / weights_t.sum()
    log_probs = log_prob_gaussian(x.unsqueeze(1), means_t, variances_t)
    log_probs += torch.log(weights_t.unsqueeze(0))
    probs = torch.nn.functional.softmax(log_probs, dim=-1).unsqueeze(-1)
    x_standardized = (x.unsqueeze(1) - means_t.unsqueeze(0)) / variances_t.unsqueeze(0)
    drift = torch.sum(
        probs * (means_dot_t.unsqueeze(0) + 0.5 * variances_dot_t.unsqueeze(0) * x_standardized), dim=1
    )
    if return_log_prob_and_grad:
        log_prob = torch.logsumexp(log_probs, dim=-1)
        grad = -torch.sum(probs * x_standardized, dim=1)
        return drift, log_prob, grad
    else:
        return drift

# Vectorize everything
log_prob_and_grad = torch.vmap(
    lambda t, x : tuple(y.squeeze(0).squeeze(0) for y in log_prob_and_grad_non_batched(t, x.unsqueeze(0), target_0, target_1))
)
drift_fn_ = torch.vmap(
    lambda t, x : drift_fn_non_batched(t, x.unsqueeze(0), target_0, target_1, False).squeeze(0),
)
drift_fn_with_aux_ = torch.vmap(
    lambda t, x : tuple(y.squeeze(0).squeeze(0) for y in drift_fn_non_batched(t, x.unsqueeze(0), target_0, target_1, True)),
)
def drift_fn(t, x, return_log_prob_and_grad=False):
    if return_log_prob_and_grad:
        return drift_fn_with_aux_(t, x)
    else:
        return drift_fn_(t, x)

def velocity_fn(t, x, return_log_prob_and_grad=False):
    """Make the velocity function"""
    drift, log_prob, score = drift_fn(t, x, return_log_prob_and_grad=True)
    velocity = drift + si.gamma_dot_times_gamma(t) * score
    if return_log_prob_and_grad:
        return velocity, log_prob, score
    else:
        return velocity

def backward_kernel(x_t, s, t, grad_t, aux_t):
    """Backward kernel"""
    return si.backward_sde_kernel(s, t, x_t, None, aux_t['velocity'], grad_t, args.diff,
        return_log_prob=True)

def forward_kernel(x_t, x_s, grad_s, s, t, aux_s):
    """Forward kernel"""
    mean_t_s, var_t_s = si.forward_sde_kernel(s, t, x_s, None, aux_s['velocity'], grad_s, args.diff,
        return_mean_var=True)
    log_prob = -0.5 * torch.sum(torch.square(mean_t_s - x_t) / var_t_s, dim=-1)
    log_prob -= 0.5 * args.dim * math.log(2. * math.pi)
    log_prob -= 0.5 * args.dim * torch.log(var_t_s).flatten()
    return log_prob

# Run SMC
samples, weights, _, diags = pdds_sampler(
    x_init=target_1.sample((args.n_particles,)),
    times=times.view((-1, 1, 1)).repeat((1, args.n_particles, 1)),
    log_prob_and_grads=ebm.log_prob_and_grad,
    kernel_aux_fn=lambda t, y : { 'velocity' : velocity_fn(t, y) },
    sde=None,
    n_warmup_mcmc_steps=0,
    n_mcmc_steps=0,
    step_sizes_per_noise=None,
    reweight_threshold=0.3,
    ignore_mcmc=True,
    integrator_fn=backward_kernel,
    forward_kernel=forward_kernel,
    verbose=True
)
samples = samples.squeeze(0)

# Save the results
with open(args.results_path + '/' + filename, 'wb') as f:
    pickle.dump({
        'config': config,
        'metrics': target_0.compute_metrics(samples, weights, compute_standard_metrics=True),
        'weights': weights.cpu(),
        'log_probs' : target_0.log_prob(samples).cpu(),
        'ess' : diags['ess'].cpu()
    }, f)
