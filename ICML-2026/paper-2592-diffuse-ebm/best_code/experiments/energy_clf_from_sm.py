# Libraries
import argparse
import numpy as np
import os
import pickle
import pprint
import random
import torch
import matplotlib.pyplot as plt
import math
from diffclf.distr.gauss import FourtyModesMOG, TwoModes, standardize_gauss
from diffclf.em.bi_level import compute_loss_both as compute_loss_both_bi_level
from diffclf.em.multi_level import compute_loss_both as compute_loss_both_multi_level
from diffclf.em.multi_level import compute_loss_edm
from diffclf.em.multi_level import compute_loss_multi_level
from diffclf.em.time_sm import compute_loss_both as compute_loss_both_tsm
from diffclf.em.cond_nce import compute_loss_both as compute_loss_both_cond_nce
from diffclf.em.cond_nce import compute_loss_cond_nce
from diffclf.em.bi_level_bregman import compute_loss_both as compute_loss_both_bi_level_bregman
from diffclf.em.multi_level_bregman import compute_loss_both as compute_loss_both_multi_level_bregman
from diffclf.em.rne import compute_loss_both as compute_loss_both_rne
from diffclf.networks.ebm import EDMEnergyPreconditioning
from diffclf.networks.ebm import DotEBM
from diffclf.networks.mlp import ImprovedFourierNet
from diffclf.networks.utils import init_bias_uniform_zeros, kaiming_uniform_zeros_
from diffclf.sde.diffusion import LinearVP, EDM
from diffclf.sde.utils import TimeSampler
from tqdm import tqdm, trange

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--cpkt_filepath', type=str)
parser.add_argument('--loss_type', type=str)
parser.add_argument('--k', type=int)
parser.add_argument('--reg_val', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--dataset_size', type=int, default=60000)
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--n_eval_samples', type=int, default=4096)
parser.add_argument('--seed', type=int)
parser.add_argument('--bregman_type', type=str, default="jensen_shannon")
args = parser.parse_args()

# Load the checkpoint
with open(args.cpkt_filepath, 'rb') as f:
    # Load the data
    ckpt_data = pickle.load(f)
    # Parse the config
    target_type = ckpt_data['config']['target_type']
    dim = ckpt_data['config']['dim']
    n_levels = ckpt_data['config']['n_levels']
    dataset_size = ckpt_data['config']['dataset_size']
    if 'sde_type' in ckpt_data['config']:
        sde_type = ckpt_data['config']['sde_type']
    else:
        sde_type = 'vp'

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
filename = 'energy_clf_final'
filename += '_target_type_' + target_type
filename += '_dim_' + str(dim)
filename += '_loss_' + args.loss_type
if args.loss_type == 'multi_level_bregman':
    filename += args.bregman_type
filename += '_k_' + str(args.k)
filename += '_seed_{}.pkl'.format(args.seed)

# Make the target distribution
if target_type == 'two_modes':
    target = TwoModes(dim=dim, a=5.0)
else:
    target = FourtyModesMOG(dim=dim)
target = standardize_gauss(target).to(device)

# Build an SDE
if sde_type == 'vp':
    sde = LinearVP(beta_max=20.0).to(device)
    t_limits = (1e-4, sde.T-1e-4)
elif sde_type == 'edm':
    sde = EDM(sigma_min=1e-3, sigma_max=10.0).to(device)
    t_limits = (sde.sigma_inv(sde.sigma_min), sde.sigma_inv(sde.sigma_max))
else:
    raise ValueError(f"Invalid SDE type: {sde_type}")

# Build the train times and normalizing constants
times = torch.linspace(*t_limits, n_levels).to(device)
log_snr_times = sde.log_snr(times)
log_snr_dist = (log_snr_times.mean().item(), log_snr_times.std().item())
time_sampler = TimeSampler(times=times, log_snr_dist=log_snr_dist).to(device)
f = torch.zeros((n_levels,), requires_grad=True, device=device)

# Build the evaluation times
eval_times = torch.linspace(*t_limits, n_levels).to(device)

# Compute the scalar variance
data_var = target.variance()
data_var_scalar = data_var.mean()
data_mean = target.mean()

# Build the EBM
base_net = ImprovedFourierNet(
    dim=dim,
    dim_out=dim,
    num_layers=4,
    channels=128,
    last_bias_init=init_bias_uniform_zeros,
    last_weight_init=kaiming_uniform_zeros_,
    use_pos_embedding=True
)
base_ebm = DotEBM(base_net, sde=sde)
ebm = EDMEnergyPreconditioning(base_ebm, sde, data_mean, data_var_scalar,
    log_snr_dist=time_sampler.log_snr_dist)
ebm = ebm.cpu()

# Load the parameters
ebm.load_state_dict(ckpt_data['ebm'])

# Move to device
ebm = ebm.to(device)

class OptimalEBM(torch.nn.Module):
    """Build the optimal EBM"""
    def __init__(self, target, sde):
        super().__init__()
        self.target = target
        self.sde = sde
    def log_prob(self, t, x):
        return self.target.marginal_log_prob_and_grad(t, x, sde=self.sde)[0]
    def score(self, t, x):
        return self.target.marginal_log_prob_and_grad(t, x, sde=self.sde)[1]
    def log_prob_and_grad(self, t, x):
        return self.target.marginal_log_prob_and_grad(t, x, sde=self.sde)

# Instantiate the optimal EBM
opt_ebm = OptimalEBM(target, sde).to(device)
opt_f = torch.zeros((n_levels,), device=device)

# Precompute the grid
i_idx, j_idx = torch.meshgrid(
    torch.arange(args.k, device=device),
    torch.arange(args.k, device=device),
    indexing='ij'
)
i_idx = i_idx.reshape(-1)
j_idx = j_idx.reshape(-1)
diag_mask = torch.eye(args.k, dtype=torch.bool, device=device).flatten()

def compute_fisher_divergence(ebm, batch_size=args.n_eval_samples):
    """Compute the fisher divergence"""
    losses = torch.empty((n_levels,), device=device)
    t_ones = torch.ones((batch_size, 1), device=device)
    for i, t in tqdm(enumerate(eval_times)):
        # Get the noisy sample
        t_ = t_ones * eval_times[i]
        x = target.marginal_sample(t_ones * eval_times[i], sde)
        # Compute the square norm
        losses[i] = torch.sum(torch.square(ebm.score(t_, x) - opt_ebm.score(t_, x)), dim=-1).mean() / dim
    return losses.cpu()

def compute_ebm_ess(ebm, n_particles=args.n_eval_samples):
    """Compute the ESS at every time"""
    esses = torch.empty((eval_times.shape[0],))
    t_ones = torch.ones((n_particles, 1), device=device)
    for i, t in tqdm(enumerate(eval_times)):
        # Get the time
        t_ = t * t_ones
        # Sample the marginal exactly
        x = target.marginal_sample(t_, sde)
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

# Build the optimizer
optimizer = torch.optim.AdamW((*ebm.parameters(), f), lr=1e-3, weight_decay=1e-4)
dataset = torch.utils.data.TensorDataset(target.sample((dataset_size,)))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Build the LR scheduler (cosine annealing with linear warmup)
total_steps = args.n_epochs * len(dataloader)
warmup_steps = int(0.05 * total_steps)
def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Build EMA of EBM parameters
ema_decay = 0.999
ema_ebm_params = {name: param.clone().detach() for name, param in ebm.named_parameters()}
ema_f = f.clone().detach()
for epoch in range(args.n_epochs):
    loop = tqdm(dataloader, leave=True)
    loop.set_description(f"Epoch {epoch+1}/{args.n_epochs}")
    for data in loop:
        optimizer.zero_grad()
        x0 = data[0]
        if args.loss_type in ['sm','tsm']:
            x0 = x0.unsqueeze(0).repeat((args.k, 1, 1)).view((-1, dim))
            if args.loss_type == 'sm':
                losses = compute_loss_edm(ebm, x0, time_sampler, sde, data_var_scalar)
            else:
                losses = compute_loss_both_tsm(ebm, x0, time_sampler, sde, data_var_scalar)
        elif args.loss_type == 'cnce':
            assert args.k == 2, "Conditional NCE requires k=2"
            losses = compute_loss_both_cond_nce(ebm, x0, f, time_sampler, sde, data_var_scalar)
        elif args.loss_type == 'bi_level':
            losses = compute_loss_both_bi_level(ebm, x0, f, time_sampler, sde, data_var_scalar)
        elif args.loss_type == 'bi_level_bregman':
            losses = compute_loss_both_bi_level_bregman(ebm, x0, f, time_sampler, sde, data_var_scalar, args.bregman_type)
        elif args.loss_type == 'multi_level_bregman':
            losses = compute_loss_both_multi_level_bregman(ebm, x0, f, args.k, i_idx, j_idx,
                diag_mask, time_sampler, sde, data_var_scalar, args.bregman_type)
        elif args.loss_type == 'rne':
            losses = compute_loss_both_rne(ebm, x0, f, time_sampler, sde, data_var_scalar)
        else:
            losses = compute_loss_both_multi_level(ebm, x0, f, args.k, i_idx, j_idx,
                diag_mask, time_sampler, sde, data_var_scalar)
        if isinstance(losses, tuple):
            sm_loss = losses[0].mean()
            other_loss = losses[1].mean()
            loss = sm_loss + args.reg_val * other_loss
        else:
            loss = losses.mean()
        loss.backward()
        # Clip gradients to 1.0 before optimizer step
        torch.nn.utils.clip_grad_norm_((*ebm.parameters(), f), max_norm=1.0)
        optimizer.step()
        # Update EMA of EBM parameters
        with torch.no_grad():
            for name, param in ebm.named_parameters():
                ema_ebm_params[name].mul_(ema_decay).add_(param, alpha=1.0 - ema_decay)
            ema_f.mul_(ema_decay).add_(f, alpha=1.0 - ema_decay)
            f -= f[-1].clone()
        scheduler.step()
        if args.loss_type == 'sm':
            loop.set_postfix(sm_loss=loss.item())
        elif args.loss_type == 'tsm':
            loop.set_postfix(loss=loss.item(), sm_loss=sm_loss.item(), tsm_loss=other_loss.item())
        elif args.loss_type == 'rne':
            loop.set_postfix(loss=loss.item(), sm_loss=sm_loss.item(), rne_loss=other_loss.item())
        elif args.loss_type == 'cnce':
            loop.set_postfix(loss=loss.item(), sm_loss=sm_loss.item(), cnce_loss=other_loss.item())
        elif args.loss_type[-7:] == 'bregman':
            loop.set_postfix(loss=loss.item(), sm_loss=sm_loss.item(), bregman_loss=other_loss.item())
        else:
            loop.set_postfix(loss=loss.item(), sm_loss=sm_loss.item(), clf_loss=other_loss.item())

# Disable gradient with respect to the parameters
for p in ebm.parameters():
    p.requires_grad_(False)

# Swap in EMA parameters for evaluation
orig_ebm_params = {name: param.clone().detach() for name, param in ebm.named_parameters()}
orig_f = f.clone().detach()
with torch.no_grad():
    for name, param in ebm.named_parameters():
        param.copy_(ema_ebm_params[name])
    f.copy_(ema_f)
if isinstance(f, torch.nn.Module):
    for p in f.parameters():
        p.requires_grad_(False)
else:
    f = f.detach()

# Compute the metrics
x0 = target.sample((args.n_eval_samples,))
metrics = {
    'multi_classif': compute_loss_multi_level(ebm, x0, f(eval_times) if callable(f) else f,
        eval_times, sde).mean().cpu().item(),
    'fisher': compute_fisher_divergence(ebm),
    'ess': compute_ebm_ess(ebm)
}
print(metrics)

# Sample the model
def sample_fn(t, x):
    """Sample the DDIM approximation"""
    mean = ebm.denoiser(t, x)
    return mean + sde.gamma(t) * torch.randn_like(x)
x = sde.sample_base_dist((args.n_eval_samples,), data_shape=(dim,))
times_ones = torch.ones((args.n_eval_samples, 1), device=device)
for i in trange(eval_times.shape[0]-1, 0, -1):
    x = sde.ddim_integration_step(x, eval_times[i-1] * times_ones, eval_times[i] * times_ones, sample_fn)
samples = x.clone().detach().cpu()
metrics_samples = target.compute_metrics(samples.to(device), compute_standard_metrics=True)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
true_samples = target.sample((args.n_eval_samples,)).cpu()
ax[0].scatter(true_samples[:, 0], true_samples[:, 1], s=1, alpha=0.5, label="true")
ax[0].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5, label="model")
ax[0].legend()
ax[1].plot(eval_times.cpu(), metrics['ess'].cpu())
ax[1].set_ylim(0, 1)
ax[1].set_title('ESS')
ax[1].set_xlabel('t')
os.makedirs(f"{args.results_path}/plots", exist_ok=True)
fig.savefig(f"{args.results_path}/plots/{filename[:-4]}.png")
print(metrics_samples)

# Save the results
with open(args.results_path + '/' + filename, 'wb') as f:
    pickle.dump({'config': config, 'metrics': metrics, 'metrics_samples': metrics_samples}, f)