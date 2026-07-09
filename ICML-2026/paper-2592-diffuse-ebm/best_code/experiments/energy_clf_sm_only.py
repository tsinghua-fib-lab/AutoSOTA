# Libraries
import argparse
import numpy as np
import os
import pickle
import pprint
import random
import torch
from diffclf.distr.gauss import FourtyModesMOG, TwoModes, standardize_gauss
from diffclf.em.multi_level import compute_loss_edm
from diffclf.networks.ebm import EDMEnergyPreconditioning
from diffclf.networks.ebm import DotEBM
from diffclf.networks.mlp import ImprovedFourierNet
from diffclf.networks.utils import init_bias_uniform_zeros, kaiming_uniform_zeros_
from diffclf.sde.diffusion import LinearVP, EDM
from diffclf.sde.utils import TimeSampler
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--dim', type=int)
parser.add_argument('--target_type', type=str)
parser.add_argument('--n_levels', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--dataset_size', type=int, default=60000)
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--seed', type=int)
parser.add_argument('--sde_type', type=str, default="vp")
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

os.makedirs(args.results_path, exist_ok=True)

# Make a filename
filename = 'energy_clf_sm_only'
filename += '_target_type_' + str(args.target_type)
filename += '_dim_' + str(args.dim)
filename += '_seed_{}.pkl'.format(args.seed)

# Make the target distribution
if args.target_type == 'two_modes':
    target = TwoModes(dim=args.dim, a=5.0)
else:
    target = FourtyModesMOG(dim=args.dim)
target = standardize_gauss(target).to(device)

# Build an SDE
if args.sde_type == 'vp':
    sde = LinearVP(beta_max=20.0).to(device)
    t_limits = (1e-4, sde.T-1e-4)
elif args.sde_type == 'edm':
    sde = EDM(sigma_min=1e-3, sigma_max=10.0).to(device)
    t_limits = (sde.sigma_inv(sde.sigma_min), sde.sigma_inv(sde.sigma_max))
else:
    raise ValueError(f"Invalid SDE type: {args.sde_type}")
config['sde_type'] = args.sde_type

# Build the train times and normalizing constants
times = torch.linspace(*t_limits, args.n_levels).to(device)
log_snr_times = sde.log_snr(times)
log_snr_dist = (log_snr_times.mean().item(), log_snr_times.std().item())
time_sampler = TimeSampler(times=times, log_snr_dist=log_snr_dist).to(device)

# Build the evaluation times
eval_times = torch.linspace(*t_limits, args.n_levels).to(device)

# Compute the scalar variance
data_var = target.variance()
data_var_scalar = data_var.mean()
data_mean = target.mean()

# Build the EBM
base_net = ImprovedFourierNet(
    dim=args.dim,
    dim_out=args.dim,
    num_layers=4,
    channels=128,
    last_bias_init=init_bias_uniform_zeros,
    last_weight_init=kaiming_uniform_zeros_,
    use_pos_embedding=True
)
base_ebm = DotEBM(base_net, sde=sde)
ebm = EDMEnergyPreconditioning(base_ebm, sde, data_mean, data_var_scalar,
    log_snr_dist=time_sampler.log_snr_dist)
ebm = ebm.to(device)

# Build the optimizer
optimizer = torch.optim.Adam(ebm.parameters(), lr=1e-3)
dataset = torch.utils.data.TensorDataset(target.sample((args.dataset_size,)))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
for epoch in range(args.n_epochs):
    loop = tqdm(dataloader, leave=True)
    loop.set_description(f"Epoch {epoch+1}/{args.n_epochs}")
    for data in loop:
        optimizer.zero_grad()
        loss = compute_loss_edm(ebm, data[0], time_sampler, sde, data_var_scalar)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        loop.set_postfix(sm_loss=loss.item())

# Disable gradient with respect to the parameters
for p in ebm.parameters():
    p.requires_grad_(False)

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

opt_ebm = OptimalEBM(target, sde).to(device)

def compute_ebm_ess(ebm, n_particles=10000):
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

# Compute the metrics
x0 = target.sample((10000,))
metrics = {
    # 'multi_classif': compute_loss_multi_level(ebm, x0, f(eval_times) if callable(f) else f,
    #     eval_times, sde).mean().cpu().item(),
    # 'fisher': compute_fisher_divergence(ebm),
    'ess': compute_ebm_ess(ebm)
}
print(metrics)

# Sample the model
def sample_fn(t, x):
    """Sample the DDIM approximation"""
    mean = ebm.denoiser(t, x)
    return mean + sde.gamma(t) * torch.randn_like(x)
x = sde.sample_base_dist((x0.shape[0],), data_shape=(args.dim,))
times_ones = torch.ones((x0.shape[0], 1), device=device)
for i in trange(eval_times.shape[0]-1, 0, -1):
    x = sde.ddim_integration_step(x, eval_times[i-1] * times_ones, eval_times[i] * times_ones, sample_fn)
samples = x.clone().detach().cpu()
metrics_samples = target.compute_metrics(samples.to(device), compute_standard_metrics=True)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
true_samples = target.sample((x0.shape[0],)).cpu()
ax[0].scatter(true_samples[:, 0], true_samples[:, 1], s=1, alpha=0.5, label="true")
ax[0].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5, label="model")
ax[0].legend()
ax[1].plot(eval_times.cpu(), metrics['ess'].cpu())
ax[1].set_ylim(0, 1)
ax[1].set_title('ESS')
ax[1].set_xlabel('t')
os.makedirs(f"{args.results_path}/plots", exist_ok=True)
fig.savefig(f"{args.results_path}/plots/{filename[:-4]}.png")

# Move EBM to CPU
ebm = ebm.cpu()

# Save the results
with open(args.results_path + '/' + filename, 'wb') as f:
    pickle.dump({ 'config': config, 'ebm': ebm.state_dict() }, f)