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
from diffclf.em.bi_level import compute_loss_both as compute_loss_both_bi_level
from diffclf.em.multi_level import compute_loss_edm
from diffclf.networks.ebm import EDMEnergyPreconditioning
from diffclf.networks.ebm import DotEBM
from diffclf.networks.egnn import EGNN_atom
from diffclf.sde.diffusion import VE, EDM
from diffclf.sde.utils import TimeSampler
from diffclf.utils.se3_utils import remove_mean
from tqdm import tqdm, trange

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--ckpt_filepath', type=str)
parser.add_argument('--loss_type', type=str)
parser.add_argument('--bilevel_type', type=str, default="uniform")
parser.add_argument('--data_path', type=str)
parser.add_argument('--vacuum_datapath', type=str, default='')
parser.add_argument('--reg_val', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--batch_size_eval', type=int, default=512)
parser.add_argument('--dataset_size', type=int, default=250000)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--n_eval_samples', type=int, default=8192)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

# Load the checkpoint
with open(args.ckpt_filepath, 'rb') as f:
    # Load the data
    ckpt_data = pickle.load(f)
    # Parse the config
    env = ckpt_data['config']['env']
    sde_type = ckpt_data['config']['sde_type']
    n_levels = ckpt_data['config']['n_levels']
    hidden_nf = ckpt_data['config']['hidden_nf']
    n_layers = ckpt_data['config']['n_layers']

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
filename = 'energy_clf_aldp'
filename += '_loss_' + args.loss_type
if args.loss_type != "sm":
    filename += f'_{args.bilevel_type}_'
filename += '_sde_type_' + sde_type
filename += '_env_' + env
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
times = sde.get_snr_time_discretization(*t_limits, n_levels).to(device)
log_snr_times = sde.log_snr(times)
log_snr_dist = (log_snr_times.mean().item(), log_snr_times.std().item())
time_sampler = TimeSampler(times=times, log_snr_dist=log_snr_dist).to(device)

# Build the evaluation times
eval_times = sde.get_snr_time_discretization(*t_limits, n_levels).to(device)

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

# Load the parameters
ebm.load_state_dict(ckpt_data['ebm'])

# Wrap f into the model
class WrapF(torch.nn.Module):
    def __init__(self, ebm, n_levels):
        super().__init__()
        self.ebm = ebm
        self.f = torch.nn.Parameter(torch.zeros((n_levels,)))
    def denoiser(self, t, x):
        return self.ebm.denoiser(t, x)
    def log_prob_and_grad(self, t, x, return_denoiser=False):
        return self.ebm.log_prob_and_grad(t, x, return_denoiser=return_denoiser)
    def log_prob(self, t, x):
        return self.ebm.log_prob(t, x)
ebm = WrapF(ebm, n_levels).to(device)

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

# Build the optimizer
optimizer = torch.optim.Adam(ebm.parameters(), lr=args.lr)
target_samples = target.sample((args.dataset_size,))
target_samples = remove_mean(target_samples)
dataset = torch.utils.data.TensorDataset(target_samples)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
for epoch in range(args.n_epochs):
    loop = tqdm(dataloader, leave=True)
    loop.set_description(f"Epoch {epoch+1}/{args.n_epochs}")
    for data in loop:
        optimizer.zero_grad()
        x0 = data[0]
        if args.loss_type == 'sm':
            x0 = x0.unsqueeze(0).repeat((2, 1, 1, 1)).view((-1, 22, 3))
            losses = compute_loss_edm(ebm, x0, time_sampler, sde, data_var_scalar, is_particles=True)
        else:
            losses = compute_loss_both_bi_level(ebm, x0, ebm.f, time_sampler, sde, data_var_scalar,
                is_particles=True, type=args.bilevel_type)
        if isinstance(losses, tuple):
            sm_loss = losses[0].mean()
            clf_loss = losses[1].mean()
            loss = sm_loss + args.reg_val * clf_loss
        else:
            loss = losses.mean()
        loss.backward()
        optimizer.step()
        ebm_ema.update_parameters(ebm)
        with torch.no_grad():
            ebm.f -= ebm.f[-1].clone()
        if args.loss_type == 'sm':
            loop.set_postfix(sm_loss=loss.item())
        else:
            loop.set_postfix(loss=loss.item(), sm_loss=sm_loss.item(), clf_loss=clf_loss.item())

# Disable gradient with respect to the parameters
for p in ebm.parameters():
    p.requires_grad_(False)

# Use both models
metrics, metrics_ema = {}, {}
for i, (model_, metrics_) in enumerate([(ebm, metrics), (ebm_ema.module, metrics_ema)]):
    samples = []
    n_total_samples = int(args.n_eval_samples / args.batch_size_eval)
    for i in range(n_total_samples):
        print('Evaluation loop {}/{}'.format(i+1, n_total_samples))
        x = sde.sample_base_dist((args.batch_size_eval,), data_shape=(22, 3))
        times_ones = torch.ones((args.batch_size_eval, 1, 1), device=device)
        for i in trange(eval_times.shape[0]-1, 0, -1):
            x = sde.second_order_heun_integration_step(x, eval_times[i-1] * times_ones,
                eval_times[i] * times_ones, model_.denoiser, apply_correction=i != 1)
            x = remove_mean(x)
        samples.append(x.clone().detach().cpu())
    samples = torch.stack(samples).view((-1, 22, 3))
    for k, v in target.compute_metrics(samples.to(device)).items():
        metrics_[k] = v
    metrics_['samples'] = target.compute_psi_phi(samples.to(device))
    os.makedirs(f"{args.results_path}/plots", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    target.plot_samples(ax[0], samples.clone().detach().cpu(), "model")
    target.plot_samples(ax[1], target.sample((samples.shape[0], )).clone().detach().cpu(), "ground truth")
    if i == 0:
        fig.savefig(f"{args.results_path}/plots/{filename[:-4]}.png")
    else:
        fig.savefig(f"{args.results_path}/plots/{filename[:-4]}_ema.png")
# Save the results
ebm = ebm.cpu()
ebm_ema = ebm_ema.cpu()
with open(args.results_path + '/' + filename, 'wb') as f:
    pickle.dump({'config': config, 'config_pkl' : ckpt_data['config'], 'metrics': metrics,
        'metrics_ema': metrics_ema, 'params' : ebm.state_dict(), 'params_ema' : ebm_ema.state_dict() }, f)
