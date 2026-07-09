# Libraries
import argparse
import numpy as np
import os
import pickle
import pprint
import random
import torch
from diffclf.distr.aldp import AlanineDipeptide
from diffclf.em.multi_level import compute_loss_edm
from diffclf.networks.ebm import EDMEnergyPreconditioning
from diffclf.networks.ebm import DotEBM
from diffclf.networks.egnn import EGNN_atom
from diffclf.sde.diffusion import VE, EDM
from diffclf.sde.utils import TimeSampler
from diffclf.utils.se3_utils import remove_mean
from tqdm import tqdm

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--env', type=str)
parser.add_argument('--sde_type', type=str)
parser.add_argument('--vacuum_datapath', type=str, default='')
parser.add_argument('--n_levels', type=int, default=512)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--dataset_size', type=int, default=250000)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--hidden_nf', type=int, default=128)
parser.add_argument('--n_layers', type=int, default=5)
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

os.makedirs(args.results_path, exist_ok=True)

# Make a filename
filename = 'energy_clf_aldp_sm_only'
filename += '_sde_type_' + args.sde_type
filename += '_env_' + args.env
filename += '_seed_{}.pkl'.format(args.seed)

# Make the target distribution
target = AlanineDipeptide(args.data_path, env=args.env).to(device)
if args.env == 'vacuum':
    if len(args.vacuum_datapath) == 0:
        print('Provide the path to vaccum data.')
        exit(0)
    else:
        target.load_data(remove_mean(
            torch.load(args.vacuum_datapath).view((-1, *target.data_shape))
        ))

# Build an SDE
if args.sde_type == 've':
    sde = VE(sigma_min=1e-4)
    t_limits = (sde.sigma_inv(sde.sigma_min), sde.sigma_inv(sde.sigma_max))
else:
    sde = EDM()
    t_limits = (sde.sigma_inv(sde.sigma_min), sde.sigma_inv(sde.sigma_max))
sde = sde.to(device)

# Build the train times and normalizing constants
times = sde.get_snr_time_discretization(*t_limits, args.n_levels).to(device)
log_snr_times = sde.log_snr(times)
log_snr_dist = (log_snr_times.mean().item(), log_snr_times.std().item())
time_sampler = TimeSampler(times=times, log_snr_dist=log_snr_dist).to(device)

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
    hidden_nf=args.hidden_nf,
    time_embedding_dim=128,
    atom_type_embedding_dim=64,
    n_layers=args.n_layers,
    recurrent=False,
    attention=True,
    tanh=False,
    use_pos_embedding=True
)
base_ebm = DotEBM(base_net, sde=sde)
ebm = EDMEnergyPreconditioning(base_ebm, sde, data_mean, data_var_scalar,
    is_particles=True, log_snr_dist=None)
ebm = ebm.to(device)

# Create an EMA model
ebm_ema = torch.optim.swa_utils.AveragedModel(
    model=ebm,
    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
    use_buffers=True
)

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
        loss = compute_loss_edm(ebm, data[0], time_sampler, sde, data_var_scalar, is_particles=True)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        ebm_ema.update_parameters(ebm)
        loop.set_postfix(sm_loss=loss.item())

# Move everything to CPU
ebm = ebm.cpu()
ebm_ema = ebm_ema.cpu()

# Save the results
with open(args.results_path + '/' + filename, 'wb') as f:
    pickle.dump({'config': config, 'ebm': ebm.state_dict(), 'ebm_ema': ebm_ema.state_dict()}, f)
