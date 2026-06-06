import sys
sys.path.insert(0, '/repo')

import numpy as np
import torch
from pathlib import Path
from operator import itemgetter

import src.DCP_pinball_smooth as cp
import src.helpers as helpers

np.random.seed(0)
torch.manual_seed(20)  # Fixed for reproducibility

# Load data
path_to_experiments = Path('/repo/experiments/')
dataset = 'cifar100'
model = 'small_resnet14'

experiments = {
    'central': helpers.load_scores(*path_to_experiments.glob(f'{dataset}_central_{model}'), dataset=dataset),
}
experiments = dict(filter(itemgetter(1), experiments.items()))

# Non-IID partition
iid_flag = False
clients_class_map = helpers.get_client_map(dataset)
num_clients = len(clients_class_map)

# Q-DCP parameters (from notebook, topo comparison section)
num_trials = 10
gossip = 1500
epsilon_0 = 0.001
q0_list = None
R = 1

alphas = np.arange(0.10, 1, 0.10)
alphas = list(map(lambda x: np.round(x, 2), alphas))

# Star topology only
topo = 'star'
G, W, P = helpers.graph_construct(topo, num_clients)

eigenvalues = np.sort(np.abs(np.linalg.eigvals(W)))
rho = 1 - np.abs(eigenvalues[-2])
print(f"Star topology rho={rho:.4f}")

method = 'lac'
allow_empty_sets = True

f = itemgetter('val_scores', 'val_targets', 'test_scores', 'test_targets')

decentralized_trials = {}
for i in range(num_trials):
    trial = helpers.get_new_trial(experiments, frac=0.1)
    trial_experiments = trial['experiments']
    
    client_index_map = {
        k: sum(trial_experiments['central']['val_targets'] == k for k in v).bool() 
        for k, v in clients_class_map.items()
    }
    
    decentral_metrics = cp.get_decentralized_coverage_size_over_alphas(
        *f(trial_experiments['central']), method=method, 
        allow_empty_sets=allow_empty_sets, alphas=alphas, 
        decentral=True, gossip=gossip, client_index_map=client_index_map, 
        W=W, R=R, G=G, epsilon_0=epsilon_0, q0_list=q0_list,
        iid_flag=iid_flag
    )
    
    decentralized_trials[i] = decentral_metrics
    # Note: keys are 1-alpha (coverage target), so alpha=0.1 -> key=0.9
    print(f'Trial={i} done: coverage@alpha=0.10 -> key 0.90: {decentral_metrics["coverage"].get(0.90, "N/A"):.3f}, size={decentral_metrics["size"].get(0.90, "N/A"):.2f}')

star_results = helpers.combine_trials(decentralized_trials)

print("\n=== Q-DCP STAR TOPOLOGY RESULTS ===")
# alpha=0.1 means key=0.9 (1-alpha)
key = 0.90
coverage = star_results['mean']['coverage'][key]
size = star_results['mean']['size'][key]
coverage_std = star_results['std']['coverage'][key]
size_std = star_results['std']['size'][key]

print(f"Alpha=0.10 (key=0.90)")
print(f"Coverage (mean): {coverage:.4f}")
print(f"Coverage (std): {coverage_std:.4f}")
print(f"Set Size (mean): {size:.4f}")
print(f"Set Size (std): {size_std:.4f}")
print(f"\nAll results:")
for k in sorted(star_results['mean']['coverage'].keys()):
    print(f"  1-alpha={k:.2f} -> coverage={star_results['mean']['coverage'][k]:.4f}, size={star_results['mean']['size'][k]:.2f}")
