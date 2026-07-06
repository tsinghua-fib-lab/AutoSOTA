import sys
sys.path.insert(0, '/repo/reproducibility/icml2026')
sys.path.insert(0, '/repo/src')
from reproducibility.icml2026.dataset import Dataset
import pyciphod.causal_discovery.federated.regret_based.iperi.utils as utils

utils.set_determine(42)
dataset = Dataset(n_samples_client=1000, n_clients=4, n_variables=3, linear=True, horizontal_split=False, noise_distribution='normal', seed=42)
datasets, cpdag, ucpdag, graphs = dataset.generate(data_type='struct', save=False)
print('Type:', type(datasets[0]))
print('Shape:', datasets[0].shape if hasattr(datasets[0], 'shape') else 'N/A')
print('Is DataFrame:', hasattr(datasets[0], 'values'))
print('Dir:', [x for x in dir(datasets[0]) if not x.startswith('_')][:10])
