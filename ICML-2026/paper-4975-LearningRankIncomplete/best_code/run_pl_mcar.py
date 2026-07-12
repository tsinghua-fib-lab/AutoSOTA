import numpy as np
import random
import os
import sys
import json
from experiment.experiment import Experiment
from algos import PirateAlgorithm
from algos.rank_centrality import RankCentralityAlgorithm
from algos.borda_count import BordaCountAlgorithm
from algos.pl_pairwise_mle import PLPairwiseMLEAlgorithm
from loss.kendall_tau import kendall_tau_loss
from algos import RankBreakableToPreferenceAdapter
from pos_filter import RandomPairPositionalFilter
from data_generator import SyntheticPositionalFilteredDataGenerator
from ranking_models import PlackettLuceRankingModel
from algos import Algorithm
from typing import List

seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

n = 10
num_samples = 500000
k = 10
num_runs = 5
num_recordings = 200

print(f'Plackett-Luce MCAR Experiment: n={n}, samples={num_samples}, k={k}, runs={num_runs}')
print(f'Using seed={seed}')

# Parameters sampled uniformly in (0,1)
pl_parameters = np.random.uniform(0, 1, size=n)
true_ranking = np.argsort(pl_parameters).tolist()[::-1]
print(f'PL parameters: {pl_parameters}')
print(f'True ranking (by PL params): {true_ranking}')

pl = PlackettLuceRankingModel(pl_params=pl_parameters)

def baselines_factory() -> List[Algorithm]:
    return [
        PirateAlgorithm(n=n),
        RankBreakableToPreferenceAdapter(RankCentralityAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(BordaCountAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(PLPairwiseMLEAlgorithm(n=n)),
    ]

exp = Experiment(
    name='Plackett-Luce MCAR',
    num_samples=num_samples,
    true_ranking=true_ranking,
    generators=[SyntheticPositionalFilteredDataGenerator(
        n=n,
        ranking_model=pl,
        pos_filter=RandomPairPositionalFilter(n=n)
    ) for _ in range(num_runs)],
    baselines_factory=baselines_factory,
    loss=kendall_tau_loss,
    k=k,
    seed=seed,
    num_recordings=num_recordings,
    initial_samples=5
)

print('Starting experiment run...')
sys.stdout.flush()
exp.run()
print('Experiment run complete.')

# Print final results
print()
print('='*60)
print('FINAL RESULTS (last recording point, ~500000 samples)')
print('='*60)
results_summary = {}
for name, runs_data in exp.results.items():
    data = np.array(runs_data)
    final_values = data[:, -1]  # last recording point for each run
    mean_val = float(np.mean(final_values))
    std_val = float(np.std(final_values))
    ci = 1.96 * std_val / np.sqrt(len(final_values))
    print(f'{name}:')
    print(f'  Mean: {mean_val:.6f}')
    print(f'  Std:  {std_val:.6f}')
    print(f'  95% CI: [{mean_val - ci:.6f}, {mean_val + ci:.6f}]')
    print(f'  Per-run: {final_values.tolist()}')
    results_summary[name] = {
        'mean': mean_val,
        'std': std_val,
        'ci_lower': mean_val - ci,
        'ci_upper': mean_val + ci,
        'per_run': final_values.tolist()
    }

# Save results
import os
os.makedirs('/repo/output', exist_ok=True)
with open('/repo/output/pl_mcar_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print()
print('Results saved to /repo/output/pl_mcar_results.json')
