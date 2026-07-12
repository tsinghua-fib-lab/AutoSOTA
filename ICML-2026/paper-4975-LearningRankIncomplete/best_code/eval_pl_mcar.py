#!/usr/bin/env python3
"""Evaluation script for Plackett-Luce MCAR experiment (paper 4975).

Reproduces the experiment from Figure 2a:
- Plackett-Luce ranking model with n=10 items
- MCAR (random pairwise) feedback
- 500,000 samples, 5 runs
- Reports average normalized Kendall metric (lower is better)
"""
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
from typing import List

def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    n = 10
    num_samples = 500000
    k = 10
    num_runs = 5

    pl_parameters = np.random.uniform(0, 1, size=n)
    true_ranking = np.argsort(pl_parameters).tolist()[::-1]
    pl = PlackettLuceRankingModel(pl_params=pl_parameters)

    def baselines_factory() -> List:
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
            n=n, ranking_model=pl,
            pos_filter=RandomPairPositionalFilter(n=n)
        ) for _ in range(num_runs)],
        baselines_factory=baselines_factory,
        loss=kendall_tau_loss,
        k=k, seed=seed, num_recordings=200, initial_samples=5
    )

    exp.run()

    # Report final metrics
    results = {}
    for name, runs_data in exp.results.items():
        data = np.array(runs_data)
        final_values = data[:, -1]
        mean_val = float(np.mean(final_values))
        std_val = float(np.std(final_values))
        results[name] = {'mean': mean_val, 'std': std_val}
        print(f'{name}: mean={mean_val:.6f} std={std_val:.6f}')

    # Primary metric: PIRATE average normalized kendall metric
    primary = results['PIRATE']['mean']
    print(f'\nPrimary metric (PIRATE avg normalized kendall): {primary:.6f}')

    # Save results
    os.makedirs('output', exist_ok=True)
    with open('output/eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return primary

if __name__ == '__main__':
    main()
