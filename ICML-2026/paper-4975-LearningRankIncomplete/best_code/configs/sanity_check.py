from experiment.experiment import Experiment
from algos import PirateAlgorithm, PreferenceToPartialOrderAdapter, RankCentralityAlgorithm, PLPairwiseMLEAlgorithm, BordaCountAlgorithm
from feedback.preference import PreferenceFeedback
from loss.kendall_tau import kendall_tau_loss
import numpy as np
from typing import Iterator, List
from algos import Algorithm

import random
import numpy as np
import os

def get_experiment(seed = 42) -> Experiment:
    print("Setting up sanity check experiment...")

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Experiment parameters
    n = 20
    num_samples = 5000
    k = 10
    num_runs = 5
    
    true_ranking = np.random.permutation(n).tolist()
    
    # 1. Update the function to yield values infinitely
    def create_generator() -> Iterator[PreferenceFeedback]:
        while True:
            # Simple generator: sample two items and return the one that is higher
            # in the true ranking.
            i, j = np.random.choice(n, 2, replace=False)
            pos_i = true_ranking.index(i)
            pos_j = true_ranking.index(j)
            
            if pos_i < pos_j:
                yield PreferenceFeedback(prec_id=i, succ_id=j)
            else:
                yield PreferenceFeedback(prec_id=j, succ_id=i)

    # Baselines
    def baselines_factory() -> List[Algorithm]: return [
        PreferenceToPartialOrderAdapter(PirateAlgorithm(n=n)),
        RankCentralityAlgorithm(n=n),
        BordaCountAlgorithm(n=n),
        PLPairwiseMLEAlgorithm(n=n)
    ]

    # Loss function
    loss = kendall_tau_loss
    
    print("Experiment setup complete.")
    return Experiment(
        name="Sanity Check",
        num_samples=num_samples,
        true_ranking=true_ranking,
        generators=[create_generator() for _ in range(num_runs)], # 3. Pass the iterator instance
        baselines_factory=baselines_factory,
        loss=loss,
        k=k,
        seed=seed
    )
