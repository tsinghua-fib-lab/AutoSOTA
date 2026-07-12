import numpy as np
from typing import List, Tuple, Any
from experiment.experiment import Experiment
from algos import PirateAlgorithm
from algos.rank_centrality import RankCentralityAlgorithm
from algos.borda_count import BordaCountAlgorithm
from algos.pl_pairwise_mle import PLPairwiseMLEAlgorithm
from loss.kendall_tau import kendall_tau_loss
from algos import RankBreakableToPreferenceAdapter
from pos_filter import DeterministicPositionalFilter, RandomPairPositionalFilter
from data_generator import SyntheticPositionalFilteredDataGenerator, SyntheticTopHDataGenerator, SyntheticBottomHDataGenerator
from ranking_models import PlackettLuceRankingModel
import random
from algos import Algorithm

import random
import numpy as np
import os

def get_experiment(seed = 42) -> Experiment:
    print("Setting up Plackett-Luce MCAR experiment...")

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    n: int = 10
    num_samples: int = 500000  
    k: int = 10
    num_runs = 5
    
    pl_parameters = np.random.uniform(0, 1, size=n)
    true_ranking: List[int] = np.argsort(pl_parameters).tolist()[::-1]

    pl = PlackettLuceRankingModel(pl_params=pl_parameters)

    def baselines_factory() -> List[Algorithm]: return [
        PirateAlgorithm(n=n),
        RankBreakableToPreferenceAdapter(RankCentralityAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(BordaCountAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(PLPairwiseMLEAlgorithm(n=n)),
    ]
    
    print("Experiment setup complete.")
    return Experiment(
        name="Plackett-Luce MCAR",
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
        seed=seed
    )
