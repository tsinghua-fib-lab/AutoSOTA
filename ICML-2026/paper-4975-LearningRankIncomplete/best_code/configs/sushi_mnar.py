import os
import urllib.request
import zipfile
from typing import Iterator, List
from tqdm import tqdm
import numpy as np

from experiment.experiment import Experiment
from algos import PirateAlgorithm, RankCentralityAlgorithm, BordaCountAlgorithm, PLPairwiseMLEAlgorithm, Algorithm
from algos.adapters import RankBreakableToPreferenceAdapter
from feedback.partial_order import PartialOrderFeedback
from loss.kendall_tau import kendall_tau_loss

from pos_filter import DeterministicPositionalFilter

import random
import numpy as np
import os

def _ensure_sushi_full(data_dir: str = "./data") -> List[List[int]]:
    """Downloads the SUSHI Set A directly from the verified URL."""
    os.makedirs(data_dir, exist_ok=True)
    url = "https://www.kamishima.net/asset/sushi3-2016.zip"
    zip_path = os.path.join(data_dir, "sushi3-2016.zip")
    order_file = "sushi3a.5000.10.order"
    order_path = os.path.join(data_dir, order_file)

    if not os.path.exists(order_path):
        print("Downloading SUSHI archive from Kamishima Lab...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(zip_path, 'wb') as f:
            f.write(resp.read())
        
        print("Extracting full rankings...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extract(f"sushi3-2016/{order_file}", data_dir)
            os.rename(os.path.join(data_dir, f"sushi3-2016/{order_file}"), order_path)

    rankings = []
    with open(order_path, 'r') as f:
        for line in f.readlines()[1:]: 
            rankings.append([int(x) for x in line.strip().split()[2:]])
    return rankings

def get_experiment(seed = 42) -> Experiment:
    print("Setting up Sushi MNAR experiment...")
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    n: int = 10
    k: int = 10
    num_runs: int = 5
    
    full_rankings = _ensure_sushi_full()
    
    indices = np.random.permutation(len(full_rankings))
    user_bins = np.array_split(indices, num_runs + 1)
    train_bins, truth_bin = user_bins[:num_runs], user_bins[-1]
    
    # 1. CLEAN GROUND TRUTH from holdout bin
    scores = np.zeros(n)
    for idx in truth_bin:
        for pos, item in enumerate(full_rankings[idx]):
            scores[item] += (n - 1 - pos)
    
    true_ranking = np.argsort(-scores).tolist()

    generators: List[Iterator[PartialOrderFeedback]] = []
    min_samples = float('inf')

    pos_filter = DeterministicPositionalFilter(
        n=n,
        positions=[
            (1, 2),
            (2, 3),
            (n-3, n-2),
            (n-2, n-1)
        ]
    )
    
    for t_idx, t_bin in enumerate(train_bins):
        bin_feedback: List[PartialOrderFeedback] = []
        for idx in tqdm(t_bin, desc=f"Filtering Run {t_idx+1}/{num_runs}"):
            user_rank = full_rankings[idx]
            
            feedback = pos_filter.filter(user_rank)
            
            if feedback.preferences:
                bin_feedback.append(feedback)
                
        np.random.shuffle(bin_feedback)
        min_samples = min(min_samples, len(bin_feedback))
        generators.append(iter(bin_feedback))

    exact_num_samples = int(min_samples)
    print(f"Final valid samples per run: {exact_num_samples}")

    def baselines_factory() -> List[Algorithm]: return [
        PirateAlgorithm(n=n),
        RankBreakableToPreferenceAdapter(RankCentralityAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(BordaCountAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(PLPairwiseMLEAlgorithm(n=n)),
    ]

    print("Experiment setup complete.")
    return Experiment(
        name="Sushi MNAR",
        num_samples=exact_num_samples,
        true_ranking=true_ranking,
        generators=generators,
        baselines_factory=baselines_factory,
        loss=kendall_tau_loss,
        k=k,
        seed=seed
    )
