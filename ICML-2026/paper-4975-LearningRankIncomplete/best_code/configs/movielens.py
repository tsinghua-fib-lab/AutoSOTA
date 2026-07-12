import os
import urllib.request
import zipfile
from typing import Iterator, List, Any, Dict

import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from experiment.experiment import Experiment
from algos import PirateAlgorithm, RankCentralityAlgorithm, BordaCountAlgorithm, PLPairwiseMLEAlgorithm, Algorithm
from algos.adapters import PartialOrderToPreferenceAdapter
from feedback.preference import PreferenceFeedback
from feedback.partial_order import PartialOrderFeedback
from loss.kendall_tau import kendall_tau_loss

import random
import numpy as np
import os

def _ensure_movielens_100k(data_dir: str = "./data") -> Any:
    """Downloads and extracts MovieLens 100K."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path: str = os.path.join(data_dir, "ml-100k.zip")
    extract_dir: str = os.path.join(data_dir, "ml-100k")
    csv_path: str = os.path.join(extract_dir, "u.data")

    if not os.path.exists(zip_path) and not os.path.exists(csv_path):
        print("Downloading MovieLens 100K dataset...")
        url: str = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        urllib.request.urlretrieve(url, zip_path)

    if not os.path.exists(csv_path):
        print("Extracting 100K dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    print("Loading 100K ratings...")
    names: List[str] = ['userId', 'movieId', 'rating', 'timestamp']
    df: Any = pd.read_csv(csv_path, sep='\t', names=names)
    return df


def _ensure_movielens_25m(data_dir: str = "./data") -> Any:
    """Downloads and extracts MovieLens 25M."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path: str = os.path.join(data_dir, "ml-25m.zip")
    extract_dir: str = os.path.join(data_dir, "ml-25m")
    csv_path: str = os.path.join(extract_dir, "ratings.csv")

    if not os.path.exists(zip_path) and not os.path.exists(csv_path):
        print("Downloading MovieLens 25M dataset (~250MB)...")
        url: str = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
        urllib.request.urlretrieve(url, zip_path)

    if not os.path.exists(csv_path):
        print("Extracting 25M dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    print("Loading 25M ratings into memory...")
    df: Any = pd.read_csv(csv_path)
    return df


def get_experiment(dataset_name: str = "25m", seed = 42) -> Experiment:
    """Sets up the ranking experiment using exact sample sizes."""
    print("Setting up MovieLens experiment...")

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Configuration Variables
    n: int = 25
    k: int = 25
    num_runs: int = 5
    
    # 1. Load the requested dataset
    df: Any
    if dataset_name.lower() == "100k":
        df = _ensure_movielens_100k()
    elif dataset_name.lower() == "25m":
        df = _ensure_movielens_25m()
    else:
        raise ValueError("dataset_name must be '100k' or '25m'")
    
    # 2. Pick top-n movies with the highest number of ratings
    print(f"Filtering for top {n} most-rated movies...")
    top_movies: Any = df['movieId'].value_counts().nlargest(n).index
    df_top: Any = df[df['movieId'].isin(top_movies)].copy()
    
    # 3. Map actual movieIds to contiguous IDs 0...n-1 for the algorithms
    movie_id_map: Dict[int, int] = {old_id: new_id for new_id, old_id in enumerate(top_movies)}
    df_top['mapped_id'] = df_top['movieId'].map(movie_id_map)
    
    # 4. Filter users who have rated at least 2 of these selected movies
    user_counts: Any = df_top['userId'].value_counts()
    valid_users: Any = user_counts[user_counts >= 2].index
    df_filtered: Any = df_top[df_top['userId'].isin(valid_users)]
    
    # 5. Split users into num_runs + 1 bins
    users: Any = np.random.permutation(valid_users.to_numpy())
    user_bins: Any = np.array_split(users, num_runs + 1)
    
    train_bins: Any = user_bins[:num_runs]
    truth_bin: Any = user_bins[-1]
    
    # 6. Compute proxy for true ranking using the truth_bin
    df_truth: Any = df_filtered[df_filtered['userId'].isin(truth_bin)]
    avg_ratings: Any = df_truth.groupby('mapped_id')['rating'].mean()
    
    global_mean: float = float(df_truth['rating'].mean())
    avg_ratings = avg_ratings.reindex(range(n), fill_value=global_mean)
    true_ranking: List[int] = avg_ratings.sort_values(ascending=False).index.tolist()
    
    # 7. Pre-compute partial orders to find the exact number of valid samples
    print("Pre-computing preference graphs to determine exact sample sizes...")
    generators: List[Iterator[PartialOrderFeedback]] = []
    min_samples: float = float('inf')
    
    for t_bin in train_bins:
        df_bin: Any = df_filtered[df_filtered['userId'].isin(t_bin)]
        grouped: Any = df_bin.groupby('userId')
        
        bin_feedback: List[PartialOrderFeedback] = []
        
        for uid, user_data in grouped:
            items: Any = user_data['mapped_id'].values
            ratings: Any = user_data['rating'].values
            
            preferences: List[PreferenceFeedback] = []
            
            # Compare every pair of items rated by this user
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    if ratings[i] > ratings[j]:
                        preferences.append(PreferenceFeedback(prec_id=int(items[i]), succ_id=int(items[j])))
                    elif ratings[j] > ratings[i]:
                        preferences.append(PreferenceFeedback(prec_id=int(items[j]), succ_id=int(items[i])))
            
            # Only count users that produced at least one strict preference
            if preferences:
                bin_feedback.append(PartialOrderFeedback(preferences=preferences))
                
        # Shuffle the finite list to randomize the order the algorithm sees them
        np.random.shuffle(bin_feedback)
        
        # Track the minimum size to safely feed the Experiment framework
        if len(bin_feedback) < min_samples:
            min_samples = len(bin_feedback)
            
        # Create an iterator from the finite list
        generators.append(iter(bin_feedback))

    # The exact number of samples we can safely pull from all bins
    exact_num_samples: int = int(min_samples)
    print(f"Exact number of valid samples per run determined to be: {exact_num_samples}")

    # 8. Baselines
    def baselines_factory() -> List[Algorithm]: return [
        PirateAlgorithm(n=n),
        PartialOrderToPreferenceAdapter(RankCentralityAlgorithm(n=n)),
        PartialOrderToPreferenceAdapter(BordaCountAlgorithm(n=n)),
        PartialOrderToPreferenceAdapter(PLPairwiseMLEAlgorithm(n=n)),
    ]

    print("Experiment setup complete.")
    return Experiment(
        name=f"MovieLens {dataset_name.upper()} Top-{n}",
        num_samples=exact_num_samples,
        true_ranking=true_ranking,
        generators=generators,
        baselines_factory=baselines_factory,
        loss=kendall_tau_loss,
        k=k,
        seed=seed
    )