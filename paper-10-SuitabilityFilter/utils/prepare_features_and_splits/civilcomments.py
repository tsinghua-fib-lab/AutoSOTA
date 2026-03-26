import pickle
import random

import numpy as np
import torch

from suitability.datasets.wilds import get_wilds_dataset, get_wilds_model
from suitability.filter.suitability_efficient import get_sf_features

# Set seeds for reproducibility
random.seed(32)
np.random.seed(32)

# Configuration
data_name = "civilcomments"
root_dir = "/path/to/root/dir"  # Change this to your root directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
algorithm = "ERM"
model_type = "last"
seed = 0
model = get_wilds_model(
    data_name, root_dir, algorithm=algorithm, seed=seed, model_type=model_type
)
model = model.to(device)
model.eval()
print(f"Model loaded to device: {device}")

# Initialize results DataFrame
features_cache_file = (
    f"suitability/results/features/{data_name}_{algorithm}_{model_type}_{seed}.pkl"
)
valid_splits = ["val", "test"]
splits_features_cache = {}

# Precompute all data features
for split_name in valid_splits:
    print(f"Computing features for split: {split_name}")
    dataset = get_wilds_dataset(
        data_name,
        root_dir,
        split_name,
        batch_size=64,
        shuffle=False,
        num_workers=4,
    )
    splits_features_cache[split_name] = get_sf_features(dataset, model, device)
print("Features computed")

# Save feature cache
with open(features_cache_file, "wb") as f:
    pickle.dump(splits_features_cache, f)

# Precompute all id split indices
cache_file = f"suitability/results/split_indices/{data_name}.pkl"

valid_splits = [
    ("val", {"sensitive": "male"}),
    ("val", {"sensitive": "female"}),
    ("val", {"sensitive": "LGBTQ"}),
    ("val", {"sensitive": "christian"}),
    ("val", {"sensitive": "muslim"}),
    ("val", {"sensitive": "other_religions"}),
    ("val", {"sensitive": "black"}),
    ("val", {"sensitive": "white"}),
    ("test", {"sensitive": "male"}),
    ("test", {"sensitive": "female"}),
    ("test", {"sensitive": "LGBTQ"}),
    ("test", {"sensitive": "christian"}),
    ("test", {"sensitive": "muslim"}),
    ("test", {"sensitive": "other_religions"}),
    ("test", {"sensitive": "black"}),
    ("test", {"sensitive": "white"}),
]

splits_indices_cache = {}
for split_name, split_filter in valid_splits:
    print(f"Computing indices for split: {split_name} with filter: {split_filter}")
    dataset, indices = get_wilds_dataset(
        data_name,
        root_dir,
        split_name,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pre_filter=split_filter,
        return_indices=True,
    )
    splits_indices_cache[(split_name, str(split_filter))] = indices

with open(cache_file, "wb") as f:
    pickle.dump(splits_indices_cache, f)

print("Split indices computed")
