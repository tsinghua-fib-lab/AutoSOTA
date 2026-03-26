import pickle
import random

import numpy as np
import torch

from suitability.datasets.wilds import get_wilds_dataset, get_wilds_model
from suitability.filter.suitability_filter import get_sf_features

# Set seeds for reproducibility
random.seed(32)
np.random.seed(32)

# Configuration
data_name = "fmow"
root_dir = "/path/to/root/dir"  # Change this to your root directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
algorithm = "ERM"
model_type = "last" # Change this to "last" or "best" as needed
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
valid_splits = ["id_val", "id_test", "val", "test"]
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
print("ID splits features computed")

# Save feature cache
with open(features_cache_file, "wb") as f:
    pickle.dump(splits_features_cache, f)

# Precompute all id split indices
id_cache_file = f"suitability/results/split_indices/{data_name}_id.pkl"

valid_id_splits = [
    ("id_val", {"year": [2002, 2003, 2004, 2005, 2006]}),
    ("id_val", {"year": [2007, 2008, 2009]}),
    ("id_val", {"year": [2010]}),
    ("id_val", {"year": [2011]}),
    ("id_val", {"year": [2012]}),
    ("id_val", {"region": ["Asia"]}),
    ("id_val", {"region": ["Europe"]}),
    ("id_val", {"region": ["Americas"]}),
    ("id_test", {"year": [2002, 2003, 2004, 2005, 2006]}),
    ("id_test", {"year": [2007, 2008, 2009]}),
    ("id_test", {"year": [2010]}),
    ("id_test", {"year": [2011]}),
    ("id_test", {"year": [2012]}),
    ("id_test", {"region": ["Asia"]}),
    ("id_test", {"region": ["Europe"]}),
    ("id_test", {"region": ["Americas"]}),
]

id_splits_indices_cache = {}
for split_name, split_filter in valid_id_splits:
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
    id_splits_indices_cache[(split_name, str(split_filter))] = indices

with open(id_cache_file, "wb") as f:
    pickle.dump(id_splits_indices_cache, f)

# Precompute all ood split indices
ood_cache_file = f"suitability/results/split_indices/{data_name}_ood.pkl"

valid_ood_splits = [
    ("val", {"year": [2013]}),
    ("val", {"year": [2014]}),
    ("val", {"year": [2015]}),
    ("val", {"region": ["Asia"]}),
    ("val", {"region": ["Europe"]}),
    ("val", {"region": ["Africa"]}),
    ("val", {"region": ["Americas"]}),
    ("val", {"region": ["Oceania"]}),
    ("val", {"region": "Europe", "year": 2013}),
    ("val", {"region": "Europe", "year": 2014}),
    ("val", {"region": "Europe", "year": 2015}),
    ("val", {"region": "Asia", "year": 2013}),
    ("val", {"region": "Asia", "year": 2014}),
    ("val", {"region": "Asia", "year": 2015}),
    ("val", {"region": "Americas", "year": 2013}),
    ("val", {"region": "Americas", "year": 2014}),
    ("val", {"region": "Americas", "year": 2015}),
    ("test", {"year": 2016}),
    ("test", {"year": 2017}),
    ("test", {"region": "Asia"}),
    ("test", {"region": "Europe"}),
    ("test", {"region": "Africa"}),
    ("test", {"region": "Americas"}),
    ("test", {"region": "Oceania"}),
    ("test", {"region": "Europe", "year": 2016}),
    ("test", {"region": "Europe", "year": 2017}),
    ("test", {"region": "Asia", "year": 2016}),
    ("test", {"region": "Asia", "year": 2017}),
    ("test", {"region": "Americas", "year": 2016}),
    ("test", {"region": "Americas", "year": 2017}),
]

ood_splits_indices_cache = {}

for split_name, split_filter in valid_ood_splits:
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
    ood_splits_indices_cache[(split_name, str(split_filter))] = indices

# Save cache
with open(ood_cache_file, "wb") as f:
    pickle.dump(ood_cache_file, f)
print("Features saved")
