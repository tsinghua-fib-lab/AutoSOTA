"""
get_modals.py

Get the modal values of the hyperparameters for each model.
"""
import os
import json
import numpy as np
from collections import Counter
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "experiments"


xgb_counter = Counter()
mlp_counter = Counter()
crda_counter = Counter()

def add_params(path, counter):
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r') as f:
            data = json.load(f)
            if "hidden_layer_sizes" in data['model_params']:
                data['model_params']['hidden_layer_sizes'] = str(data['model_params']['hidden_layer_sizes'])

            params_tuple = tuple(sorted(data['model_params'].items()))
            for param in params_tuple:
                counter[param] += 1

            crda_params = tuple(sorted(data['best_aug_params'].items()))
            for param in crda_params:
                crda_counter[param] += 1

for dir in os.listdir(path):
    if os.path.isdir(os.path.join(path, dir)):
        for subdir in os.listdir(os.path.join(path, dir)):
            if os.path.isdir(os.path.join(path, dir, subdir)):
                if 'xgboost' in subdir:
                    add_params(os.path.join(path, dir, subdir, 'params'), xgb_counter)
                elif 'mlp' in subdir:
                    add_params(os.path.join(path, dir, subdir, 'params'), mlp_counter)


max_key = None
max_value = 0
for key, value in xgb_counter.items():
    if "max_depth" in key:
        if value > max_value:
            max_value = value
            max_key = key
if max_key:
    print(max_key, max_value)

# max_key = None
# max_value = 0
# for key, value in mlp_counter.items():
#     if "hidden_layer_sizes" in key:
#         if value > max_value:
#             max_value = value
#             max_key = key
# if max_key:
#     print(max_key, max_value)


# max_key = None
# max_value = 0
# for key, value in crda_counter.items():
#     if "max_perturb_percent" in key:
#         if value > max_value:
#             max_value = value
#             max_key = key
# if max_key:
#     print(max_key, max_value)



