import itertools
import json
import os

# Sweep space
dataset_names = ['cifar10-valid', 'cifar100', 'ImageNet16-120']
seeds = list(range(50))
acquisition_functions = [
    'PBGI(1e-4)',
    'PBGI(1e-5)',
    'PBGI(1e-6)',
    'LogEIPC',
    'LCB',
    'TS'
]

# Common settings
common_config = {
    "output_standardize": True,
    "maximize": False,
    "dim": 5,
    "num_iteration": 400,
    "num_configs": 32768
}

# Output directory
output_dir = "scripts/config/NATS_configs/sss_configs"
os.makedirs(output_dir, exist_ok=True)
print("Output directory exists:", os.path.exists(output_dir))

# Generate config files
counter = 0
for dataset, seed, acq in itertools.product(dataset_names, seeds, acquisition_functions):
    config = dict(common_config)
    config["dataset_name"] = dataset
    config["seed"] = seed
    config["acquisition_function"] = acq

    file_path = os.path.join(output_dir, f"config_{counter}.json")
    with open(file_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote: {file_path}")  # Print each file created
    counter += 1

print(f"Total configs generated: {counter}")

