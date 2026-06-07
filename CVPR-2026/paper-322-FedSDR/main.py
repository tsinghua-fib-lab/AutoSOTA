import fgl.config as config
from fgl.flcore.trainer import FGLTrainer
from function import *
from config_datasets import attribute
from fgl.utils.basic_utils import seed_everything
from fgl.data.distributed_dataset_loader import FGLDataset
import os

args = config.args

# Don't hardcode — use CLI args
if args.dataset == []:
    args.dataset = ["PubMed"]

if args.root == "change_to_your_root_path":
    args.root = os.environ.get("DATA_ROOT", "/repo/data")

# Get dataset-specific defaults (CLI overrides attribute values)
num_clients_d, classes_d, num_rounds_d, lr_d = attribute(args.dataset)

# CLI values take precedence; attribute() values are defaults for unspecified params
# Note: argparse defaults: num_clients=10, num_rounds=100, lr=0.01
# For PubMed, attribute() returns 10, 3, 1000, 0.01
# We respect CLI --num_clients, --num_rounds, --lr if explicitly passed
args.classes = classes_d

seed_everything(args.seed)

# Step 1: Create FGLDataset — generates clean Louvain partitions
print("Creating FGLDataset (Louvain partitioning)...")
fgl_dataset = FGLDataset(args)
clean_processed_dir = fgl_dataset.processed_dir
print(f"Clean data at: {clean_processed_dir}")

# Step 2: Apply structural noise to corrupted clients
print("Applying structural noise...")
client_data_list = load_client_list(client_data_dir=clean_processed_dir)
num_clients = len(client_data_list)
num_corrupted_clients = int(num_clients * args.corruption_ratio)
print(f"Corrupting {num_corrupted_clients}/{num_clients} clients")

import random
corrupted_client_indices = random.sample(range(num_clients), num_corrupted_clients)
print(f"Corrupted client indices: {corrupted_client_indices}")

for client_id in corrupted_client_indices:
    splitted_data = client_data_list[client_id]
    processed_data = random_topology_noise(splitted_data, noise_prob=args.noise_extent)
    client_data_list[client_id] = processed_data

# Save corrupted data back to processed_dir
for client_id, client_data in enumerate(client_data_list):
    torch.save(client_data, os.path.join(clean_processed_dir, f"data_{client_id}.pt"))

print("Noise applied and data saved.")

# Step 3: Train with FedSDR (FGLTrainer creates new FGLDataset loading corrupted data)
print("Starting FedSDR training...")
trainer = FGLTrainer(args)
trainer.train()
