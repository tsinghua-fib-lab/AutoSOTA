"""FedQueue simulation for paper reproduction."""

import argparse
import random
import time
import json
import os
import numpy as np
from omegaconf import OmegaConf
from appfl.agent import SimClientAgent, SimServerAgent
from appfl.loader import load_and_split_dataset

argparser = argparse.ArgumentParser()
argparser.add_argument("--config", type=str, default="./sim_mnist_fedqueue.yaml")
argparser.add_argument("--output", type=str, default="./output/fedqueue_results.json")
args = argparser.parse_args()

config = OmegaConf.load(args.config)
scheduler_config = config.algorithm_configs.scheduler_kwargs
num_clients = int(scheduler_config.num_clients)
data_seed = int(config.data_configs.partition_kwargs.get("data_seed", 42))
rng = random.Random(data_seed)

print(f"Loading MNIST dataset (Dirichlet alpha=0.5, seed={data_seed}, {num_clients} clients)...")
client_datasets, server_dataset, dataset_meta = load_and_split_dataset(
    config.data_configs, num_clients
)
total_client_train = sum(len(ds[0]) for ds in client_datasets)
print(f"Dataset loaded. Server test: {len(server_dataset)}, Client train total: {total_client_train}")

sample_sizes = {
    str(client_id): len(data[0] if isinstance(data, (tuple, list)) else data)
    for client_id, data in enumerate(client_datasets)
}
for cid, sz in sample_sizes.items():
    print(f"  Client {cid}: {sz} train samples")

# Create server agent
server_agent = SimServerAgent(server_agent_config=config)
server_agent.set_sample_size(sample_sizes=sample_sizes)
server_agent.load_server_val_dataset(server_dataset)

# Create client agents
client_agents = [
    SimClientAgent(
        client_agent_config=config,
        client_id=client_id,
        client_dataset=client_datasets[client_id],
    )
    for client_id in range(len(client_datasets))
]

# Load initial global model from server
result = server_agent.get_parameters(serial_run=True)
if isinstance(result, tuple):
    global_model = result[0]
else:
    global_model = result

for client in client_agents:
    client.load_parameters(global_model)

target_accuracy = 95.0
max_accuracy = 0.0
time_to_target = None
total_local_steps = 0
start_wall_time = time.time()

num_rounds = config.algorithm_configs.num_global_epochs
print(f"\nRunning FedQueue for {num_rounds} rounds...\n")

client_metadata = {cid: {} for cid in range(num_clients)}
metrics_history = []

for r in range(num_rounds):
    round_start = time.time()
    futures = []

    for client_id in range(num_clients):
        client = client_agents[client_id]
        meta = client_metadata.get(client_id, {})

        train_kwargs = {"round": r}
        if meta:
            train_kwargs.update(meta)

        client.train(**train_kwargs)

        local_model = client.get_parameters()
        if isinstance(local_model, tuple):
            local_model, train_meta = local_model[0], local_model[1]
        else:
            train_meta = {}

        # For round 0 (first submission), provide sensible defaults
        if r == 0 and not meta:
            train_meta["queue_time"] = 0.0
            train_meta["compute_second_per_step"] = 0.01

        if "current_local_steps" in train_meta:
            total_local_steps += train_meta["current_local_steps"]

        future = server_agent.global_update(
            client_id=client.get_id(),
            local_model=local_model,
            blocking=False,
            **train_meta,
        )
        futures.append((client_id, future))

    # Collect results
    for client_id, future in futures:
        result = future.result() if hasattr(future, "result") else future
        if isinstance(result, tuple):
            global_model, meta = result
            client_metadata[client_id] = meta
        else:
            global_model = result

    # Load global model
    for client in client_agents:
        client.load_parameters(global_model)

    # Evaluate
    result = server_agent.server_validate()
    if result is not None:
        test_loss, test_acc = result
        elapsed = time.time() - start_wall_time

        if test_acc > max_accuracy:
            max_accuracy = test_acc

        if time_to_target is None and test_acc >= target_accuracy:
            time_to_target = elapsed

        metrics_history.append({
            "round": r + 1,
            "elapsed_s": round(elapsed, 3),
            "test_loss": round(test_loss, 4),
            "test_accuracy": round(test_acc, 2),
        })

        print(f"Round {r+1:3d}/{num_rounds} | Acc: {test_acc:.2f}% | "
              f"Max: {max_accuracy:.2f}% | T: {elapsed:.1f}s"
              + (f" | TTA*={time_to_target:.1f}s" if time_to_target else ""))

# Final evaluation
result = server_agent.server_validate()
if result is not None:
    test_loss, test_acc = result
    if test_acc > max_accuracy:
        max_accuracy = test_acc

total_wall_time = time.time() - start_wall_time
if time_to_target is None and max_accuracy >= target_accuracy:
    time_to_target = total_wall_time

sep = "=" * 60
print(f"\n{sep}")
print(f"FedQueue MNIST Results")
print(f"{sep}")
print(f"Max Accuracy (Max-A):       {max_accuracy:.2f}%")
if time_to_target:
    print(f"Time-to-A* (TTA*):          {time_to_target:.1f}s")
else:
    print(f"Time-to-A* (TTA*):          NOT REACHED (max acc: {max_accuracy:.2f}%)")
print(f"Total Local Steps (#Ek):    {total_local_steps}")
print(f"Total Wall Time:            {total_wall_time:.1f}s")
print(f"Num Rounds Completed:       {num_rounds}")

os.makedirs(os.path.dirname(args.output), exist_ok=True)
results = {
    "paper": "FedQueue",
    "max_accuracy": round(max_accuracy, 2),
    "time_to_target": round(time_to_target, 1) if time_to_target else None,
    "total_local_steps": total_local_steps,
    "total_wall_time": round(total_wall_time, 1),
    "num_rounds": num_rounds,
    "target_accuracy": target_accuracy,
    "metrics_history": metrics_history,
}
with open(args.output, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {args.output}")
