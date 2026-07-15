"""FedQueue simulation v2 - efficient controlled queue simulation."""

import argparse
import random
import time
import json
import os
import math
import numpy as np
import torch
from omegaconf import OmegaConf
from appfl.agent import SimClientAgent, SimServerAgent
from appfl.loader import load_and_split_dataset
from appfl.algorithm.aggregator.fedqueue_aggregator import FedQueueAggregator
from appfl.logger import ServerAgentFileLogger

argparser = argparse.ArgumentParser()
argparser.add_argument("--config", type=str, default="./sim_mnist_fedqueue_v2.yaml")
argparser.add_argument("--output", type=str, default="./output/fedqueue_v2_results.json")
args = argparser.parse_args()

config = OmegaConf.load(args.config)
sched_cfg = config.algorithm_configs.scheduler_kwargs
num_clients = int(sched_cfg.num_clients)
data_seed = int(config.data_configs.partition_kwargs.get("data_seed", 42))
num_rounds = config.algorithm_configs.num_global_epochs

# Paper parameters
Tsync = float(sched_cfg.get("t_sync", 10.0))
delta = float(sched_cfg.get("safety_buffer", 2.0))
ewma_alpha = float(sched_cfg.get("alpha_queue", 0.5))
warmup_steps = int(sched_cfg.get("warm_up_steps", 10))
base_lr = float(sched_cfg.get("lr_base", 0.003))
queue_mode = str(sched_cfg.get("queue_mode", "lognormal"))
queue_rho = float(str(sched_cfg.get("queue_means", "0.9")).split(",")[0].strip())
queue_sigma = float(sched_cfg.get("queue_sigma", 0.4))
queue_seed = int(sched_cfg.get("queue_seed", 42))
staleness_fn_name = config.algorithm_configs.aggregator_kwargs.get("staleness_fn", "harmonic")
staleness_beta = config.algorithm_configs.aggregator_kwargs.get("staleness_fn_kwargs", {}).get("beta", 0.5)
alpha_admission = float(sched_cfg.get("alpha_admission", 0.2))

print(f"FedQueue Simulation v2")
print(f"  Tsync={Tsync}, delta={delta}, ewma_alpha={ewma_alpha}")
print(f"  queue_mode={queue_mode}, rho={queue_rho}, sigma={queue_sigma}")
print(f"  staleness={staleness_fn_name}, beta={staleness_beta}")

# Load data
print(f"\nLoading MNIST (Dirichlet alpha=0.5, seed={data_seed})...")
client_datasets, server_dataset, dataset_meta = load_and_split_dataset(
    config.data_configs, num_clients
)
sample_sizes = {
    str(cid): len(ds[0]) for cid, ds in enumerate(client_datasets)
}
print(f"Loaded: {len(server_dataset)} test, client sizes: {sample_sizes}")

# Create server
server_agent = SimServerAgent(server_agent_config=config)
server_agent.set_sample_size(sample_sizes=sample_sizes)
server_agent.load_server_val_dataset(server_dataset)

# Create clients
client_agents = [
    SimClientAgent(client_agent_config=config, client_id=cid, client_dataset=client_datasets[cid])
    for cid in range(num_clients)
]

# Initial model
result = server_agent.get_parameters(serial_run=True)
global_model = result[0] if isinstance(result, tuple) else result
for c in client_agents:
    c.load_parameters(global_model)

# --- FedQueue Simulation Logic ---
# Queue delay RNG
queue_rng = random.Random(queue_seed)

# State
q_hat = {cid: queue_rho for cid in range(num_clients)}  # queue estimate
c_hat = {cid: 0.01 for cid in range(num_clients)}        # compute rate estimate
client_deadline = {cid: queue_rho + alpha_admission * Tsync for cid in range(num_clients)}
late_buffer = []  # buffered late updates
global_round_counter = 0

# Staleness function
if staleness_fn_name == "harmonic":
    def staleness_fn(u):
        return 1.0 / (1.0 + staleness_beta * u)
elif staleness_fn_name == "exponential":
    def staleness_fn(u):
        return math.exp(-staleness_beta * u)
else:
    def staleness_fn(u):
        return 1.0

def sample_queue_delay(client_id):
    """Sample lognormal queue delay."""
    mu = math.log(max(1e-6, queue_rho)) - 0.5 * queue_sigma * queue_sigma
    return max(0.0, queue_rng.lognormvariate(mu, queue_sigma))

# Metrics tracking
target_accuracy = 95.0
max_accuracy = 0.0
time_to_target = None
total_local_steps = 0
start_wall_time = time.time()
metrics_history = []

print(f"\nRunning {num_rounds} rounds...\n")

for r in range(num_rounds):
    round_start = time.time()

    # Determine local_steps for each client based on queue predictions
    local_steps_dict = {}
    lr_dict = {}
    Jk_dict = {}
    for cid in range(num_clients):
        Jk = max(0.5, Tsync - q_hat[cid] - delta)
        Ek = max(warmup_steps, int(Jk / max(c_hat[cid], 1e-6)))
        Jk_dict[cid] = Jk
        local_steps_dict[cid] = Ek

    Emin = min(local_steps_dict.values())
    for cid in range(num_clients):
        lr_dict[cid] = base_lr * (Emin / max(local_steps_dict[cid], 1))

    # Generate queue delays for this round
    queue_delays = {cid: sample_queue_delay(cid) for cid in range(num_clients)}

    # Train each client with queue delay and local_steps
    client_updates = {}
    for cid in range(num_clients):
        client = client_agents[cid]

        # Simulate queue wait
        q_delay = queue_delays[cid]
        if q_delay > 0:
            time.sleep(min(q_delay, 1.0))  # Cap sleep at 1s for speed

        train_start = time.time()
        client.train(
            round=r,
            local_steps=local_steps_dict[cid],
            learning_rate=lr_dict[cid],
            start_time=train_start,
            queue_delay=q_delay,
            origin_round=global_round_counter,
        )
        train_time = time.time() - train_start

        local_model = client.get_parameters()
        if isinstance(local_model, tuple):
            local_model, train_meta = local_model[0], local_model[1]
        else:
            train_meta = {}

        steps_done = train_meta.get("current_local_steps", local_steps_dict[cid])
        total_local_steps += steps_done

        # Update compute rate estimate
        actual_rate = train_time / max(steps_done, 1)
        c_hat[cid] = ewma_alpha * actual_rate + (1 - ewma_alpha) * c_hat[cid]

        # Determine admission
        deadline = client_deadline.get(cid, q_hat[cid] + alpha_admission * Tsync)
        admitted = q_delay <= deadline

        # Update queue estimate
        q_hat[cid] = ewma_alpha * q_delay + (1 - ewma_alpha) * q_hat[cid]
        # Update deadline for next round
        client_deadline[cid] = q_hat[cid] + alpha_admission * Tsync

        if admitted:
            client_updates[cid] = {
                "model": local_model,
                "origin_round": global_round_counter,
                "q_delay": q_delay,
            }
        else:
            late_buffer.append({
                "model": local_model,
                "origin_round": global_round_counter,
                "q_delay": q_delay,
                "client_id": cid,
            })

    # Merge late buffer clients whose origin_round < current global_round
    merged_from_buffer = []
    remaining_buffer = []
    for entry in late_buffer:
        if entry["origin_round"] < global_round_counter:
            cid = entry["client_id"]
            if cid not in client_updates:
                client_updates[cid] = {
                    "model": entry["model"],
                    "origin_round": entry["origin_round"],
                    "q_delay": entry["q_delay"],
                }
            merged_from_buffer.append(entry)
        else:
            remaining_buffer.append(entry)
    late_buffer = remaining_buffer

    # If no admitted clients, use all available
    if len(client_updates) == 0 and len(late_buffer) > 0:
        for entry in late_buffer:
            cid = entry["client_id"]
            client_updates[cid] = {
                "model": entry["model"],
                "origin_round": entry["origin_round"],
                "q_delay": entry["q_delay"],
            }
        late_buffer = []

    # Aggregate with FedQueue staleness-aware weighting
    if len(client_updates) > 0:
        local_models = {cid: u["model"] for cid, u in client_updates.items()}
        staleness = {cid: global_round_counter - u["origin_round"] for cid, u in client_updates.items()}
        local_steps_info = {cid: local_steps_dict.get(cid, warmup_steps) for cid in client_updates}

        # Manual FedQueue aggregation
        total_samples = sum(sample_sizes[str(cid)] for cid in client_updates)
        aggregation_factors = {}
        for cid in client_updates:
            w = sample_sizes[str(cid)] / total_samples if total_samples > 0 else 1.0 / len(client_updates)
            aggregation_factors[cid] = staleness_fn(staleness[cid]) * w

        factor_sum = sum(aggregation_factors.values())
        if factor_sum > 0:
            aggregation_factors = {cid: f / factor_sum for cid, f in aggregation_factors.items()}

        # Weighted average
        with torch.no_grad():
            for i, cid in enumerate(client_updates):
                model = local_models[cid]
                weight = aggregation_factors[cid]
                if i == 0:
                    global_model = {k: v.clone() * weight for k, v in model.items()}
                else:
                    for k in global_model:
                        global_model[k] += model[k] * weight

        global_round_counter += 1

    # Load global model
    for c in client_agents:
        c.load_parameters(global_model)

    # Server evaluation
    server_agent.model.load_state_dict(global_model)
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
              f"Max: {max_accuracy:.2f}% | T: {elapsed:.1f}s | "
              f"Admitted: {len(client_updates)}/4 | Steps: {sum(local_steps_dict.values())}"
              + (f" | TTA*={time_to_target:.1f}s" if time_to_target else ""))

# Final
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
    print(f"Time-to-A*:                NOT REACHED (max: {max_accuracy:.2f}%)")
print(f"Total Local Steps (#Ek):    {total_local_steps}")
print(f"Total Wall Time:            {total_wall_time:.1f}s")
print(f"Target accuracy:            {target_accuracy}%")

os.makedirs(os.path.dirname(args.output), exist_ok=True)
results = {
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
