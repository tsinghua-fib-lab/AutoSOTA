"""
Serial simulation of Federated learning.
"""

import argparse
import random
from omegaconf import OmegaConf
from appfl.agent import SimClientAgent, SimServerAgent
from appfl.loader import load_and_split_dataset


# Load configuration
argparser = argparse.ArgumentParser()
argparser.add_argument("--config", type=str, default="./sim_tinyimagenet.yaml")
args = argparser.parse_args()

# Parse and normalize compatible configs for server and clients
config = OmegaConf.load(args.config)
scheduler_config = config.algorithm_configs.scheduler_kwargs
num_clients = scheduler_config.num_clients
data_seed = int(config.data_configs.partition_kwargs.get("data_seed", 42))
rng = random.Random(data_seed)
data_config = config.data_configs

# Create server agent
server_agent = SimServerAgent(server_agent_config=config)

# Load dataset
client_datasets, server_dataset, dataset_meta = load_and_split_dataset(
    data_config, num_clients
)
sample_sizes = {
    str(client_id): len(data[0] if isinstance(data, (tuple, list)) else data)
    for client_id, data in enumerate(client_datasets)
}

## TODO: `dataset_meta` will be used for model loading;
## e.g., for model initialization and training.
## For example, the number of classes in the dataset
## can be used to determine the output dimension of the model.
## The input dimension can also be determined by
##the dataset meta information.

# Store the meta information and server-side evaluation set
server_agent.set_sample_size(sample_sizes=sample_sizes)
server_agent.load_server_val_dataset(server_dataset)

# Create client agents and assign datasets to clients
client_agents = [
    SimClientAgent(
        client_agent_config=config,
        client_id=client_id,
        client_dataset=client_datasets[client_id],
    )
    for client_id in range(len(client_datasets))
]

# Load initial global model from the server
global_model = server_agent.get_parameters(serial_run=True)
for client in client_agents:
    client.load_parameters(global_model)

# For partial participation, sample a subset of clients each round.
num_clients_per_round = min(
    int(scheduler_config.get("num_clients_per_round", num_clients)),
    int(num_clients),
)

# Sychrnonous simulation of FL
for r in range(config.algorithm_configs.num_global_epochs):
    new_global_models = []
    sampled_client_ids = sorted(rng.sample(range(num_clients), k=num_clients_per_round))
    sampled_clients = [client_agents[client_id] for client_id in sampled_client_ids]
    # SyncScheduler aggregates after `num_clients` submissions. For serial partial
    # participation, set that threshold to the selected population for this round.
    server_agent.scheduler.num_clients = len(sampled_clients)
    server_agent.scheduler.scheduler_configs.num_clients = len(sampled_clients)

    for client in sampled_clients:
        # Client local training
        client.train(round=r)
        local_model = client.get_parameters()

        # Need to check if `client.get_parameters()` returns a tuple of (model, metadata)
        # or just the model itself, depending on the implementation of the client agent.
        if isinstance(local_model, tuple):
            local_model, metadata = local_model[0], local_model[1]
        else:
            metadata = {}

        new_global_model_future = server_agent.global_update(
            client_id=client.get_id(),
            local_model=local_model,
            blocking=False,
            **metadata,
        )
        new_global_models.append(new_global_model_future)
    # Load the new global model from the server
    result = new_global_models[-1]
    global_model = result.result() if hasattr(result, "result") else result
    for client in client_agents:
        client.load_parameters(global_model)

# Evaluate global model on the server-side evaluation set
result = server_agent.server_validate()
if result is not None:
    test_loss, test_acc = result
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
