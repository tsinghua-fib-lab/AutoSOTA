import copy
import argparse
import numpy as np
from mpi4py import MPI
from typing import List
from omegaconf import OmegaConf
from appfl.agent import ClientAgent, ServerAgent
from appfl.comm.mpi import MPIClientCommunicator, MPIServerCommunicator

argparse = argparse.ArgumentParser()
argparse.add_argument(
    "--server_config",
    type=str,
    default="./resources/configs/mnist/server_fedasync.yaml",
)
argparse.add_argument(
    "--client_config", type=str, default="./resources/configs/mnist/client_1.yaml"
)
argparse.add_argument("--num_clients", type=int, default=10)
args = argparse.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_clients = max(args.num_clients, size - 1)
# Split the clients into batches for each rank
client_batch = [
    [int(num) for num in array]
    for array in np.array_split(np.arange(num_clients), size - 1)
]

if rank == 0:
    # Load and set the server configurations
    server_agent_config = OmegaConf.load(args.server_config)
    server_agent_config.server_configs.num_clients = num_clients
    # Create the server agent and communicator
    server_agent = ServerAgent(server_agent_config=server_agent_config)
    server_communicator = MPIServerCommunicator(
        comm, server_agent, logger=server_agent.logger
    )
    # Start the server to serve the clients
    server_communicator.serve()
else:
    # Create client agents for each client in the batch
    client_agents: List[ClientAgent] = []
    client_agent_config = OmegaConf.load(args.client_config)
    for batch_idx, client_id in enumerate(client_batch[rank - 1]):
        client_agent_config.client_id = f"Client{client_id}"
        client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
        client_agent_config.data_configs.dataset_kwargs.client_id = client_id
        client_agent_config.data_configs.dataset_kwargs.visualization = (
            True if client_id == 0 else False
        )
        # Only enable wandb logging for the first client in the batch is sufficient
        if hasattr(
            client_agent_config, "wandb_configs"
        ) and client_agent_config.wandb_configs.get("enable_wandb", False):
            client_agent_config.wandb_configs.enable_wandb = (
                True if batch_idx == 0 else False
            )
        client_agents.append(
            ClientAgent(client_agent_config=copy.deepcopy(client_agent_config))
        )
    # Create the client communicator for batched clients
    client_communicator = MPIClientCommunicator(
        comm,
        server_rank=0,
        client_ids=[f"Client{client_id}" for client_id in client_batch[rank - 1]],
    )
    # Get and load the general client configurations
    client_config = client_communicator.get_configuration()
    for client_agent in client_agents:
        client_agent.load_config(client_config)
    # Get and load the initial global model
    init_global_model = client_communicator.get_global_model(init_model=True)
    for client_agent in client_agents:
        client_agent.load_parameters(init_global_model)
    # Send the sample size to the server
    client_sample_sizes = {
        client_id: {"sample_size": client_agent.get_sample_size(), "sync": True}
        for client_id, client_agent in zip(
            [f"Client{client_id}" for client_id in client_batch[rank - 1]],
            client_agents,
        )
    }
    client_communicator.invoke_custom_action(
        action="set_sample_size", kwargs=client_sample_sizes
    )

    # Generate data readiness report
    if (
        hasattr(client_config, "data_readiness_configs")
        and hasattr(client_config.data_readiness_configs, "generate_dr_report")
        and client_config.data_readiness_configs.generate_dr_report
    ):
        data_readiness = {
            client_id: client_agent.generate_readiness_report(client_config)
            for client_id, client_agent in zip(
                [f"Client{client_id}" for client_id in client_batch[rank - 1]],
                client_agents,
            )
        }
        client_communicator.invoke_custom_action(
            action="get_data_readiness_report", kwargs=data_readiness
        )

    # Local training and global model update iterations
    finish_flag = False
    while True:
        for client_id, client_agent in zip(
            [f"Client{client_id}" for client_id in client_batch[rank - 1]],
            client_agents,
        ):
            client_agent.train()
            local_model = client_agent.get_parameters()
            if isinstance(local_model, tuple):
                local_model, metadata = local_model
            else:
                metadata = {}
            new_global_model, metadata = client_communicator.update_global_model(
                local_model, client_id=client_id, **metadata
            )
            if metadata["status"] == "DONE":
                finish_flag = True
                break
            client_agent.load_parameters(new_global_model)
        if finish_flag:
            break
    client_communicator.invoke_custom_action(action="close_connection")
