import time
import socket
import argparse
import numpy as np
from mpi4py import MPI
from omegaconf import OmegaConf
from appfl.agent import ClientAgent, ServerAgent
from appfl.comm.mpi import MPIClientCommunicator, MPIServerCommunicator

argparse = argparse.ArgumentParser()

argparse.add_argument(
    "--server_config",
    type=str,
    default="./resources/configs/cifar10/server_fedavg_scaling.yaml",
)
argparse.add_argument(
    "--client_config",
    type=str,
    default="./resources/configs/cifar10/client_1.yaml",
)
argparse.add_argument(
    "--clients_per_gpu",
    type=int,
    default=1,
    help="Number of clients per GPU, used to set the device for each client.",
)
argparse.add_argument(
    "--gpu_per_node",
    type=int,
    default=4,
    help="Number of GPUs per node, used to set the device for each client. Default is 4 on Polaris.",
)
args = argparse.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_clients = size - 1

if rank == 0:
    # Load and set the server configurations
    server_agent_config = OmegaConf.load(args.server_config)
    server_agent_config.server_configs.num_clients = num_clients
    server_agent_config.server_configs.benchmarking = True
    # Create the server agent and communicator
    server_agent = ServerAgent(server_agent_config=server_agent_config)
    server_communicator = MPIServerCommunicator(
        comm, server_agent, logger=server_agent.logger
    )
    # Start the server to serve the clients
    server_communicator.serve()
else:
    local_round_times = []
    # Set the client configurations
    client_agent_config = OmegaConf.load(args.client_config)
    client_agent_config.client_id = f"Client{rank - 1}"
    client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
    client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1
    client_agent_config.data_configs.dataset_kwargs.visualization = (
        True if rank == 1 else False
    )
    num_clients_per_node = args.clients_per_gpu * args.gpu_per_node
    client_agent_config.train_configs.device = (
        f"cuda:{(rank - 1) % num_clients_per_node // args.clients_per_gpu}"
    )
    # Create the client agent and communicator
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    client_agent.logger.info(
        f"Client {client_agent_config.client_id} is using device {client_agent_config.train_configs.device} on host {socket.gethostname()}"
    )
    client_communicator = MPIClientCommunicator(
        comm, server_rank=0, client_id=client_agent_config.client_id
    )
    # Load the configurations and initial global model
    client_config = client_communicator.get_configuration()
    client_agent.load_config(client_config)
    init_global_model, metadata = client_communicator.get_global_model(init_model=True)
    local_round_start_time = time.time()
    send_time = metadata["send_time"]
    recv_time = time.time()
    client_agent.load_parameters(init_global_model)

    # Local training and global model update iterations
    while True:
        train_start_time = time.time()
        client_agent.train()
        local_model = client_agent.get_parameters()
        train_end_time = time.time()
        if isinstance(local_model, tuple):
            local_model, metadata = local_model[0], local_model[1]
        else:
            metadata = {}
        metadata["training_time"] = train_end_time - train_start_time
        metadata["communication_time"] = recv_time - send_time
        metadata["send_time"] = time.time()
        new_global_model, metadata = client_communicator.update_global_model(
            local_model, **metadata
        )
        send_time = metadata["send_time"]
        recv_time = time.time()
        local_round_times.append(time.time() - local_round_start_time)
        local_round_start_time = time.time()
        if metadata["status"] == "DONE":
            break
        if "local_steps" in metadata:
            client_agent.trainer.train_configs.num_local_steps = metadata["local_steps"]
        client_agent.load_parameters(new_global_model)
    client_communicator.invoke_custom_action(action="close_connection")
    if rank == 1:
        # log the average local round time and standard deviation
        client_agent.logger.info(
            f"Average local round time: {np.mean(local_round_times):.2f} seconds ± {np.std(local_round_times):.2f} seconds"
        )
