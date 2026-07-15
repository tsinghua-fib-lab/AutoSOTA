import os
import argparse
from mpi4py import MPI
from omegaconf import OmegaConf
from appfl.agent import ClientAgent, ServerAgent
from appfl.comm.mpi import MPIClientCommunicator, MPIServerCommunicator

argparse = argparse.ArgumentParser()

argparse.add_argument(
    "--server_config",
    type=str,
    default="./resources/configs/fedsb/fedsb_server_config.yaml",
)
argparse.add_argument(
    "--client_config",
    type=str,
    default="./resources/configs/fedsb/fedsb_client1_config.yaml",
)
args = argparse.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_clients = size - 1

os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
os.environ["MASTER_ADDR"] = "x3005c0s37b0n0"  # or the hostname of rank 0
os.environ["MASTER_PORT"] = "12355"  # ensure all processes agree on this port
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(size)

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
    # Set the client configurations
    client_agent_config = OmegaConf.load(args.client_config)
    client_agent_config.client_id = f"Client{rank}"
    client_agent_config.train_configs.client_idx = rank - 1
    # Create the client agent and communicator
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    client_communicator = MPIClientCommunicator(
        comm, server_rank=0, client_id=client_agent_config.client_id
    )
    # Load the configurations and initial global model
    client_config = client_communicator.get_configuration()
    client_agent.load_config(client_config)

    client_agent.train()
    local_model, metadata = client_agent.get_parameters()
    client_communicator.update_global_model(local_model, **metadata)
    client_communicator.invoke_custom_action(action="close_connection")
