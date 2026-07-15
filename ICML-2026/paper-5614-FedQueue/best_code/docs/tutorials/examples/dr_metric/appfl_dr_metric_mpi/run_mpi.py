import argparse
from mpi4py import MPI
from omegaconf import OmegaConf
from resources.cadre_module import CADREModule
from appfl.agent import ServerAgent
from appfl.comm.mpi import MPIClientCommunicator, MPIServerCommunicator

argparse = argparse.ArgumentParser()
argparse.add_argument("--server_config", type=str, default="./config_server.yaml")
argparse.add_argument("--client_config", type=str, default="./config_client.yaml")
args = argparse.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_clients = size - 1

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
    client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
    client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1
    client_agent_config.data_configs.dataset_kwargs.visualization = (
        True if rank == 1 else False
    )
    # Create the client agent and communicator
    client_agent = CADREModule(client_agent_config=client_agent_config)
    client_communicator = MPIClientCommunicator(comm, server_rank=0)
    # Load the configurations
    client_config = client_communicator.get_configuration()
    client_agent.load_config(client_config)
    # Generate the readiness report
    data_readiness = client_agent.generate_mnist_readiness_report()
    client_communicator.invoke_custom_action(
        action="get_mnist_readiness_report", **data_readiness
    )
    # Load the initial global model
    init_global_model = client_communicator.get_global_model(init_model=True)
    client_agent.load_parameters(init_global_model)
    # Local training and global model update iterations
    while True:
        client_agent.train()
        local_model = client_agent.get_parameters()
        if isinstance(local_model, tuple):
            local_model, meta_data_local = local_model[0], local_model[1]
        else:
            meta_data_local = {}
        new_global_model, metadata = client_communicator.update_global_model(
            local_model, **meta_data_local
        )
        if metadata["status"] == "DONE":
            break
        client_agent.load_parameters(new_global_model)
    client_communicator.invoke_custom_action(action="close_connection")
