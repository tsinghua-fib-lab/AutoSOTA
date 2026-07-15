import os
import pytest
import torchvision
import numpy as np
from mpi4py import MPI
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor
from appfl.agent import ClientAgent, ServerAgent
from appfl.comm.mpi import MPIClientCommunicator, MPIServerCommunicator
from appfl.comm.grpc import GRPCServerCommunicator, GRPCClientCommunicator, serve


# Prepare the MNIST data first for the tests
def readyMNISTdata():
    currentpath = os.getcwd()
    datafolderpath = os.path.join(currentpath, "_data")

    if not (os.path.exists(datafolderpath) and os.path.isdir(datafolderpath)):
        os.mkdir(datafolderpath)
    mnistfolderpath = os.path.join(datafolderpath, "MNIST")
    if not (os.path.exists(mnistfolderpath) and os.path.isdir(mnistfolderpath)):
        print("Download MNIST data")
        torchvision.datasets.MNIST(
            "./_data", download=True, train=False, transform=ToTensor()
        )


comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
if comm_size > 1:
    comm_rank = comm.Get_rank()
    if comm_rank == 0:
        readyMNISTdata()
    comm.Barrier()
else:
    # Serial
    readyMNISTdata()


# Test for MPI Communication for FedAvg
@pytest.mark.mpi(min_size=2)
def test_mpi_fedavg():
    server_config = "./tests/resources/configs/server_fedavg.yaml"
    client_config = "./tests/resources/configs/client_1.yaml"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_clients = size - 1

    if rank == 0:
        # Load and set the server configurations
        server_agent_config = OmegaConf.load(server_config)
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
        client_agent_config = OmegaConf.load(client_config)
        client_agent_config.client_id = f"Client{rank}"
        client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
        client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1
        client_agent_config.data_configs.dataset_kwargs.visualization = (
            True if rank == 1 else False
        )
        # Create the client agent and communicator
        client_agent = ClientAgent(client_agent_config=client_agent_config)
        client_communicator = MPIClientCommunicator(comm, server_rank=0)
        # Load the configurations and initial global model
        client_config = client_communicator.get_configuration()
        client_agent.load_config(client_config)
        init_global_model = client_communicator.get_global_model(init_model=True)
        client_agent.load_parameters(init_global_model)
        # Send the sample size to the server
        sample_size = client_agent.get_sample_size()
        client_communicator.invoke_custom_action(
            action="set_sample_size", sample_size=sample_size
        )
        # Local training and global model update iterations
        while True:
            client_agent.train()
            local_model = client_agent.get_parameters()
            if isinstance(local_model, tuple):
                local_model, metadata = local_model[0], local_model[1]
            else:
                metadata = {}
            new_global_model, metadata = client_communicator.update_global_model(
                local_model, **metadata
            )
            if metadata["status"] == "DONE":
                break
            if "local_steps" in metadata:
                client_agent.trainer.train_configs.num_local_steps = metadata[
                    "local_steps"
                ]
            client_agent.load_parameters(new_global_model)
        client_communicator.invoke_custom_action(action="close_connection")


# Test for MPI Communication for FedCompass
@pytest.mark.mpi(min_size=2)
def test_mpi_fedcompass():
    server_config = "./tests/resources/configs/server_fedcompass.yaml"
    client_config = "./tests/resources/configs/client_1.yaml"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_clients = size - 1

    if rank == 0:
        # Load and set the server configurations
        server_agent_config = OmegaConf.load(server_config)
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
        client_agent_config = OmegaConf.load(client_config)
        client_agent_config.client_id = f"Client{rank}"
        client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
        client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1
        client_agent_config.data_configs.dataset_kwargs.visualization = (
            True if rank == 1 else False
        )
        # Create the client agent and communicator
        client_agent = ClientAgent(client_agent_config=client_agent_config)
        client_communicator = MPIClientCommunicator(comm, server_rank=0)
        # Load the configurations and initial global model
        client_config = client_communicator.get_configuration()
        client_agent.load_config(client_config)
        init_global_model = client_communicator.get_global_model(init_model=True)
        client_agent.load_parameters(init_global_model)
        # Send the sample size to the server
        sample_size = client_agent.get_sample_size()
        client_communicator.invoke_custom_action(
            action="set_sample_size", sample_size=sample_size
        )
        # Local training and global model update iterations
        while True:
            client_agent.train()
            local_model = client_agent.get_parameters()
            if isinstance(local_model, tuple):
                local_model, metadata = local_model[0], local_model[1]
            else:
                metadata = {}
            new_global_model, metadata = client_communicator.update_global_model(
                local_model, **metadata
            )
            if metadata["status"] == "DONE":
                break
            if "local_steps" in metadata:
                client_agent.trainer.train_configs.num_local_steps = metadata[
                    "local_steps"
                ]
            client_agent.load_parameters(new_global_model)
        client_communicator.invoke_custom_action(action="close_connection")


# Test for MPI Communication for FedAsync
@pytest.mark.mpi(min_size=2)
def test_mpi_fedasync():
    server_config = "./tests/resources/configs/server_fedasync.yaml"
    client_config = "./tests/resources/configs/client_1.yaml"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_clients = size - 1

    if rank == 0:
        # Load and set the server configurations
        server_agent_config = OmegaConf.load(server_config)
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
        client_agent_config = OmegaConf.load(client_config)
        client_agent_config.client_id = f"Client{rank}"
        client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
        client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1
        client_agent_config.data_configs.dataset_kwargs.visualization = (
            True if rank == 1 else False
        )
        # Create the client agent and communicator
        client_agent = ClientAgent(client_agent_config=client_agent_config)
        client_communicator = MPIClientCommunicator(comm, server_rank=0)
        # Load the configurations and initial global model
        client_config = client_communicator.get_configuration()
        client_agent.load_config(client_config)
        init_global_model = client_communicator.get_global_model(init_model=True)
        client_agent.load_parameters(init_global_model)
        # Send the sample size to the server
        sample_size = client_agent.get_sample_size()
        client_communicator.invoke_custom_action(
            action="set_sample_size", sample_size=sample_size
        )
        # Local training and global model update iterations
        while True:
            client_agent.train()
            local_model = client_agent.get_parameters()
            if isinstance(local_model, tuple):
                local_model, metadata = local_model[0], local_model[1]
            else:
                metadata = {}
            new_global_model, metadata = client_communicator.update_global_model(
                local_model, **metadata
            )
            if metadata["status"] == "DONE":
                break
            if "local_steps" in metadata:
                client_agent.trainer.train_configs.num_local_steps = metadata[
                    "local_steps"
                ]
            client_agent.load_parameters(new_global_model)
        client_communicator.invoke_custom_action(action="close_connection")


# Test for Batched MPI Communication for FedAvg
@pytest.mark.mpi(min_size=2)
def test_batched_mpi_fedavg():
    server_config = "./tests/resources/configs/server_fedavg.yaml"
    client_config = "./tests/resources/configs/client_1.yaml"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_clients = (size - 1) * 2
    client_batch = [
        [int(num) for num in array]
        for array in np.array_split(np.arange(num_clients), size - 1)
    ]
    if rank == 0:
        # Load and set the server configurations
        server_agent_config = OmegaConf.load(server_config)
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
        client_agents = []
        client_agent_config = OmegaConf.load(client_config)
        for client_id in client_batch[rank - 1]:
            client_agent_config.client_id = f"Client{client_id}"
            client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
            client_agent_config.data_configs.dataset_kwargs.client_id = client_id
            client_agent_config.data_configs.dataset_kwargs.visualization = (
                True if client_id == 0 else False
            )
            client_agents.append(ClientAgent(client_agent_config=client_agent_config))
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

        # Local training and global model update iterations
        while True:
            client_local_models = {}
            client_metadata = {}
            for client_id, client_agent in zip(
                [f"Client{client_id}" for client_id in client_batch[rank - 1]],
                client_agents,
            ):
                client_agent.train()
                local_model = client_agent.get_parameters()
                if isinstance(local_model, tuple):
                    local_model, metadata = local_model[0], local_model[1]
                    client_metadata[client_id] = metadata
                client_local_models[client_id] = local_model
            new_global_model, metadata = client_communicator.update_global_model(
                client_local_models, kwargs=client_metadata
            )
            if all(metadata[client_id]["status"] == "DONE" for client_id in metadata):
                break
            for client_id, client_agent in zip(
                [f"Client{client_id}" for client_id in client_batch[rank - 1]],
                client_agents,
            ):
                client_agent.load_parameters(new_global_model)
        client_communicator.invoke_custom_action(action="close_connection")


# Test for Batched MPI Communication for FedAsync
@pytest.mark.mpi(min_size=2)
def test_batched_mpi_fedasync():
    server_config = "./tests/resources/configs/server_fedasync.yaml"
    client_config = "./tests/resources/configs/client_1.yaml"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_clients = (size - 1) * 2
    client_batch = [
        [int(num) for num in array]
        for array in np.array_split(np.arange(num_clients), size - 1)
    ]
    if rank == 0:
        # Load and set the server configurations
        server_agent_config = OmegaConf.load(server_config)
        server_agent_config.server_configs.num_global_epochs *= 2
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
        client_agents = []
        client_agent_config = OmegaConf.load(client_config)
        for client_id in client_batch[rank - 1]:
            client_agent_config.client_id = f"Client{client_id}"
            client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
            client_agent_config.data_configs.dataset_kwargs.client_id = client_id
            client_agent_config.data_configs.dataset_kwargs.visualization = (
                True if client_id == 0 else False
            )
            client_agents.append(ClientAgent(client_agent_config=client_agent_config))
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


# Test for MPI with Data Readiness
@pytest.mark.mpi(min_size=2)
def test_mpi_dr():
    server_config = "./tests/resources/configs/server_dr.yaml"
    client_config = "./tests/resources/configs/client_1.yaml"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_clients = size - 1
    if rank == 0:
        # Load and set the server configurations
        server_agent_config = OmegaConf.load(server_config)
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
        client_agent_config = OmegaConf.load(client_config)
        client_agent_config.client_id = f"Client{rank}"
        client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
        client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1
        client_agent_config.data_configs.dataset_kwargs.visualization = (
            True if rank == 1 else False
        )
        # Create the client agent and communicator
        client_agent = ClientAgent(client_agent_config=client_agent_config)
        client_communicator = MPIClientCommunicator(
            comm, server_rank=0, client_id=client_agent_config.client_id
        )
        # Load the configurations and initial global model
        client_config = client_communicator.get_configuration()
        client_agent.load_config(client_config)
        init_global_model = client_communicator.get_global_model(init_model=True)
        client_agent.load_parameters(init_global_model)
        # Send the sample size to the server
        sample_size = client_agent.get_sample_size()
        client_communicator.invoke_custom_action(
            action="set_sample_size", sample_size=sample_size
        )
        # Generate data readiness report
        if (
            hasattr(client_config, "data_readiness_configs")
            and hasattr(client_config.data_readiness_configs, "generate_dr_report")
            and client_config.data_readiness_configs.generate_dr_report
        ):
            data_readiness = client_agent.generate_readiness_report(client_config)
            client_communicator.invoke_custom_action(
                action="get_data_readiness_report", **data_readiness
            )
        client_communicator.invoke_custom_action(action="close_connection")


# Test for gRPC with FedAvg and Data Readiness
@pytest.mark.mpi(min_size=2)
def test_grpc():
    server_config = "./tests/resources/configs/server_dr.yaml"
    client_config = "./tests/resources/configs/client_1.yaml"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_clients = size - 1
    if rank == 0:
        server_agent_config = OmegaConf.load(server_config)
        server_agent_config.server_configs.num_clients = num_clients
        server_agent = ServerAgent(server_agent_config=server_agent_config)

        communicator = GRPCServerCommunicator(
            server_agent,
            logger=server_agent.logger,
            **server_agent_config.server_configs.comm_configs.grpc_configs,
        )

        serve(
            communicator,
            **server_agent_config.server_configs.comm_configs.grpc_configs,
        )
    else:
        client_agent_config = OmegaConf.load(client_config)
        client_agent_config.client_id = f"Client{rank}"
        client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
        client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1
        client_agent_config.data_configs.dataset_kwargs.visualization = (
            True if rank == 1 else False
        )

        client_agent = ClientAgent(client_agent_config=client_agent_config)
        client_communicator = GRPCClientCommunicator(
            client_id=client_agent.get_id(),
            **client_agent_config.comm_configs.grpc_configs,
        )

        client_config = client_communicator.get_configuration()
        client_agent.load_config(client_config)

        init_global_model = client_communicator.get_global_model(init_model=True)
        client_agent.load_parameters(init_global_model)

        # Send the number of local data to the server
        sample_size = client_agent.get_sample_size()
        client_communicator.invoke_custom_action(
            action="set_sample_size", sample_size=sample_size
        )

        # Generate data readiness report
        if (
            hasattr(client_config, "data_readiness_configs")
            and hasattr(client_config.data_readiness_configs, "generate_dr_report")
            and client_config.data_readiness_configs.generate_dr_report
        ):
            data_readiness = client_agent.generate_readiness_report(client_config)
            client_communicator.invoke_custom_action(
                action="get_data_readiness_report", **data_readiness
            )

        while True:
            client_agent.train()
            local_model = client_agent.get_parameters()
            if isinstance(local_model, tuple):
                local_model, metadata = local_model[0], local_model[1]
            else:
                metadata = {}
            new_global_model, metadata = client_communicator.update_global_model(
                local_model, **metadata
            )
            if metadata["status"] == "DONE":
                break
            if "local_steps" in metadata:
                client_agent.trainer.train_configs.num_local_steps = metadata[
                    "local_steps"
                ]
            client_agent.load_parameters(new_global_model)
        client_communicator.invoke_custom_action(action="close_connection")


# Test serial run for FedAvg
def test_serial():
    server_config = "./tests/resources/configs/server_dr.yaml"
    client_config = "./tests/resources/configs/client_1.yaml"
    num_clients = 2

    # Load server agent configurations and set the number of clients
    server_agent_config = OmegaConf.load(server_config)
    server_agent_config.server_configs.num_clients = num_clients

    # Create server agent
    server_agent = ServerAgent(server_agent_config=server_agent_config)

    # Load base client configurations and set corresponding fields for different clients
    client_agent_configs = [OmegaConf.load(client_config) for _ in range(num_clients)]
    for i in range(num_clients):
        client_agent_configs[i].client_id = f"Client{i + 1}"
        client_agent_configs[i].data_configs.dataset_kwargs.num_clients = num_clients
        client_agent_configs[i].data_configs.dataset_kwargs.client_id = i
        client_agent_configs[i].data_configs.dataset_kwargs.visualization = (
            True if i == 0 else False
        )

    # Load client agents
    client_agents = [
        ClientAgent(client_agent_config=client_agent_configs[i])
        for i in range(num_clients)
    ]

    # Get additional client configurations from the server
    client_config_from_server = server_agent.get_client_configs()
    for client_agent in client_agents:
        client_agent.load_config(client_config_from_server)

    # Load initial global model from the server
    init_global_model = server_agent.get_parameters(serial_run=True)
    for client_agent in client_agents:
        client_agent.load_parameters(init_global_model)

    # [Optional] Set number of local data to the server
    for i in range(num_clients):
        sample_size = client_agents[i].get_sample_size()
        server_agent.set_sample_size(
            client_id=client_agents[i].get_id(), sample_size=sample_size
        )

    while not server_agent.training_finished():
        new_global_models = []
        for client_agent in client_agents:
            # Client local training
            client_agent.train()
            local_model = client_agent.get_parameters()
            if isinstance(local_model, tuple):
                local_model, metadata = local_model[0], local_model[1]
            else:
                metadata = {}
            # "Send" local model to server and get a Future object for the new global model
            # The Future object will be resolved when the server receives local models from all clients
            new_global_model_future = server_agent.global_update(
                client_id=client_agent.get_id(),
                local_model=local_model,
                blocking=False,
                **metadata,
            )
            new_global_models.append(new_global_model_future)
        # Load the new global model from the server
        for client_agent, new_global_model_future in zip(
            client_agents, new_global_models
        ):
            client_agent.load_parameters(new_global_model_future.result())


# mpirun -n 2 python -m pytest --with-mpi
