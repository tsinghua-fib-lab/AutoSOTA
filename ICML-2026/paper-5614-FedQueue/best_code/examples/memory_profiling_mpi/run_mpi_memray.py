#!/usr/bin/env python3
"""
Memory profiling wrapper for MPI federated learning using memray
"""

import argparse
import sys
import os
import memray
from pathlib import Path
from mpi4py import MPI

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from omegaconf import OmegaConf
from appfl.agent import ServerAgent, ClientAgent
from appfl.comm.mpi import MPIServerCommunicator, MPIClientCommunicator


def run_mpi_with_profiling(
    server_config_path: str,
    client_config_path: str,
    output_dir: str = "./memory_profiles",
    use_optimized: bool = False,
):
    """Run MPI federated learning with memray profiling enabled"""

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    os.makedirs(output_dir, exist_ok=True)

    # Choose profile file name based on version and rank
    version_suffix = "_optimized" if use_optimized else "_original"
    profile_file = f"{output_dir}/mpi_rank_{rank}{version_suffix}_memory_profile.bin"

    with memray.Tracker(profile_file):
        if rank == 0:
            # Server process
            run_server(server_config_path, comm, use_optimized)
        else:
            # Client processes - use base client config for all clients
            run_client(client_config_path, comm, rank, size, use_optimized)


def run_server(config_path: str, comm, use_optimized: bool):
    """Run the MPI server with memory profiling"""
    # Load server configuration
    server_agent_config = OmegaConf.load(config_path)
    server_agent_config.server_configs.num_clients = (
        comm.Get_size() - 1
    )  # Total clients (excluding server)

    # Modify config to use optimizations if using optimized version
    if use_optimized:
        # Enable memory optimization in ServerAgent
        server_agent_config.server_configs.optimize_memory = True
        # Enable memory optimization for clients (passed through config)
        server_agent_config.client_configs.optimize_memory = True
        # Enable memory optimization in trainer
        if hasattr(server_agent_config.client_configs, "train_configs"):
            server_agent_config.client_configs.train_configs.optimize_memory = True
        # Enable memory optimization in scheduler
        if hasattr(server_agent_config.server_configs, "scheduler_kwargs"):
            server_agent_config.server_configs.scheduler_kwargs.optimize_memory = True
        # Enable memory optimization in aggregator
        if hasattr(server_agent_config.server_configs, "aggregator_kwargs"):
            server_agent_config.server_configs.aggregator_kwargs.optimize_memory = True
    else:
        # Disable memory optimization in ServerAgent
        server_agent_config.server_configs.optimize_memory = False
        # Disable memory optimization for clients (passed through config)
        server_agent_config.client_configs.optimize_memory = False
        # Disable memory optimization in trainer
        if hasattr(server_agent_config.client_configs, "train_configs"):
            server_agent_config.client_configs.train_configs.optimize_memory = False
        # Disable memory optimization in scheduler
        if hasattr(server_agent_config.server_configs, "scheduler_kwargs"):
            server_agent_config.server_configs.scheduler_kwargs.optimize_memory = False
        # Disable memory optimization in aggregator
        if hasattr(server_agent_config.server_configs, "aggregator_kwargs"):
            server_agent_config.server_configs.aggregator_kwargs.optimize_memory = False

    # Create server agent
    server_agent = ServerAgent(server_agent_config=server_agent_config)

    # Create MPI server communicator
    server_comm = MPIServerCommunicator(
        comm=comm,
        server_agent=server_agent,
        logger=server_agent.logger,
        optimize_memory=use_optimized,  # Pass optimization flag
    )

    # Run server loop
    try:
        server_comm.serve()
    except KeyboardInterrupt:
        print("[Server] Shutting down...")


def run_client(
    client_config_path: str, comm, client_rank: int, size: int, use_optimized: bool
):
    """Run an MPI client with memory profiling - following mpi/run_mpi.py pattern"""

    num_clients = size - 1  # Total clients (excluding server)

    # Load base client configuration (following mpi/run_mpi.py pattern)
    client_agent_config = OmegaConf.load(client_config_path)

    # Set client-specific configurations (following mpi/run_mpi.py pattern)
    client_agent_config.client_id = f"Client{client_rank}"

    # Update dataset configurations if they exist
    if (
        hasattr(client_agent_config, "data_configs")
        and client_agent_config.data_configs is not None
    ):
        if (
            hasattr(client_agent_config.data_configs, "dataset_kwargs")
            and client_agent_config.data_configs.dataset_kwargs is not None
        ):
            # Update dataset kwargs for this client (following mpi/run_mpi.py pattern)
            client_agent_config.data_configs.dataset_kwargs.client_id = client_rank - 1
            client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients

            # Enable visualization only for the first client (rank 1)
            if hasattr(
                client_agent_config.data_configs.dataset_kwargs, "visualization"
            ):
                client_agent_config.data_configs.dataset_kwargs.visualization = (
                    client_rank == 1
                )

    # Modify config to use optimizations if using optimized version
    if use_optimized:
        # Enable memory optimization in ClientAgent
        client_agent_config.optimize_memory = True
        # Enable memory optimization in trainer
        if hasattr(client_agent_config, "train_configs"):
            client_agent_config.train_configs.optimize_memory = True
    else:
        # Disable memory optimization in ClientAgent
        client_agent_config.optimize_memory = False
        # Disable memory optimization in trainer
        if hasattr(client_agent_config, "train_configs"):
            client_agent_config.train_configs.optimize_memory = False

    # Create client agent with customized configuration
    client_agent = ClientAgent(client_agent_config=client_agent_config)

    # Create MPI client communicator
    client_comm = MPIClientCommunicator(
        comm=comm,
        server_rank=0,  # Server is always rank 0
        client_id=client_agent_config.client_id,
        optimize_memory=use_optimized,  # Pass optimization flag
    )

    try:
        # Get additional configuration from server (following gRPC pattern)
        server_config = client_comm.get_configuration()

        # Load server configuration into existing client agent
        client_agent.load_config(server_config)

        # Get initial global model from server
        init_global_model = client_comm.get_global_model(init_model=True)

        # Load initial model into client agent
        client_agent.load_parameters(init_global_model)

        # Send sample size to server (following gRPC pattern)
        sample_size = client_agent.get_sample_size()
        client_comm.invoke_custom_action(
            action="set_sample_size", sample_size=sample_size
        )

        while True:
            # Perform local training
            client_agent.train()

            # Get local model parameters
            local_model = client_agent.get_parameters()
            if isinstance(local_model, tuple):
                local_model, metadata = local_model[0], local_model[1]
            else:
                metadata = {}

            # Send model update to server and get new global model
            new_global_model, response_metadata = client_comm.update_global_model(
                local_model=local_model, **metadata
            )

            # Check if training is complete
            if response_metadata.get("status") == "DONE":
                break

            # Update local steps if provided by server
            if "local_steps" in response_metadata:
                client_agent.trainer.train_configs.num_local_steps = response_metadata[
                    "local_steps"
                ]

            # Load new global model
            client_agent.load_parameters(new_global_model)

        # Close connection
        client_comm.invoke_custom_action("close_connection")

    except Exception as e:
        print(f"[Client {client_rank}] Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--server_config",
        type=str,
        default="./memory_profiling/configs/server_resnet_dummy.yaml",
        help="Path to the server configuration file.",
    )
    argparser.add_argument(
        "--client_config",
        type=str,
        default="./memory_profiling/configs/client_1_resnet_dummy.yaml",
        help="Path to base client configuration file (will be customized for each MPI rank).",
    )
    argparser.add_argument(
        "--output-dir",
        type=str,
        default="./memory_profiles",
        help="Directory to save memory profiles.",
    )
    argparser.add_argument(
        "--use_optimized_version",
        action="store_true",
        help="Use optimized memory-efficient versions of MPI communicators.",
    )
    args = argparser.parse_args()

    run_mpi_with_profiling(
        args.server_config,
        args.client_config,
        args.output_dir,
        args.use_optimized_version,
    )
