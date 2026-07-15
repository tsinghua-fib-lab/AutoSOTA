#!/usr/bin/env python3
"""
Memory profiling wrapper for gRPC client using memray
"""

import argparse
import sys
import os
import memray
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from omegaconf import OmegaConf
from appfl.agent import ClientAgent
from appfl.comm.grpc import GRPCClientCommunicator

# Optimizations are now built into the main classes via optimize_memory flags


def run_client_with_profiling(
    config_path: str,
    output_dir: str = "./memory_profiles",
    use_optimized: bool = False,
    num_clients: int = 1,
    client_idx: int = 0,
    server_uri: str = "localhost:50051",
):
    """Run client with memray profiling enabled"""
    os.makedirs(output_dir, exist_ok=True)

    # Load config to get client_id
    client_agent_config = OmegaConf.load(config_path)
    client_id = f"Client{client_idx + 1}"
    client_agent_config.client_id = client_id
    client_agent_config.comm_configs.grpc_configs.server_uri = server_uri
    client_agent_config.data_configs.dataset_kwargs.client_id = client_idx
    client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients

    # Choose profile file name based on version
    version_suffix = "_optimized" if use_optimized else "_original"
    profile_file = f"{output_dir}/client_{client_id}{version_suffix}_memory_profile.bin"

    with memray.Tracker(profile_file):
        # Modify config to use optimizations if using optimized version
        if use_optimized:
            # Note: Client will receive optimization flags from server via get_configuration()
            # But we can also set them locally for initialization
            client_agent_config.optimize_memory = True
            if hasattr(client_agent_config, "train_configs"):
                client_agent_config.train_configs.optimize_memory = True
            if hasattr(client_agent_config, "comm_configs") and hasattr(
                client_agent_config.comm_configs, "grpc_configs"
            ):
                client_agent_config.comm_configs.grpc_configs.optimize_memory = True
        else:
            # Disable memory optimization in ClientAgent
            client_agent_config.optimize_memory = False
            # Disable memory optimization in trainer
            if hasattr(client_agent_config, "train_configs"):
                client_agent_config.train_configs.optimize_memory = False
            # Disable memory optimization in communicator configs if present
            if hasattr(client_agent_config, "comm_configs") and hasattr(
                client_agent_config.comm_configs, "grpc_configs"
            ):
                client_agent_config.comm_configs.grpc_configs.optimize_memory = False

        # Create client agent with built-in optimization flags
        client_agent = ClientAgent(client_agent_config=client_agent_config)
        optimization_status = (
            "with memory optimizations" if use_optimized else "without optimizations"
        )
        print(f"Client agent created {optimization_status} for client {client_id}")

        # Create gRPC communicator with built-in optimization flags
        client_communicator = GRPCClientCommunicator(
            client_id=client_agent.get_id(),
            logger=client_agent.logger,
            **client_agent_config.comm_configs.grpc_configs,
        )
        print(f"gRPC client communicator created {optimization_status}")

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
            # Check CADREModule availability and if the data needs remediation
            if (
                hasattr(
                    client_config.data_readiness_configs.dr_metrics,
                    "cadremodule_configs",
                )
                and hasattr(
                    client_config.data_readiness_configs.dr_metrics.cadremodule_configs,
                    "remedy_action",
                )
                and client_config.data_readiness_configs.dr_metrics.cadremodule_configs.remedy_action
            ):
                client_agent.adapt_data(client_config=client_config)
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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config",
        type=str,
        default="./resources/configs/mnist/client_1.yaml",
        help="Path to the configuration file.",
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
        help="Use optimized memory-efficient versions of agents and communicators.",
    )
    argparser.add_argument(
        "--num_clients",
        type=int,
        default=2,
        help="Number of clients to simulate.",
    )
    argparser.add_argument(
        "--client_idx",
        type=int,
        default=0,
        help="Index of the client.",
    )
    argparser.add_argument(
        "--server_uri",
        type=str,
        default="localhost:50051",
        help="URI of the gRPC server.",
    )
    args = argparser.parse_args()

    run_client_with_profiling(
        args.config,
        args.output_dir,
        args.use_optimized_version,
        args.num_clients,
        args.client_idx,
        args.server_uri,
    )
