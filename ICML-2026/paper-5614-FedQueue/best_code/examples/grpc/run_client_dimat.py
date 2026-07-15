"""
gRPC run script for DIMAT with pre-training phase.

Matches the paper setup:
  1. Pre-train each agent independently for N epochs (default 10)
  2. Run merge-train iterations via the standard FL loop

launch server first:
python grpc/run_server.py --config ./resources/configs/mnist/server_dimat.yaml

then launch two clients in separate terminals:
python grpc/run_client_dimat.py --config ./resources/configs/mnist/client_1_dimat.yaml --pretrain_epochs 10
python grpc/run_client_dimat.py --config ./resources/configs/mnist/client_2_dimat.yaml --pretrain_epochs 10
"""

import time
import argparse
from omegaconf import OmegaConf
from appfl.agent import ClientAgent
from appfl.comm.grpc import GRPCClientCommunicator


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config",
    type=str,
    default="./resources/configs/mnist/client_1_dimat.yaml",
    help="Path to the configuration file.",
)
argparser.add_argument(
    "--pretrain_epochs",
    type=int,
    default=10,
    help="Number of local pre-training epochs before merge-train begins",
)
args = argparser.parse_args()

client_agent_config = OmegaConf.load(args.config)

client_agent = ClientAgent(client_agent_config=client_agent_config)
client_communicator = GRPCClientCommunicator(
    client_id=client_agent.get_id(),
    logger=client_agent.logger,
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
    # Check CADREModule availability and if the data needs remediation
    if (
        hasattr(client_config.data_readiness_configs.dr_metrics, "cadremodule_configs")
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

# =========================================================
# Phase 1: Local pre-training (independent, no communication)
# =========================================================
pretrain_epochs = args.pretrain_epochs
if pretrain_epochs > 0:
    # Save the original num_local_epochs for the merge-train phase
    original_epochs = client_agent.trainer.train_configs.num_local_epochs

    # Set pre-training epochs
    client_agent.trainer.train_configs.num_local_epochs = pretrain_epochs
    print(
        f"[{client_agent_config.client_id}] "
        f"Pre-training for {pretrain_epochs} epochs..."
    )
    t0 = time.time()
    client_agent.train()
    dt = time.time() - t0
    print(f"[{client_agent_config.client_id}] Pre-training done in {dt:.1f}s")

    # Restore original num_local_epochs for merge-train phase
    client_agent.trainer.train_configs.num_local_epochs = original_epochs

# =========================================================
# Phase 2: Merge-train iterations (standard FL loop)
# =========================================================
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
        client_agent.trainer.train_configs.num_local_steps = metadata["local_steps"]
    client_agent.load_parameters(new_global_model)
client_communicator.invoke_custom_action(action="close_connection")
