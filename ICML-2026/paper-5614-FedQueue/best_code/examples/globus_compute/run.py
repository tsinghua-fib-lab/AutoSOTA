import pprint
import argparse
from omegaconf import OmegaConf
from concurrent.futures import Future
from appfl.agent import ServerAgent
from appfl.comm.globus_compute import GlobusComputeServerCommunicator

argparser = argparse.ArgumentParser(
    description="Federated Learning using Globus Compute"
)
argparser.add_argument(
    "--server_config",
    type=str,
    default="./resources/config_gc/mnist/server_fedcompass.yaml",
    help="Path to server configuration file",
)
argparser.add_argument(
    "--client_config",
    type=str,
    default="./resources/config_gc/mnist/clients.yaml",
    help="Path to client configuration file",
)
argparser.add_argument(
    "--compute_token", required=False, help="Globus Compute authentication token"
)
argparser.add_argument(
    "--openid_token", required=False, help="Globus OpenID authentication token"
)
argparser.add_argument("--get_sample_size", action="store_true")
args = argparser.parse_args()

# Load server and client agents configurations
print("[Setup] Loading configurations...")
server_agent_config = OmegaConf.load(args.server_config)
client_agent_configs = OmegaConf.load(args.client_config)
print(f"[Setup] Server config: {args.server_config}")
print(f"[Setup] Client config: {args.client_config}")

# Create server agent
print("[Server] Initializing server agent...")
server_agent = ServerAgent(server_agent_config=server_agent_config)
server_agent.logger.info(
    f"[Server] Will run for {server_agent_config.server_configs.num_global_epochs} global rounds"
)

# Create server communicator
server_agent.logger.info(
    "[Communicator] Initializing Globus Compute server communicator..."
)
server_communicator = GlobusComputeServerCommunicator(
    server_agent_config=server_agent.server_agent_config,
    client_agent_configs=client_agent_configs["clients"],
    logger=server_agent.logger,
    **(
        {
            "compute_token": args.compute_token,
            "openid_token": args.openid_token,
        }
        if args.compute_token is not None and args.openid_token is not None
        else {}
    ),
)
server_agent.logger.info(
    f"[Communicator] Connected to {len(client_agent_configs['clients'])} client endpoints"
)

# Get sample size from clients
if args.get_sample_size:
    server_agent.logger.info("[Clients] Requesting sample sizes from all clients...")
    server_communicator.send_task_to_all_clients(task_name="get_sample_size")
    sample_size_ret = server_communicator.recv_result_from_all_clients()[1]
    for client_endpoint_id, sample_size in sample_size_ret.items():
        server_agent.set_sample_size(client_endpoint_id, sample_size["sample_size"])
    server_agent.logger.info(
        f"[Client {client_endpoint_id}] Sample size: {sample_size['sample_size']}"
    )

# Data readiness report (if enabled)
if (
    hasattr(server_agent_config.client_configs, "data_readiness_configs")
    and hasattr(
        server_agent_config.client_configs.data_readiness_configs, "generate_dr_report"
    )
    and server_agent_config.client_configs.data_readiness_configs.generate_dr_report
):
    server_agent.logger.info("[Server] Generating data readiness report...")
    server_communicator.send_task_to_all_clients(task_name="data_readiness_report")
    readiness_reports = server_communicator.recv_result_from_all_clients()[1]
    # Restructure the data to match the function's expected input
    restructured_report = {}
    for client_endpoint_id, client_report in readiness_reports.items():
        # Assuming 'data_readiness' is the key containing the actual report
        client_data = client_report.get("data_readiness", {})
        for attribute, value in client_data.items():
            if attribute not in restructured_report:
                restructured_report[attribute] = {}
            restructured_report[attribute][client_endpoint_id] = value
    # Handle 'plots' separately if they exist
    if "plots" in restructured_report:
        plot_data = restructured_report.pop("plots")
        restructured_report["plots"] = {
            client_id: plot_data.get(client_id, {})
            for client_id in readiness_reports.keys()
        }
    # Call the data_readiness_report function
    server_agent.data_readiness_report(restructured_report)
    server_agent.logger.info("[Server] Data readiness report generated")

# Train the model
server_agent.logger.info("Starting Federated Learning Training")
init_model = server_agent.get_parameters(globus_compute_run=True)
if isinstance(init_model, tuple):
    init_model, metadata = init_model[0], init_model[1]
else:
    metadata = {}
server_communicator.send_task_to_all_clients(
    task_name="train",
    model=init_model,
    metadata=metadata,
    need_model_response=True,
)

model_futures = {}
client_rounds = {}

server_agent.logger.info("[Training] Waiting for client updates...")

while not server_agent.training_finished():
    client_endpoint_id, client_model, client_metadata = (
        server_communicator.recv_result_from_one_client()
    )

    # Track client round
    if client_endpoint_id not in client_rounds:
        client_rounds[client_endpoint_id] = 0
    client_rounds[client_endpoint_id] += 1

    server_agent.logger.info(
        f"Received model from client {client_endpoint_id} (Round {client_rounds[client_endpoint_id]}), "
        f"with metadata:\n{pprint.pformat(client_metadata)}"
    )
    global_model = server_agent.global_update(
        client_endpoint_id,
        client_model,
        **client_metadata,
    )
    if isinstance(global_model, Future):
        model_futures[client_endpoint_id] = global_model
    else:
        if isinstance(global_model, tuple):
            global_model, metadata = global_model
        else:
            metadata = {}
        metadata["round"] = client_rounds[client_endpoint_id]
        if not server_agent.training_finished():
            server_communicator.send_task_to_one_client(
                client_endpoint_id,
                task_name="train",
                model=global_model,
                metadata=metadata,
                need_model_response=True,
            )
            server_agent.logger.info(
                f"[Server] Sent updated global model to {client_endpoint_id}"
            )

    # Deal with the model futures
    del_keys = []
    for client_endpoint_id in model_futures:
        if model_futures[client_endpoint_id].done():
            global_model = model_futures[client_endpoint_id].result()
            if isinstance(global_model, tuple):
                global_model, metadata = global_model
            else:
                metadata = {}
            metadata["round"] = client_rounds[client_endpoint_id]
            if not server_agent.training_finished():
                server_communicator.send_task_to_one_client(
                    client_endpoint_id,
                    task_name="train",
                    model=global_model,
                    metadata=metadata,
                    need_model_response=True,
                )
                server_agent.logger.info(
                    f"[Server] Sent updated global model to {client_endpoint_id}"
                )
            del_keys.append(client_endpoint_id)
    for key in del_keys:
        model_futures.pop(key)

# Cleanup
server_communicator.cancel_all_tasks()
server_communicator.shutdown_all_clients()

# Finalize training
server_agent.logger.info("Federated Learning Training Completed!")
server_agent.logger.info(
    f"Results saved to: {server_agent_config.server_configs.logging_output_dirname}"
)
