import pprint
import argparse
from omegaconf import OmegaConf
from concurrent.futures import Future
from appfl.agent import ServerAgent
from appfl.comm.ray import RayServerCommunicator

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--server_config",
    type=str,
    default="./resources/config_ray/mnist/server_fedcompass.yaml",
)
argparser.add_argument(
    "--client_config", type=str, default="./resources/config_ray/mnist/clients.yaml"
)
args = argparser.parse_args()

# Load server and client agents configurations
server_agent_config = OmegaConf.load(args.server_config)
client_agent_configs = OmegaConf.load(args.client_config)

# Create server agent
server_agent = ServerAgent(server_agent_config=server_agent_config)

# Create server communicator
server_communicator = RayServerCommunicator(
    server_agent_config=server_agent.server_agent_config,
    client_agent_configs=client_agent_configs["clients"],
    logger=server_agent.logger,
)

# Get sample size from clients
server_communicator.send_task_to_all_clients(task_name="get_sample_size")
sample_size_ret = server_communicator.recv_result_from_all_clients()[1]
for client_id, sample_size in sample_size_ret.items():
    server_agent.set_sample_size(client_id, sample_size["sample_size"])

if (
    hasattr(server_agent_config.client_configs, "data_readiness_configs")
    and hasattr(
        server_agent_config.client_configs.data_readiness_configs, "generate_dr_report"
    )
    and server_agent_config.client_configs.data_readiness_configs.generate_dr_report
):
    readiness_reports_dict = {}
    server_communicator.send_task_to_all_clients(task_name="data_readiness_report")
    readiness_reports = server_communicator.recv_result_from_all_clients()[1]
    # Restructure the data to match the function's expected input
    restructured_report = {}
    for client_id, client_report in readiness_reports.items():
        # Assuming 'data_readiness' is the key containing the actual report
        client_data = client_report.get("data_readiness", {})
        for attribute, value in client_data.items():
            if attribute not in restructured_report:
                restructured_report[attribute] = {}
            restructured_report[attribute][client_id] = value
    # Handle 'plots' separately if they exist
    if "plots" in restructured_report:
        plot_data = restructured_report.pop("plots")
        restructured_report["plots"] = {
            client_id: plot_data.get(client_id, {})
            for client_id in readiness_reports.keys()
        }
    # Call the data_readiness_report function
    server_agent.data_readiness_report(restructured_report)

# Train the model
server_communicator.send_task_to_all_clients(
    task_name="train",
    model=server_agent.get_parameters(globus_compute_run=True),
    need_model_response=True,
)

model_futures = {}
client_rounds = {}
while not server_agent.training_finished():
    client_id, client_model, client_metadata = (
        server_communicator.recv_result_from_one_client()
    )
    server_agent.logger.info(
        f"Received model from client {client_id}, with metadata:\n{pprint.pformat(client_metadata)}"
    )
    global_model = server_agent.global_update(
        client_id,
        client_model,
        **client_metadata,
    )
    if isinstance(global_model, Future):
        model_futures[client_id] = global_model
    else:
        if isinstance(global_model, tuple):
            global_model, metadata = global_model
        else:
            metadata = {}
        if client_id not in client_rounds:
            client_rounds[client_id] = 0
        client_rounds[client_id] += 1
        metadata["round"] = client_rounds[client_id]
        if not server_agent.training_finished():
            server_communicator.send_task_to_one_client(
                client_id,
                task_name="train",
                model=global_model,
                metadata=metadata,
                need_model_response=True,
            )
    # Deal with the model futures
    del_keys = []
    for client_id in model_futures:
        if model_futures[client_id].done():
            global_model = model_futures[client_id].result()
            if isinstance(global_model, tuple):
                global_model, metadata = global_model
            else:
                metadata = {}
            if client_id not in client_rounds:
                client_rounds[client_id] = 0
            client_rounds[client_id] += 1
            metadata["round"] = client_rounds[client_id]
            if not server_agent.training_finished():
                server_communicator.send_task_to_one_client(
                    client_id,
                    task_name="train",
                    model=global_model,
                    metadata=metadata,
                    need_model_response=True,
                )
            del_keys.append(client_id)
    for key in del_keys:
        model_futures.pop(key)

server_communicator.cancel_all_tasks()
server_communicator.shutdown_all_clients()
