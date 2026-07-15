import argparse
from omegaconf import OmegaConf
from appfl.agent import ClientAgent
from appfl.comm.grpc import GRPCClientCommunicator


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config",
    type=str,
    default="./resources/configs/mnist/client_1.yaml",
    help="Path to the configuration file.",
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
if isinstance(init_global_model, tuple):
    init_global_model, metadata = init_global_model[0], init_global_model[1]
else:
    metadata = {}
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

while True:
    client_agent.train(**metadata)
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
