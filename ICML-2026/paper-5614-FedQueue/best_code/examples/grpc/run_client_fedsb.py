import argparse
from omegaconf import OmegaConf
from appfl.agent import ClientAgent
from appfl.comm.grpc import GRPCClientCommunicator


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config",
    type=str,
    default="./resources/configs/fedsb/fedsb_client1_config.yaml",
    help="Path to the configuration file.",
)
args = argparser.parse_args()

client_agent_config = OmegaConf.load(args.config)

client_agent = ClientAgent(client_agent_config=client_agent_config)
client_communicator = GRPCClientCommunicator(
    client_id=client_agent.get_id(),
    **client_agent_config.comm_configs.grpc_configs,
)

client_config = client_communicator.get_configuration()
client_agent.load_config(client_config)

client_agent.train()
local_model, metadata = client_agent.get_parameters()
new_global_model, metadata = client_communicator.update_global_model(
    local_model, **metadata
)

client_communicator.invoke_custom_action(action="close_connection")
