import argparse
from omegaconf import OmegaConf
from appfl.agent import ServerAgent
from appfl.comm.grpc import GRPCServerCommunicator, serve

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config",
    type=str,
    default="./resources/configs/mnist/server_fedavg.yaml",
    help="Path to the configuration file.",
)
args = argparser.parse_args()

server_agent_config = OmegaConf.load(args.config)
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
