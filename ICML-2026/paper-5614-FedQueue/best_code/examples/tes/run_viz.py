#!/usr/bin/env python3
"""
APPFL TES Federated Learning Runner

Similar to examples/globus_compute/run.py but for TES integration.
"""

import pprint
import argparse
from urllib.parse import urlparse
from omegaconf import OmegaConf
from concurrent.futures import Future
from appfl.agent import ServerAgent
from appfl.comm.tes import TESServerCommunicator

import fedviz
from fedviz.emitters import SSEEmitter
from fedviz.geo import get_location, is_local, parse_ip


class FedVizServerAgent(ServerAgent):
    def __init__(self, *args, client_agent_configs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_round = -1
        self._client_locs = {}

        # Build peer map from TES endpoint URLs in client configs
        self._peer_by_client = {}
        for cfg in client_agent_configs or []:
            client_id = str(cfg.get("client_id", ""))
            endpoint = (
                cfg.get("comm_configs", {})
                .get("tes_configs", {})
                .get("tes_endpoint", "")
            )
            if client_id and endpoint:
                host = urlparse(endpoint).hostname or ""
                self._peer_by_client[client_id] = f"ipv4:{host}"

    def global_update(self, client_id, local_model, *args, **kwargs):
        round_num = kwargs.get("round", 0)

        if round_num != self._current_round:
            if self._current_round >= 0:
                fedviz.log_round(round=self._current_round)
            self._current_round = round_num
            fedviz.round_start(round_num)

        # Resolve geo once per client from the TES endpoint IP
        raw_peer = self._peer_by_client.get(str(client_id), "")
        parsed_ip = parse_ip(raw_peer)
        geo = {}
        if parsed_ip and not is_local(parsed_ip):
            if str(client_id) not in self._client_locs:
                self._client_locs[str(client_id)] = get_location(parsed_ip)
            loc = self._client_locs.get(str(client_id)) or {}
            geo = {
                k: v for k, v in loc.items() if k in ("lat", "lng", "city", "country")
            }

        result = super().global_update(client_id, local_model, *args, **kwargs)
        fedviz.log_client_update(client_id=client_id, **kwargs, **geo)

        return result


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--server_config",
    type=str,
    default="./resources/config_tes/simple_net/server.yaml",
)
argparser.add_argument(
    "--client_config",
    type=str,
    default="./resources/config_tes/simple_net/clients.yaml",
)
argparser.add_argument("--auth_token", required=False)
args = argparser.parse_args()

# Load server and client agent configurations
server_agent_config = OmegaConf.load(args.server_config)
client_agent_configs = OmegaConf.load(args.client_config)

fedviz.init(
    algorithm="Federated GWAS Meta-Analysis",
    config=OmegaConf.to_container(server_agent_config.server_configs, resolve=True),
    emitters=[
        SSEEmitter(port=7070, serve_map=True),
    ],
)


# Override auth token if provided
if args.auth_token:
    if "comm_configs" not in server_agent_config:
        server_agent_config.comm_configs = {}
    if "tes_configs" not in server_agent_config.comm_configs:
        server_agent_config.comm_configs.tes_configs = {}
    server_agent_config.comm_configs.tes_configs.auth_token = args.auth_token

# Create server agent
server_agent = FedVizServerAgent(
    server_agent_config=server_agent_config,
    client_agent_configs=OmegaConf.to_container(
        client_agent_configs["clients"], resolve=True
    ),
)

# Create server communicator
server_communicator = TESServerCommunicator(
    server_agent_config=server_agent.server_agent_config,
    client_agent_configs=client_agent_configs["clients"],
    logger=server_agent.logger,
    **({"auth_token": args.auth_token} if args.auth_token else {}),
)

# Get sample size from clients
server_communicator.send_task_to_all_clients(task_name="get_sample_size")
sample_size_ret = server_communicator.recv_result_from_all_clients()[1]
server_agent.logger.info(
    f"Sample sizes from clients: {pprint.pformat(sample_size_ret)}"
)
for client_id, sample_size_data in sample_size_ret.items():
    if sample_size_data and "sample_size" in sample_size_data:
        server_agent.set_sample_size(client_id, sample_size_data["sample_size"])

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
    if len(client_metadata) > 0:
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

fedviz.log_round(round=server_agent._current_round)
fedviz.finish()
