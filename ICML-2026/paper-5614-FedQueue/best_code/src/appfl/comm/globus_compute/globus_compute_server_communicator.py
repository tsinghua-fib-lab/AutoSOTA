import time
import uuid
import warnings
from omegaconf import OmegaConf
from collections import OrderedDict
from globus_sdk.scopes import AuthScopes, ComputeScopes
from globus_sdk import AccessTokenAuthorizer
from globus_compute_sdk import Executor, Client
from concurrent.futures import as_completed
from typing import Optional, Dict, List, Union, Tuple, Any
from appfl.comm.utils.s3_utils import send_model_by_s3
from appfl.logger import ServerAgentFileLogger
from appfl.comm.base import BaseServerCommunicator
from appfl.comm.utils.s3_storage import CloudStorage
from appfl.config import ClientAgentConfig, ServerAgentConfig
from .utils.endpoint import GlobusComputeClientEndpoint
from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk.sdk.login_manager import AuthorizerLoginManager


class GlobusComputeServerCommunicator(BaseServerCommunicator):
    """
    Communicator used by the federated learning server which plans to use Globus Compute
    for orchestrating the federated learning experiments.

    Globus Compute is a distributed function-as-a-service platform that allows users to run
    functions on specified remote endpoints. For more details, check the Globus Compute SDK
    documentation at https://globus-compute.readthedocs.io/en/latest/endpoints.html.

    :param `gcc`: Globus Compute client object
    :param `server_agent_config`: The server agent configuration
    :param `client_agent_configs`: A list of client agent configurations.
    :param [Optional] `logger`: Optional logger object.
    """

    def __init__(
        self,
        server_agent_config: ServerAgentConfig,
        client_agent_configs: List[ClientAgentConfig],
        logger: Optional[ServerAgentFileLogger] = None,
        **kwargs,
    ):
        self.comm_type = "globus_compute"
        super().__init__(
            server_agent_config=server_agent_config,
            client_agent_configs=client_agent_configs,
            logger=logger,
            **kwargs,
        )
        self._load_gce(**kwargs)

        # Initiate the Globus Compute client endpoints.
        self.client_endpoints: Dict[str, GlobusComputeClientEndpoint] = {}
        _client_id_check_set = set()
        for client_config in client_agent_configs:
            assert hasattr(client_config, "endpoint_id"), (
                "Client configuration must have an endpoint_id."
            )
            # Read the client dataloader source file
            with open(client_config.data_configs.dataset_path) as file:
                client_config.data_configs.dataset_source = file.read()
            del client_config.data_configs.dataset_path
            client_id = str(
                client_config.client_id
                if hasattr(client_config, "client_id")
                else (
                    client_config.train_configs.logging_id
                    if (
                        hasattr(client_config, "train_configs")
                        and hasattr(client_config.train_configs, "logging_id")
                    )
                    else client_config.endpoint_id
                )
            )
            assert client_id not in _client_id_check_set, (
                f"Client ID {client_id} is not unique for this client configuration.\n{client_config}"
            )
            _client_id_check_set.add(client_id)
            client_endpoint_id = client_config.endpoint_id
            client_config.experiment_id = self.experiment_id
            client_config.comm_type = self.comm_type
            # Raise deprecation warning for logging_id
            if hasattr(client_config.train_configs, "logging_id"):
                warnings.warn(
                    "client_agent_config.train_configs.logging_id is deprecated. Please use client_id instead.",
                    DeprecationWarning,
                )
            # logging information regarding wandb
            if hasattr(
                client_config, "wandb_configs"
            ) and client_config.wandb_configs.get("enable_wandb", False):
                self.logger.info(f"{client_id} is using wandb for logging. ")

            self.client_endpoints[client_id] = GlobusComputeClientEndpoint(
                client_id=client_id,
                client_endpoint_id=client_endpoint_id,
                client_config=OmegaConf.merge(
                    server_agent_config.client_configs, client_config
                ),
            )

    def send_task_to_all_clients(
        self,
        task_name: str,
        *,
        model: Optional[Union[Dict, OrderedDict, bytes]] = None,
        metadata: Union[Dict, List[Dict]] = {},
        need_model_response: bool = False,
    ):
        """
        Send a specific task to all clients.
        :param `task_name`: Name of the task to be executed on the clients
        :param [Optional] `model`: Model to be sent to the clients
        :param [Optional] `metadata`: Additional metadata to be sent to the clients
        :param `need_model_response`: Whether the task requires a model response from the clients
            If so, the server will provide a pre-signed URL for the clients to upload the model if using S3.
        """
        if self.use_s3bucket and model is not None:
            model = send_model_by_s3(
                self.experiment_id, self.comm_type, model, "server"
            )
        elif self.use_proxystore and model is not None:
            model = self.proxystore.proxy(model)
        for i, client_id in enumerate(self.client_endpoints):
            client_metadata = metadata[i] if isinstance(metadata, list) else metadata
            if need_model_response and self.use_s3bucket:
                local_model_key = f"{str(uuid.uuid4())}_client_state_{client_id}"
                local_model_url = CloudStorage.presign_upload_object(local_model_key)
                client_metadata["local_model_key"] = local_model_key
                client_metadata["local_model_url"] = local_model_url
            task_id, task_future = self.client_endpoints[client_id].submit_task(
                self.gce,
                task_name,
                model,
                client_metadata,
            )
            self._register_task(task_id, task_future, client_id, task_name)
            self.logger.info(f"Task '{task_name}' is assigned to {client_id}.")

    def send_task_to_one_client(
        self,
        client_id: str,
        task_name: str,
        *,
        model: Optional[Union[Dict, OrderedDict, bytes]] = None,
        metadata: Optional[Dict] = {},
        need_model_response: bool = False,
    ):
        """
        Send a specific task to one specific client endpoint.
        :param `client_id`: The client id to which the task is sent.
        :param `task_name`: Name of the task to be executed on the clients
        :param [Optional] `model`: Model to be sent to the clients
        :param [Optional] `metadata`: Additional metadata to be sent to the clients
        :param `need_model_response`: Whether the task requires a model response from the clients
            If so, the server will provide a pre-signed URL for the clients to upload the model if using S3.
        """
        if self.use_s3bucket and model is not None:
            model = send_model_by_s3(
                self.experiment_id, self.comm_type, model, "server"
            )
        elif self.use_proxystore and model is not None:
            model = self.proxystore.proxy(model)
        if need_model_response and self.use_s3bucket:
            local_model_key = f"{str(uuid.uuid4())}_client_state_{client_id}"
            local_model_url = CloudStorage.presign_upload_object(local_model_key)
            metadata["local_model_key"] = local_model_key
            metadata["local_model_url"] = local_model_url
        task_id, task_future = self.client_endpoints[client_id].submit_task(
            self.gce,
            task_name,
            model,
            metadata,
        )
        self._register_task(task_id, task_future, client_id, task_name)
        self.logger.info(f"Task '{task_name}' is assigned to {client_id}.")

    def recv_result_from_all_clients(self) -> Tuple[Dict, Dict]:
        """
        Receive task results from all clients that have running tasks.
        :return `client_results`: A dictionary containing the results from all clients - Dict[client_id, client_model]
        :return `client_metadata`: A dictionary containing the metadata from all clients - Dict[client_id, client_metadata]
        """
        client_results, client_metadata = {}, {}
        while len(self.executing_task_futs):
            fut = next(as_completed(list(self.executing_task_futs)))
            task_id = self.executing_task_futs[fut]
            client_id = self.executing_tasks[task_id].client_id
            try:
                result = fut.result()
                client_model, client_metadata_local = self._parse_result(result)
                client_metadata_local = self._check_deprecation(
                    client_id, client_metadata_local
                )
                client_results[client_id] = client_model
                client_metadata[client_id] = client_metadata_local
                # Set the status of the finished task
                client_log = client_metadata_local.get("log", {})
                self.executing_tasks[task_id].end_time = time.time()
                self.executing_tasks[task_id].success = True
                self.executing_tasks[task_id].log = client_log  # TODO: Check this line
                # Clean up the task
                self.logger.info(
                    f"Received results of task '{self.executing_tasks[task_id].task_name}' from {client_id}."
                )
                self.client_endpoints[client_id].status
                self.executing_tasks.pop(task_id)
                self.executing_task_futs.pop(fut)
            except Exception as e:
                self.logger.error(
                    f"Task {self.executing_tasks[task_id].task_name} on {client_id} failed with an error."
                )
                raise e
        return client_results, client_metadata

    def recv_result_from_one_client(self) -> Tuple[str, Any, Dict]:
        """
        Receive task results from the first client that finishes the task.
        :return `client_id`: The client endpoint id from which the result is received.
        :return `client_model`: The model returned from the client
        :return `client_metadata`: The metadata returned from the client
        """
        assert len(self.executing_task_futs), (
            "There is no active client endpoint running tasks."
        )
        try:
            fut = next(as_completed(list(self.executing_task_futs)))
            task_id = self.executing_task_futs[fut]
            result = fut.result()
            client_id = self.executing_tasks[task_id].client_id
            client_model, client_metadata = self._parse_result(result)
            client_metadata = self._check_deprecation(client_id, client_metadata)
            # Set the status of the finished task
            client_log = client_metadata.get("log", {})
            self.executing_tasks[task_id].end_time = time.time()
            self.executing_tasks[task_id].success = True
            self.executing_tasks[task_id].log = client_log  # TODO: Check this line
            # Clean up the task
            self.logger.info(
                f"Received results of task '{self.executing_tasks[task_id].task_name}' from {client_id}."
            )
            self.client_endpoints[client_id].status
            self.executing_tasks.pop(task_id)
            self.executing_task_futs.pop(fut)
        except Exception as e:
            client_id = self.executing_tasks[task_id].client_id
            self.logger.error(
                f"Task {self.executing_tasks[task_id].task_name} on {client_id} failed with an error."
            )
            raise e
        return client_id, client_model, client_metadata

    def shutdown_all_clients(self):
        """Cancel all the running tasks on the clients and shutdown the globus compute executor."""
        self.logger.info("Shutting down all clients......")
        self.gce.shutdown(wait=False, cancel_futures=True)
        # Clean-up cloud storage
        if self.use_s3bucket:
            CloudStorage.clean_up()
        # Clean-up proxystore
        if hasattr(self, "proxystore") and self.proxystore is not None:
            try:
                self.proxystore.close(clear=True)
            except:  # noqa: E722
                self.proxystore.close()
        self.logger.info(
            "The server and all clients have been shutted down successfully."
        )

    def cancel_all_tasks(self):
        """Cancel all on-the-fly client tasks."""
        for task_fut in self.executing_task_futs:
            task_fut.cancel()
            task_id = self.executing_task_futs[task_fut]
            client_id = self.executing_tasks[task_id].client_id
            self.client_endpoints[client_id].cancel_task()
        self.executing_task_futs = {}
        self.executing_tasks = {}

    def _load_gce(self, **kwargs):
        """
        Load the Globus Compute Executor.
        """
        # Assert compute_token and openid_token are both provided if necessary
        assert ("compute_token" in kwargs and "openid_token" in kwargs) or (
            "compute_token" not in kwargs and "openid_token" not in kwargs
        ), (
            "Both compute_token and openid_token must be provided if one of them is provided."
        )

        if "compute_token" in kwargs and "openid_token" in kwargs:
            compute_login_manager = AuthorizerLoginManager(
                authorizers={
                    ComputeScopes.resource_server: AccessTokenAuthorizer(
                        kwargs["compute_token"]
                    ),
                    AuthScopes.resource_server: AccessTokenAuthorizer(
                        kwargs["openid_token"]
                    ),
                }
            )
            compute_login_manager.ensure_logged_in()
            gcc = Client(
                login_manager=compute_login_manager,
                code_serialization_strategy=CombinedCode(),
            )
        else:
            gcc = Client(
                code_serialization_strategy=CombinedCode(),
            )
        self.gce = Executor(client=gcc)  # Globus Compute Executor
