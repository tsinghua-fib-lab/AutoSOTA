import ray
import uuid
import time
from omegaconf import OmegaConf
from appfl.logger import ServerAgentFileLogger
from appfl.comm.ray import RayClientCommunicator
from appfl.comm.base import BaseServerCommunicator
from appfl.comm.utils.config import ClientTask
from appfl.comm.utils.s3_storage import CloudStorage, LargeObjectWrapper
from appfl.config import ServerAgentConfig, ClientAgentConfig
from typing import List, Optional, Union, Dict, OrderedDict, Any, Tuple


class RayServerCommunicator(BaseServerCommunicator):
    def __init__(
        self,
        server_agent_config: ServerAgentConfig,
        client_agent_configs: List[ClientAgentConfig],
        logger: Optional[ServerAgentFileLogger] = None,
        **kwargs,
    ):
        self.comm_type = "ray"
        super().__init__(server_agent_config, client_agent_configs, logger, **kwargs)
        ray.init(address="auto")
        self.client_actors = {}
        _client_id_check_set = set()
        for client_config in client_agent_configs:
            assert hasattr(client_config, "client_id"), (
                "Client configuration must have an client_id."
            )
            client_id = client_config["client_id"]
            assert client_id not in _client_id_check_set, (
                f"Client ID {client_id} is not unique for this client configuration.\n{client_config}"
            )
            _client_id_check_set.add(client_id)
            # Read the client dataloader source file
            with open(client_config.data_configs.dataset_path) as file:
                client_config.data_configs.dataset_source = file.read()
            del client_config.data_configs.dataset_path

            client_config.experiment_id = self.experiment_id
            client_config = OmegaConf.merge(
                server_agent_config.client_configs, client_config
            )
            client_config.comm_configs.comm_type = self.comm_type
            # If specific client is required to use specific resource type, i.e,  `assign_random = False`
            if (
                hasattr(server_agent_config.server_configs, "comm_configs")
                and hasattr(
                    server_agent_config.server_configs.comm_configs, "ray_configs"
                )
                and not server_agent_config.server_configs.comm_configs.ray_configs.get(
                    "assign_random", False
                )
            ):
                self.client_actors[client_id] = RayClientCommunicator.options(
                    resources={client_config["client_id"]: 1}
                ).remote(server_agent_config, client_config)
            else:
                self.client_actors[client_id] = RayClientCommunicator.remote(
                    server_agent_config, client_config
                )

        self.executing_tasks: Dict[str, ClientTask] = {}
        self.executing_task_futs: Dict[Any, str] = {}

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
            model_wrapper = LargeObjectWrapper(
                data=model,
                name=str(uuid.uuid4()) + "_server_state",
            )
            model = CloudStorage.upload_object(model_wrapper, register_for_clean=True)
        for i, client_id in enumerate(self.client_actors):
            client_metadata = metadata[i] if isinstance(metadata, list) else metadata
            if need_model_response and self.use_s3bucket:
                local_model_key = f"{str(uuid.uuid4())}_client_state_{client_id}"
                local_model_url = CloudStorage.presign_upload_object(local_model_key)
                client_metadata["local_model_key"] = local_model_key
                client_metadata["local_model_url"] = local_model_url
            task_id, task_ref = self.__send_task(
                self.client_actors[client_id], task_name, model, client_metadata
            )
            super()._register_task(task_id, task_ref, client_id, task_name)
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
        Send a specific task to one specific client.
        :param `client_id`: The client id to which the task is sent.
        :param `task_name`: Name of the task to be executed on the clients
        :param [Optional] `model`: Model to be sent to the clients
        :param [Optional] `metadata`: Additional metadata to be sent to the clients
        :param `need_model_response`: Whether the task requires a model response from the clients
            If so, the server will provide a pre-signed URL for the clients to upload the model if using S3.
        """
        if self.use_s3bucket and model is not None:
            model_wrapper = LargeObjectWrapper(
                data=model,
                name=str(uuid.uuid4()) + "_server_state",
            )
            model = CloudStorage.upload_object(model_wrapper, register_for_clean=True)
        if need_model_response and self.use_s3bucket:
            local_model_key = f"{str(uuid.uuid4())}_client_state_{client_id}"
            local_model_url = CloudStorage.presign_upload_object(local_model_key)
            metadata["local_model_key"] = local_model_key
            metadata["local_model_url"] = local_model_url
        task_id, task_ref = self.__send_task(
            self.client_actors[client_id], task_name, model, metadata
        )
        self._register_task(task_id, task_ref, client_id, task_name)
        self.logger.info(f"Task '{task_name}' is assigned to {client_id}.")

    def recv_result_from_all_clients(self) -> Tuple[Dict, Dict]:
        """
        Receive task results from all clients that have running tasks.
        :return `client_results`: A dictionary containing the results from all clients - Dict[client_id, client_model]
        :return `client_metadata`: A dictionary containing the metadata from all clients - Dict[client_id, client_metadata]
        """
        client_results, client_metadata = {}, {}
        while len(self.executing_task_futs):
            done_futs, _ = ray.wait(
                list(self.executing_task_futs.keys()), num_returns=1
            )

            if not done_futs:
                continue

            fut = done_futs[0]
            task_id = self.executing_task_futs[fut]
            client_id = self.executing_tasks[task_id].client_id

            try:
                result = ray.get(fut)
                client_model, client_metadata_local = self._parse_result(result)
                client_metadata_local = self._check_deprecation(
                    client_id, client_metadata_local
                )
                client_results[client_id] = client_model
                client_metadata[client_id] = client_metadata_local
                self.__update_executing_task(
                    client_metadata_local, task_id, client_id, fut
                )

            except Exception as e:
                self.logger.error(
                    f"Task {self.executing_tasks[task_id].task_name} on {client_id} failed with an error."
                )
                raise e

        return client_results, client_metadata

    def recv_result_from_one_client(self) -> Tuple[str, Any, Dict]:
        """
        Receive task results from the first client that finishes the task.
        :return `client_id`: The client id from which the result is received.
        :return `client_model`: The model returned from the client
        :return `client_metadata`: The metadata returned from the client
        """
        assert len(self.executing_task_futs), "There is no active client running tasks."
        ready_refs, _ = ray.wait(
            list(self.executing_task_futs), num_returns=1, timeout=None
        )
        finished_ref = ready_refs[0]
        task_id = self.executing_task_futs[finished_ref]
        try:
            result = ray.get(finished_ref)
            client_id = self.executing_tasks[task_id].client_id
            client_model, client_metadata = self._parse_result(result)
            client_metadata = self._check_deprecation(client_id, client_metadata)
            self.__update_executing_task(
                client_metadata, task_id, client_id, finished_ref
            )
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
        ray.shutdown()
        # Clean-up cloud storage
        if self.use_s3bucket:
            CloudStorage.clean_up()
        # Clean-up proxystore
        if hasattr(self, "proxystore") and self.proxystore is not None:
            try:
                self.proxystore.close(clear=True)
            except Exception:  # noqa: E722
                self.proxystore.close()
        self.logger.info(
            "The server and all clients have been shutted down successfully."
        )

    def cancel_all_tasks(self):
        """Cancel all on-the-fly client tasks."""
        for task_fut in self.executing_task_futs:
            ray.cancel(task_fut)
            task_id = self.executing_task_futs[task_fut]
            client_id = self.executing_tasks[task_id].client_id
            self.logger.info(f"cancelled task id: {task_id}, client id: {client_id}")
        self.executing_task_futs = {}
        self.executing_tasks = {}

    def __update_executing_task(
        self, client_metadata_local, task_id, client_id, task_fut
    ):
        try:
            self.executing_tasks[task_id].end_time = time.time()
            self.executing_tasks[task_id].success = True
            self.executing_tasks[task_id].log = client_metadata_local.get("log", {})
            # Clean up the task
            self.logger.info(
                f"Received results of task '{self.executing_tasks[task_id].task_name}' from {client_id}."
            )
            self.executing_tasks.pop(task_id)
            self.executing_task_futs.pop(task_fut)
        except Exception as e:
            self.logger.error(
                f"Task {self.executing_tasks[task_id].task_name} on {client_id} failed with an error."
            )
            raise e

    def __send_task(self, client: RayClientCommunicator, task_name, model, metadata):
        ref = None
        if task_name == "get_sample_size":
            ref = client.get_sample_size.remote()
        elif task_name == "data_readiness_report":
            ref = client.data_readiness_report.remote()
        elif task_name == "train":
            ref = client.train.remote(model, metadata)
        return str(uuid.uuid4()), ref
