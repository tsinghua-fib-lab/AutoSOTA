import os
import time
import logging
import pathlib
from datetime import datetime
from abc import abstractmethod
from omegaconf import OmegaConf
from proxystore.store import Store
from proxystore.proxy import Proxy, extract
from appfl.comm.utils.config import ClientTask
from appfl.logger import ServerAgentFileLogger
from appfl.misc.utils import get_proxystore_connector
from appfl.comm.utils.s3_storage import CloudStorage
from appfl.config import ClientAgentConfig, ServerAgentConfig
from typing import List, Optional, Union, Dict, OrderedDict, Tuple, Any


class BaseServerCommunicator:
    def __init__(
        self,
        server_agent_config: ServerAgentConfig,
        client_agent_configs: List[ClientAgentConfig],
        logger: Optional[ServerAgentFileLogger] = None,
        **kwargs,
    ):
        self.server_agent_config = server_agent_config
        self.client_agent_configs = client_agent_configs
        self.logger = logger if logger is not None else self._default_logger()
        self.experiment_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self._sanity_check()
        self._check_and_initialize_s3(server_agent_config)
        self._load_proxystore(server_agent_config)
        assert not (self.use_proxystore and self.use_s3bucket), (
            "Proxystore and S3 bucket cannot be used together."
        )
        self.executing_tasks: Dict[str, ClientTask] = {}
        self.executing_task_futs: Dict[Any, str] = {}

    @abstractmethod
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
        pass

    @abstractmethod
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

    @abstractmethod
    def recv_result_from_all_clients(self) -> Tuple[Dict, Dict]:
        """
        Receive task results from all clients that have running tasks.
        :return `client_results`: A dictionary containing the results from all clients - Dict[client_id, client_model]
        :return `client_metadata`: A dictionary containing the metadata from all clients - Dict[client_id, client_metadata]
        """
        pass

    @abstractmethod
    def recv_result_from_one_client(self) -> Tuple[str, Any, Dict]:
        """
        Receive task results from the first client that finishes the task.
        :return `client_id`: The client id from which the result is received.
        :return `client_model`: The model returned from the client
        :return `client_metadata`: The metadata returned from the client
        """
        pass

    @abstractmethod
    def shutdown_all_clients(self):
        """Cancel all the running tasks on the clients and shutdown the globus compute executor."""
        pass

    @abstractmethod
    def cancel_all_tasks(self):
        """Cancel all on-the-fly client tasks."""
        pass

    def _default_logger(self):
        """Create a default logger for the gRPC server if no logger provided."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s %(levelname)-4s server]: %(message)s")
        s_handler = logging.StreamHandler()
        s_handler.setLevel(logging.INFO)
        s_handler.setFormatter(fmt)
        logger.addHandler(s_handler)
        return logger

    def _register_task(self, task_id, task_fut, client_id, task_name):
        """
        Register new client task to the list of executing tasks - call after task submission.
        """
        self.executing_tasks[task_id] = OmegaConf.structured(
            ClientTask(
                task_id=task_id,
                task_name=task_name,
                client_id=client_id,
                start_time=time.time(),
            )
        )
        self.executing_task_futs[task_fut] = task_id

    def _check_and_initialize_s3(self, server_agent_config):
        # check if s3 enable
        self.use_s3bucket = False
        s3_bucket = None
        if hasattr(server_agent_config.server_configs, "comm_configs") and hasattr(
            server_agent_config.server_configs.comm_configs, "s3_configs"
        ):
            self.use_s3bucket = (
                server_agent_config.server_configs.comm_configs.s3_configs.get(
                    "enable_s3", False
                )
            )
            s3_bucket = server_agent_config.server_configs.comm_configs.s3_configs.get(
                "s3_bucket", None
            )
            self.use_s3bucket = self.use_s3bucket and s3_bucket is not None
        # backward compatibility for globus compute
        if (
            hasattr(server_agent_config.server_configs, "comm_configs")
            and hasattr(
                server_agent_config.server_configs.comm_configs,
                "globus_compute_configs",
            )
            and hasattr(
                server_agent_config.server_configs.comm_configs.globus_compute_configs,
                "s3_bucket",
            )
        ):
            self.logger.warning(
                "[Deprecation] Use of globus_compute_configs in server configs is deprecated. Moving forward use s3_configs key to configure AWS S3 you can find new examples here https://github.com/APPFL/APPFL/blob/main/examples/resources/config_gc/"
            )
            s3_bucket = server_agent_config.server_configs.comm_configs.globus_compute_configs.get(
                "s3_bucket", None
            )
            self.use_s3bucket = s3_bucket is not None
            # copy globus_compute_configs to s3_configs
            server_agent_config.server_configs.comm_configs.s3_configs = (
                server_agent_config.server_configs.comm_configs.globus_compute_configs
            )
            server_agent_config.server_configs.comm_configs.s3_configs["enable_s3"] = (
                self.use_s3bucket
            )

        if self.use_s3bucket:
            self.logger.info(f"Using S3 bucket {s3_bucket} for model transfer.")
            s3_creds_file = (
                server_agent_config.server_configs.comm_configs.s3_configs.get(
                    "s3_creds_file", None
                )
            )
            s3_temp_dir_default = str(
                pathlib.Path.home()
                / ".appfl"
                / self.comm_type
                / "server"
                / self.experiment_id
            )
            s3_temp_dir = (
                server_agent_config.server_configs.comm_configs.s3_configs.get(
                    "s3_temp_dir", s3_temp_dir_default
                )
            )
            if not os.path.exists(s3_temp_dir):
                pathlib.Path(s3_temp_dir).mkdir(parents=True, exist_ok=True)
            CloudStorage.init(s3_bucket, s3_creds_file, s3_temp_dir, self.logger)

    def _load_proxystore(self, server_agent_config) -> None:
        """
        Create the proxystore for storing and sending model parameters from the server to the clients.
        """
        self.proxystore = None
        self.use_proxystore = False
        if (
            hasattr(server_agent_config.server_configs, "comm_configs")
            and hasattr(
                server_agent_config.server_configs.comm_configs, "proxystore_configs"
            )
            and server_agent_config.server_configs.comm_configs.proxystore_configs.get(
                "enable_proxystore", False
            )
        ):
            self.use_proxystore = True
            self.proxystore = Store(
                name="server-proxystore",
                connector=get_proxystore_connector(
                    server_agent_config.server_configs.comm_configs.proxystore_configs.connector_type,
                    server_agent_config.server_configs.comm_configs.proxystore_configs.connector_configs,
                ),
            )
            self.logger.info(
                f"Server using proxystore for model transfer with store: {server_agent_config.server_configs.comm_configs.proxystore_configs.connector_type}."
            )

    def _parse_result(self, result):
        """
        Parse the returned results from the client.
        The results can be composed of two parts:
        - Model parameters (can be model, gradients, compressed model, etc.)
        - Metadata (may contain additional information such as logs, etc.)
        :param `result`: The result returned from the client.
        :return `model`: The model parameters returned from the client
        :return `metadata`: The metadata returned from the client
        """
        if isinstance(result, tuple):
            model, metadata = result
        else:
            model, metadata = result, {}
        # Download model from S3 bucket or ProxyStore if necessary
        if isinstance(model, Proxy):
            model = extract(model)
        if self.use_s3bucket:
            if CloudStorage.is_cloud_storage_object(model):
                model = CloudStorage.download_object(
                    model, delete_cloud=True, delete_local=True
                )
        return model, metadata

    def _sanity_check(self):
        # Sanity check for number of clients
        num_clients = (
            self.server_agent_config.server_configs.num_clients
            if hasattr(self.server_agent_config.server_configs, "num_clients")
            else self.server_agent_config.server_configs.scheduler_kwargs.num_clients
            if (
                hasattr(self.server_agent_config.server_configs, "scheduler_kwargs")
                and hasattr(
                    self.server_agent_config.server_configs.scheduler_kwargs,
                    "num_clients",
                )
            )
            else self.server_agent_config.server_configs.aggregator_kwargs.num_clients
        )
        assert num_clients == len(self.client_agent_configs), (
            "Number of clients in the server configuration does not match the number of client configurations."
        )

    def _check_deprecation(
        self,
        client_id: str,
        client_metadata: Dict,
    ):
        """
        This function is used to check deprecation on the client site packages.
        """
        if not hasattr(self, "_version_deprecation_warning_set"):
            self._version_deprecation_warning_set = set()
        if "_deprecated" in client_metadata:
            if client_id not in self._version_deprecation_warning_set:
                self.logger.warning(
                    f"{client_id} is using a deprecated version of appfl, and it is highly recommended to update it to at least version 1.2.1."
                )
                self._version_deprecation_warning_set.add(client_id)
            client_metadata.pop("_deprecated")
        return client_metadata
