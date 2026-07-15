import gc
import time
import yaml
import torch
import logging
import threading
import numpy as np
from mpi4py import MPI
from omegaconf import OmegaConf
from typing import Optional, Dict, OrderedDict
from concurrent.futures import Future
from appfl.agent import ServerAgent
from appfl.logger import ServerAgentFileLogger
from .config import MPITask, MPITaskRequest, MPITaskResponse, MPIServerStatus
from .serializer import byte_to_request, response_to_byte, model_to_byte, byte_to_model
from appfl.misc.memory_utils import optimize_memory_cleanup, memory_efficient_model_io


class MPIServerCommunicator:
    def __init__(
        self,
        comm,
        server_agent: ServerAgent,
        logger: Optional[ServerAgentFileLogger] = None,
        optimize_memory: bool = True,
        **kwargs,
    ) -> None:
        self.comm = comm
        self.comm_rank = comm.Get_rank()
        self.comm_size = comm.Get_size()
        self.server_agent = server_agent
        self.logger = logger if logger is not None else self._default_logger()
        # Check for optimize_memory in kwargs, server_agent configs, or use parameter
        self.optimize_memory = (
            kwargs.get("optimize_memory", True)
            if "optimize_memory" in kwargs
            else getattr(
                server_agent.server_agent_config.server_configs, "optimize_memory", True
            )
            if hasattr(
                server_agent.server_agent_config.server_configs, "optimize_memory"
            )
            else optimize_memory
        )
        self._get_global_model_futures: Dict[int, Future] = {}
        self._update_global_model_futures: Dict[int, Future] = {}
        self._sample_size_futures: Dict[int, Future] = {}
        self._client_id_to_client_rank = {}  # client_id to client_rank mapping
        self._benchmarking = self.server_agent.server_agent_config.server_configs.get(
            "benchmarking", False
        )
        if self._benchmarking:
            self._communication_times = []
            self._training_times = []

    def serve(self):
        """
        Start the MPI server to serve the clients.
        """
        self.logger.info("Server starting...")
        status = MPI.Status()
        while not self.server_agent.server_terminated():
            time.sleep(0.1)
            msg_flag = self.comm.iprobe(
                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
            )
            if msg_flag:
                source = status.Get_source()
                tag = status.Get_tag()
                count = status.Get_count(MPI.BYTE)
                request_buffer = bytearray(count)
                self.comm.Recv(request_buffer, source=source, tag=tag)
                request = byte_to_request(request_buffer)

                # Memory optimization: Clean up request buffer immediately
                if self.optimize_memory:
                    del request_buffer
                    gc.collect()

                response = self._request_handler(
                    client_rank=source, request_tag=tag, request=request
                )
                if response is not None:
                    response_bytes = response_to_byte(response)
                    self.comm.Send(response_bytes, dest=source, tag=source)

                    # Memory optimization: Clean up after sending
                    if self.optimize_memory:
                        optimize_memory_cleanup(response, response_bytes, force_gc=True)
        self.logger.info("Server terminated.")

    def _request_handler(
        self,
        client_rank: int,
        request_tag: int,
        request: MPITaskRequest,
    ) -> Optional[MPITaskResponse]:
        """
        Handle the request from the clients.
        :param `request`: the request from the clients
        :param `request_tag`: the tag of the request
        :return `response`: the response to the clients
        """
        request_type = request_tag // self.comm_size
        if request_type == MPITask.GET_CONFIGURATION.value:
            return self._get_configuration(client_rank, request)
        elif request_type == MPITask.GET_GLOBAL_MODEL.value:
            return self._get_global_model(client_rank, request)
        elif request_type == MPITask.UPDATE_GLOBAL_MODEL.value:
            return self._update_global_model(client_rank, request)
        elif request_type == MPITask.INVOKE_CUSTOM_ACTION.value:
            return self._invoke_custom_action(client_rank, request)
        else:
            raise ValueError(f"Invalid request tag: {request_tag}")

    def _get_configuration(
        self, client_rank: int, request: MPITaskRequest
    ) -> MPITaskResponse:
        """
        Client requests the FL configurations that are shared among all clients from the server.

        :param: `client_rank`: The rank of the client in MPI
        :param: `request.meta_data`: YAML serialized metadata dictionary (if needed)
        :return `response.status`: Server status
        :return `response.meta_data`: YAML serialized FL configurations
        """
        meta_data = (
            yaml.unsafe_load(request.meta_data) if len(request.meta_data) > 0 else {}
        )
        client_ids = meta_data.get("_client_ids", [client_rank])
        if len(client_ids) > 1:
            self.logger.info(
                f"Received GetConfiguration request from batched clients: {client_ids} [MPI rank {client_rank}]"
            )
        else:
            self.logger.info(
                f"Received GetConfiguration request from {client_ids[0]} [MPI rank {client_rank}]"
            )
        client_configs = self.server_agent.get_client_configs(**meta_data)
        client_configs = OmegaConf.to_container(client_configs, resolve=True)
        client_configs_serialized = yaml.dump(client_configs)
        response = MPITaskResponse(
            status=MPIServerStatus.RUN.value,
            meta_data=client_configs_serialized,
        )
        return response

    def _get_global_model(
        self, client_rank: int, request: MPITaskRequest
    ) -> Optional[MPITaskResponse]:
        """
        Return the global model to clients. This method is supposed to provide clients with
        the initial and final global model.

        :param: `client_rank`: The rank of the client(s) in MPI
        :param: `request.meta_data`: YAML serialized metadata dictionary (if needed)

             - `meta_data['_client_ids']`: A list of client ids to get the global model for batched clients
             - `meta_data['init_model']`: Whether to get the initial global model or not
        :return `response.status`: Server status
        :return `response.payload`: Serialized global model
        :return `response.meta_data`: YAML serialized metadata dictionary (if needed)
        """
        meta_data = (
            yaml.unsafe_load(request.meta_data) if len(request.meta_data) > 0 else {}
        )
        client_ids = meta_data.get("_client_ids", [client_rank])
        if len(client_ids) > 1:
            self.logger.info(
                f"Received GetGlobalModel request from batched clients: {client_ids} [MPI rank {client_rank}]"
            )
        else:
            self.logger.info(
                f"Received GetGlobalModel request from {client_ids[0]} [MPI rank {client_rank}]"
            )
        self._client_id_to_client_rank[client_rank] = client_rank
        meta_data["num_batched_clients"] = len(client_ids)
        model = self.server_agent.get_parameters(**meta_data, blocking=False)
        if not isinstance(model, Future):
            if isinstance(model, tuple):
                meta_data = model[1]
                model = model[0]
            else:
                meta_data = {}
            if self._benchmarking:
                meta_data["send_time"] = time.time()

            # Memory optimization: Use optimized serialization
            if self.optimize_memory:
                model_serialized = self._model_to_byte_optimized(model)
                # Clean up model from memory before creating response
                del model
                gc.collect()
            else:
                model_serialized = model_to_byte(model)

            return MPITaskResponse(
                status=MPIServerStatus.RUN.value,
                payload=model_serialized,
                meta_data=yaml.dump(meta_data),
            )
        else:
            self._get_global_model_futures[client_rank] = model
            self._check_get_global_model_futures()
            return None

    def _update_global_model(
        self, client_rank: int, request: MPITaskRequest
    ) -> Optional[MPITaskResponse]:
        """
        Update the global model with the local model from the client,
        and return the updated global model to the client.

        :param: `client_rank`: The rank of the client in MPI
        :param: `request.payload`: Serialized local model
        :param: `request.meta_data`: YAML serialized metadata dictionary (if needed)
        :return `response.status`: Server status
        :return `response.payload`: Serialized updated global model
        :return `response.meta_data`: YAML serialized metadata dictionary (if needed)
        """
        local_model = request.payload
        meta_data = (
            yaml.unsafe_load(request.meta_data) if len(request.meta_data) > 0 else {}
        )
        if meta_data.get("_torch_serialized", True):
            # Memory optimization: Use optimized deserialization
            if self.optimize_memory:
                local_model = self._byte_to_model_optimized(local_model)
            else:
                local_model = byte_to_model(local_model)
        # read the client ids from the metadata if any
        client_ids = meta_data.get("_client_ids", [client_rank])
        for client_id in client_ids:
            self._client_id_to_client_rank[client_id] = client_rank
        if len(client_ids) > 1:
            assert (
                self.server_agent.server_agent_config.server_configs.scheduler
                == "SyncScheduler"
            ), "Batched clients are only supported with SyncScheduler."
            self.logger.info(
                f"Received UpdateGlobalModel request from batched clients: {client_ids} [MPI rank {client_rank}]"
            )
        else:
            self.logger.info(
                f"Received UpdateGlobalModel request from {client_ids[0]} [MPI rank {client_rank}]"
            )
        for client_id in client_ids:
            client_metadata = (
                meta_data[client_id] if client_id in meta_data else meta_data
            )
            if self._benchmarking:
                self._communication_times.append(
                    client_metadata["communication_time"]
                    + time.time()
                    - client_metadata["send_time"]
                )
                self._training_times.append(client_metadata["training_time"])
                # compute the average communication and training times and standard deviation
                self.logger.info(
                    f"Communication time: {sum(self._communication_times) / len(self._communication_times):.4f} ± {np.std(self._communication_times):.4f} seconds"
                )
                self.logger.info(
                    f"Training time: {sum(self._training_times) / len(self._training_times):.4f} ± {np.std(self._training_times):.4f} seconds"
                )
            client_local_model = (
                local_model[client_id]
                if (
                    (
                        isinstance(local_model, dict)
                        or isinstance(local_model, OrderedDict)
                    )
                    and client_id in local_model
                )
                else local_model
            )
            global_model = self.server_agent.global_update(
                client_id, client_local_model, blocking=False, **client_metadata
            )

            # Memory optimization: Clean up client local model immediately after processing
            if self.optimize_memory:
                del client_local_model
                gc.collect()

            if not isinstance(global_model, Future):
                meta_data = {}
                if isinstance(global_model, tuple):
                    meta_data[client_id] = global_model[1]
                    global_model = global_model[0]
                else:
                    meta_data[client_id] = {}
                if self._benchmarking:
                    meta_data[client_id]["send_time"] = time.time()

                # Memory optimization: Use optimized serialization
                if self.optimize_memory:
                    global_model_serialized = self._model_to_byte_optimized(
                        global_model
                    )
                    # Clean up global model from memory before creating response
                    del global_model
                    gc.collect()
                else:
                    global_model_serialized = model_to_byte(global_model)

                status = (
                    MPIServerStatus.DONE.value
                    if self.server_agent.training_finished()
                    else MPIServerStatus.RUN.value
                )
                return MPITaskResponse(
                    status=status,
                    payload=global_model_serialized,
                    meta_data=yaml.dump(meta_data),
                )
            else:
                self._update_global_model_futures[client_id] = global_model
                self._check_update_global_model_futures()
        return None

    def _invoke_custom_action(
        self,
        client_rank: int,
        request: MPITaskRequest,
    ) -> Optional[MPITaskResponse]:
        """
        Invoke custom action on the server.
        :param: `client_rank`: The rank of the client in MPI
        :param: `request.meta_data`: YAML serialized metadata dictionary (if needed)
        :return `response.status`: Server status
        :return `response.meta_data`: YAML serialized metadata dictionary (if needed)
        """
        meta_data = (
            yaml.unsafe_load(request.meta_data) if len(request.meta_data) > 0 else {}
        )
        assert "action" in meta_data, "The action is not specified in the metadata"
        action = meta_data["action"]
        client_ids = meta_data.get("_client_ids", [client_rank])
        if len(client_ids) > 1:
            self.logger.info(
                f"Received InvokeCustomAction ({meta_data['action']}) request from batched clients: {client_ids} [MPI rank {client_rank}]"
            )
        else:
            self.logger.info(
                f"Received InvokeCustomAction ({meta_data['action']}) request from {client_ids[0]} [MPI rank {client_rank}]"
            )

        del meta_data["action"]
        if action == "set_sample_size":
            sync = True
            for client_id in client_ids:
                self._client_id_to_client_rank[client_id] = client_rank
                client_metadata = (
                    meta_data[client_id] if client_id in meta_data else meta_data
                )
                client_metadata["blocking"] = False
                ret_val = self.server_agent.set_sample_size(
                    client_id, **client_metadata
                )
                if ret_val is None:
                    sync = False
                else:
                    self._sample_size_futures[client_id] = ret_val
                    self._check_sample_size_future()
            return None if sync else MPITaskResponse(status=MPIServerStatus.RUN.value)
        elif action == "close_connection":
            for client_id in client_ids:
                self.server_agent.close_connection(client_id)
            return MPITaskResponse(status=MPIServerStatus.DONE.value)
        elif action == "get_data_readiness_report":
            num_clients = self.server_agent.get_num_clients()
            if not hasattr(self, "_dr_metrics_lock"):
                self._dr_metrics = {}
                self._dr_metrics_client_ids = set()
                self._dr_metrics_lock = threading.Lock()
            for client_id in client_ids:
                client_metadata = (
                    meta_data[client_id] if client_id in meta_data else meta_data
                )
                with self._dr_metrics_lock:
                    self._dr_metrics_client_ids.add(client_id)
                    for k, v in client_metadata.items():
                        if k not in self._dr_metrics:
                            self._dr_metrics[k] = {}
                        self._dr_metrics[k][client_id] = v
                    if len(self._dr_metrics_client_ids) == num_clients:
                        self.server_agent.data_readiness_report(self._dr_metrics)
                        response = MPITaskResponse(
                            status=MPIServerStatus.RUN.value,
                        )
                        response_bytes = response_to_byte(response)
                        responded_client_ranks = set()
                        for client_id in self._dr_metrics_client_ids:
                            client_rank = self._client_id_to_client_rank[client_id]
                            if client_rank not in responded_client_ranks:
                                responded_client_ranks.add(client_rank)
                                self.comm.Send(
                                    response_bytes, dest=client_rank, tag=client_rank
                                )
                        self._dr_metrics = {}
                        self._dr_metrics_client_ids = set()
            return None
        else:
            raise NotImplementedError(f"Custom action {action} is not implemented.")

    def _check_sample_size_future(self):
        """
        Return the updated relative sample size to the client if the `Future` object is available.
        """
        delete_keys = []
        responses = {}
        for client_id, future in self._sample_size_futures.items():
            if future.done():
                meta_data = future.result()
                client_rank = self._client_id_to_client_rank[client_id]
                if client_rank not in responses:
                    responses[client_rank] = {}
                responses[client_rank][client_id] = meta_data
                delete_keys.append(client_id)
        for client_rank, meta_data in responses.items():
            response = MPITaskResponse(
                status=MPIServerStatus.RUN.value,
                meta_data=yaml.dump(meta_data),
            )
            response_bytes = response_to_byte(response)
            self.comm.Send(response_bytes, dest=client_rank, tag=client_rank)
        for key in delete_keys:
            del self._sample_size_futures[key]

    def _check_get_global_model_futures(self):
        """
        Return the global model to the client if the global model `Future` object is available.
        """
        delete_keys = []
        status = (
            MPIServerStatus.DONE.value
            if self.server_agent.training_finished()
            else MPIServerStatus.RUN.value
        )
        for client_id, future in self._get_global_model_futures.items():
            if future.done():
                global_model = future.result()
                if isinstance(global_model, tuple):
                    meta_data = global_model[1]
                    global_model = global_model[0]
                else:
                    meta_data = {}
                if self._benchmarking:
                    meta_data["send_time"] = time.time()
                client_rank = self._client_id_to_client_rank[client_id]

                # Memory optimization: Use optimized serialization
                if self.optimize_memory:
                    global_model_serialized = self._model_to_byte_optimized(
                        global_model
                    )
                    # Clean up global model from memory before creating response
                    del global_model
                    gc.collect()
                else:
                    global_model_serialized = model_to_byte(global_model)

                response = MPITaskResponse(
                    status=status,
                    payload=global_model_serialized,
                    meta_data=yaml.dump(meta_data),
                )
                response_bytes = response_to_byte(response)
                self.comm.Send(response_bytes, dest=client_rank, tag=client_rank)

                # Memory optimization: Clean up after sending
                if self.optimize_memory:
                    optimize_memory_cleanup(
                        global_model_serialized, response, response_bytes, force_gc=True
                    )

                delete_keys.append(client_id)
        for key in delete_keys:
            del self._get_global_model_futures[key]

    def _check_update_global_model_futures(self):
        """
        Return the updated global model to the client if the global model `Future` object is available.
        """
        delete_keys = []
        model_responses = {}
        meta_data_responses = {}
        status = (
            MPIServerStatus.DONE.value
            if self.server_agent.training_finished()
            else MPIServerStatus.RUN.value
        )
        for client_id, future in self._update_global_model_futures.items():
            if future.done():
                global_model = future.result()
                if isinstance(global_model, tuple):
                    meta_data = global_model[1]
                    global_model = global_model[0]
                else:
                    meta_data = {}
                if self._benchmarking:
                    meta_data["send_time"] = time.time()
                client_rank = self._client_id_to_client_rank[client_id]
                if client_rank not in model_responses:
                    # Memory optimization: Use optimized serialization
                    if self.optimize_memory:
                        global_model_serialized = self._model_to_byte_optimized(
                            global_model
                        )
                        # Clean up global model from memory before creating response
                        del global_model
                        gc.collect()
                    else:
                        global_model_serialized = model_to_byte(global_model)
                    model_responses[client_rank] = global_model_serialized
                    meta_data_responses[client_rank] = {}
                meta_data_responses[client_rank][client_id] = meta_data
                delete_keys.append(client_id)
        for client_rank in model_responses:
            response = MPITaskResponse(
                status=status,
                payload=model_responses[client_rank],
                meta_data=yaml.dump(meta_data_responses[client_rank]),
            )
            response_bytes = response_to_byte(response)
            self.comm.Send(response_bytes, dest=client_rank, tag=client_rank)

            # Memory optimization: Clean up after sending
            if self.optimize_memory:
                optimize_memory_cleanup(response, response_bytes, force_gc=True)
        for key in delete_keys:
            del self._update_global_model_futures[key]

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

    def _model_to_byte_optimized(self, model):
        """
        Memory-optimized model serialization using context manager.
        """
        with memory_efficient_model_io(optimize_memory=self.optimize_memory) as buffer:
            torch.save(model, buffer)
            serialized_data = buffer.getvalue()
        return serialized_data

    def _byte_to_model_optimized(self, model_bytes: bytes):
        """
        Memory-optimized model deserialization with CPU loading and cleanup.
        """
        with memory_efficient_model_io(optimize_memory=self.optimize_memory) as buffer:
            buffer.write(model_bytes)
            buffer.seek(0)
            model = torch.load(buffer, map_location="cpu")
        gc.collect()
        return model
