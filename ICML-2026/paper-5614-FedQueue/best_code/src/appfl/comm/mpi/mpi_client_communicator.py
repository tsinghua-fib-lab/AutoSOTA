import gc
import yaml
import torch
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Union, Tuple, OrderedDict, Optional, List
from .config import MPITaskRequest, MPITaskResponse, MPIServerStatus, MPITask
from .serializer import request_to_byte, byte_to_response, byte_to_model, model_to_byte
from appfl.misc.memory_utils import optimize_memory_cleanup, memory_efficient_model_io


class MPIClientCommunicator:
    """
    MPI client communicator for federated learning.

    :param comm: the MPI communicator from mpi4py
    :param server_rank: the rank of the server in the MPI communicator
    :param client_id: [optional] an optional client ID for one client for logging purposes, mutually exclusive with client_ids
    :param client_ids: [optional] a list of client IDs for a batched clients,
        this is only required when the MPI process represents multiple clients
    """

    def __init__(
        self,
        comm,
        server_rank: int,
        client_id: Optional[Union[str, int]] = None,
        client_ids: Optional[List[Union[str, int]]] = None,
        optimize_memory: bool = True,
        **kwargs,
    ):
        self.comm = comm
        self.comm_rank = comm.Get_rank()
        self.comm_size = comm.Get_size()
        self.server_rank = server_rank
        # Check for optimize_memory in kwargs or use parameter
        self.optimize_memory = (
            kwargs.get("optimize_memory", True)
            if "optimize_memory" in kwargs
            else optimize_memory
        )
        assert not (client_id is not None and client_ids is not None), (
            "client_id and client_ids are mutually exclusive. Use client_id for one client and client_ids for multiple clients."
        )
        if client_ids is not None:
            self.client_ids = [client_id for client_id in client_ids]
        elif client_id is not None:
            self.client_ids = [client_id]
        else:
            self.client_ids = [self.comm_rank]
        self._default_batching = client_ids is not None

    def get_configuration(self, **kwargs) -> DictConfig:
        """
        Get the federated learning configurations from the server.

        :param kwargs: additional metadata to be sent to the server
        :return: the federated learning configurations
        """
        kwargs["_client_ids"] = self.client_ids
        meta_data = yaml.dump(kwargs)
        request = MPITaskRequest(
            meta_data=meta_data,
        )
        request_bytes = request_to_byte(request)
        tag = self.comm_rank + self.comm_size * MPITask.GET_CONFIGURATION.value
        self.comm.Send(request_bytes, dest=self.server_rank, tag=tag)
        response = self._recv_response()
        if response.status == MPIServerStatus.ERROR.value:
            raise Exception("Server returned an error, stopping the client.")
        configuration = OmegaConf.create(response.meta_data)
        return configuration

    def get_global_model(
        self, **kwargs
    ) -> Union[Union[Dict, OrderedDict], Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Get the global model from the server.

        :param kwargs: additional metadata to be sent to the server
        :return: the global model with additional metadata (if any)
        """
        kwargs["_client_ids"] = self.client_ids
        meta_data = yaml.dump(kwargs)
        request = MPITaskRequest(
            meta_data=meta_data,
        )
        request_bytes = request_to_byte(request)
        tag = self.comm_rank + self.comm_size * MPITask.GET_GLOBAL_MODEL.value
        self.comm.Send(request_bytes, dest=self.server_rank, tag=tag)

        # Memory optimization: Clean up request bytes immediately
        if self.optimize_memory:
            del request_bytes
            gc.collect()

        response = self._recv_response()
        if response.status == MPIServerStatus.ERROR.value:
            raise Exception("Server returned an error, stopping the client.")

        # Memory optimization: Use optimized deserialization
        if self.optimize_memory:
            model = self._byte_to_model_optimized(response.payload)
        else:
            model = byte_to_model(response.payload)

        meta_data = yaml.unsafe_load(response.meta_data)

        # Memory optimization: Clean up response immediately
        if self.optimize_memory:
            optimize_memory_cleanup(response, force_gc=True)

        if len(meta_data) == 0:
            return model
        else:
            return model, meta_data

    def update_global_model(
        self,
        local_model: Union[Dict, OrderedDict, bytes],
        client_id: Optional[Union[str, int]] = None,
        **kwargs,
    ) -> Tuple[Union[Dict, OrderedDict], Dict]:
        """
        Send local model(s) to the FL server for global update, and return the new global model.

        :param local_model: the local model to be sent to the server for global aggregation

            - `local_model` can be a single model if one MPI process has only one client or one MPI process
                has multiple clients but the user wants to send one model at a time
            - `local_model` can be a dictionary of multiple models as well if one MPI process has multiple clients
                and the user wants to send all models
        :param client_id (optional): the client ID for the local model. It is only required when the MPI process has multiple clients
            and the user only wants to send one model at a time.
        :param kwargs (optional): additional metadata to be sent to the server. When sending local models for multiple clients,
            use the client ID as the key and the metadata as the value, e.g.,

        ```
        update_global_model(
            local_model=...,
            kwargs = {
                client_id1: {key1: value1, key2: value2},
                client_id2: {key1: value1, key2: value2},
            }
        )
        ```
        :return model: the updated global model

            - Note: the global model is only one model even if multiple local models are sent, which means that
            the server should have synchronous aggregation. If asynchronous aggregation is needed, the user should
            pass the local models one by one.

        :return meta_data: additional metadata from the server. When updating local models for multiple clients, the response will
            be a dictionary with the client ID as the key and the response as the value, e.g.,
        ```
        {
            client_id1: {ret1: value1, ret2: value2},
            client_id2: {ret1: value1, ret2: value2},
        }
        ```
        """
        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        if isinstance(local_model, dict) or isinstance(local_model, OrderedDict):
            kwargs["_torch_serialized"] = True
        else:
            kwargs["_torch_serialized"] = False
        kwargs["_client_ids"] = self.client_ids if client_id is None else [client_id]
        meta_data = yaml.dump(kwargs)

        # Memory optimization: Use optimized serialization for models
        if not isinstance(local_model, bytes):
            if self.optimize_memory:
                payload = self._model_to_byte_optimized(local_model)
                # Clean up local model from memory after serialization
                del local_model
                gc.collect()
            else:
                payload = model_to_byte(local_model)
        else:
            payload = local_model

        request = MPITaskRequest(
            payload=payload,
            meta_data=meta_data,
        )
        request_bytes = request_to_byte(request)
        tag = self.comm_rank + self.comm_size * MPITask.UPDATE_GLOBAL_MODEL.value
        self.comm.Send(request_bytes, dest=self.server_rank, tag=tag)

        # Memory optimization: Clean up intermediate data
        if self.optimize_memory:
            optimize_memory_cleanup(payload, request, request_bytes, force_gc=True)

        response = self._recv_response()
        if response.status == MPIServerStatus.ERROR.value:
            raise Exception("Server returned an error, stopping the client.")

        # Memory optimization: Use optimized deserialization
        if self.optimize_memory:
            model = self._byte_to_model_optimized(response.payload)
        else:
            model = byte_to_model(response.payload)

        meta_data = yaml.unsafe_load(response.meta_data)

        # Memory optimization: Clean up response immediately
        if self.optimize_memory:
            optimize_memory_cleanup(response, force_gc=True)
        # post-process the results if the client has multiple clients
        status = "DONE" if response.status == MPIServerStatus.DONE.value else "RUNNING"
        if client_id is not None or (not self._default_batching):
            meta_data = meta_data[kwargs["_client_ids"][0]]
            meta_data["status"] = status
        else:
            for id in self.client_ids:
                if id in meta_data:
                    meta_data[id]["status"] = status
        return model, meta_data

    def invoke_custom_action(
        self, action: str, client_id: Optional[Union[str, int]] = None, **kwargs
    ) -> Dict:
        """
        Invoke a custom action on the server.

        :param action: the action to be invoked
        :param client_id (optional): the client ID for the action. It is only required when the MPI process has multiple clients
            and the action is specific to a client instead of all clients.
        :param kwargs (optional): additional metadata to be sent to the server. When invoking custom action for multiple clients,
            use the client ID as the key and the metadata as the value, e.g.,
        ```
        invoke_custom_action(
            action=...,
            kwargs = {
                client_id1: {key1: value1, key2: value2},
                client_id2: {key1: value1, key2: value2},
            }
        )
        ```
        :return: the response from the server (if any). When invoking custom action for multiple clients, the response will
            be a dictionary with the client ID as the key and the response as the value, e.g.,
        ```
        {
            client_id1: {ret1: value1, ret2: value2},
            client_id2: {ret1: value1, ret2: value2},
        }
        ```
        """
        # Parse the kwargs if the user passes the kwargs as a dictionary
        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        kwargs["action"] = action
        kwargs["_client_ids"] = self.client_ids if client_id is None else [client_id]
        meta_data = yaml.dump(kwargs)
        request = MPITaskRequest(
            meta_data=meta_data,
        )
        request_bytes = request_to_byte(request)
        tag = self.comm_rank + self.comm_size * MPITask.INVOKE_CUSTOM_ACTION.value
        self.comm.Send(request_bytes, dest=self.server_rank, tag=tag)
        response = self._recv_response()
        if response.status == MPIServerStatus.ERROR.value:
            raise Exception("Server returned an error, stopping the client.")
        if len(response.meta_data) == 0:
            return {}
        else:
            try:
                results = yaml.unsafe_load(response.meta_data)
                # post-process the results if the client has multiple clients
                if client_id is not None or (not self._default_batching):
                    results = results[kwargs["_client_ids"][0]]
                return results
            except Exception as e:
                print(e)
                return {}

    def _recv_response(self) -> MPITaskResponse:
        status = MPI.Status()
        self.comm.probe(source=self.server_rank, tag=self.comm_rank, status=status)
        count = status.Get_count(MPI.BYTE)
        response_buffer = bytearray(count)
        self.comm.Recv(response_buffer, source=self.server_rank, tag=self.comm_rank)
        response = byte_to_response(response_buffer)

        # Memory optimization: Clean up buffer immediately
        if self.optimize_memory:
            del response_buffer
            gc.collect()

        return response

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
