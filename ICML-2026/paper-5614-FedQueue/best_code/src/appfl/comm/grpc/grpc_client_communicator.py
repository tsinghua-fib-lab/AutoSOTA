import gc
import io
import os
import uuid
import grpc
import time
import yaml
import torch
import pathlib
from datetime import datetime
from appfl.comm.utils.s3_storage import CloudStorage
from appfl.comm.utils.s3_utils import extract_model_from_s3, send_model_by_s3
from .grpc_communicator_pb2 import (
    ClientHeader,
    ConfigurationRequest,
    GetGlobalModelRequest,
    GetGlobalModelRespone,
    UpdateGlobalModelRequest,
    UpdateGlobalModelResponse,
    CustomActionRequest,
    CustomActionResponse,
    ServerStatus,
)
from .grpc_communicator_pb2_grpc import GRPCCommunicatorStub
from omegaconf import OmegaConf, DictConfig
from typing import Union, Dict, OrderedDict, Tuple, Optional, Any
from appfl.comm.grpc import (
    proto_to_databuffer,
    serialize_model,
    deserialize_model,
    create_grpc_channel,
)
from proxystore.store import Store
from proxystore.proxy import Proxy, extract
from appfl.misc.utils import deserialize_yaml, get_proxystore_connector
from appfl.misc.memory_utils import (
    efficient_bytearray_concatenation,
    optimize_memory_cleanup,
    get_state_dict_memory_info,
    split_state_dict_by_size,
)


class GRPCClientCommunicator:
    """
    gRPC communicator for federated learning clients.
    """

    def __init__(
        self,
        client_id: Union[str, int],
        *,
        server_uri: str,
        use_ssl: bool = False,
        use_authenticator: bool = False,
        root_certificate: Optional[Union[str, bytes]] = None,
        authenticator: Optional[str] = None,
        authenticator_args: Dict[str, Any] = {},
        max_message_size: int = 2 * 1024 * 1024,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        """
        Create a channel to the server and initialize the gRPC client stub.

        :param client_id: A unique client ID.
        :param server_uri: The URI of the server to connect to.
        :param use_ssl: Whether to use SSL/TLS to authenticate the server and encrypt communicated data.
        :param use_authenticator: Whether to use an authenticator to authenticate the client in each RPC. Must have `use_ssl=True` if `True`.
        :param root_certificate: The PEM-encoded root certificates as a byte string, or `None` to retrieve them from a default location chosen by gRPC runtime.
        :param authenticator: The name of the authenticator to use for authenticating the client in each RPC.
        :param authenticator_args: The arguments to pass to the authenticator.
        :param max_message_size: The maximum message size in bytes.
        """
        self.client_id = client_id
        self.logger = logger
        self.max_message_size = max_message_size
        # Check for optimize_memory in kwargs (from grpc_configs), default to True
        self.optimize_memory = kwargs.get("optimize_memory", True)

        # Streamed aggregation configuration (works with any transport mechanism)
        self.use_model_chunking = kwargs.get("use_model_chunking", False)
        self.model_chunk_size = kwargs.get(
            "model_chunk_size", 1 * 1024 * 1024 * 1024
        )  # 1GB default

        channel = create_grpc_channel(
            server_uri,
            use_ssl=use_ssl,
            use_authenticator=use_authenticator,
            root_certificate=root_certificate,
            authenticator=authenticator,
            authenticator_args=authenticator_args,
            max_message_size=max_message_size,
        )
        grpc.channel_ready_future(channel).result(timeout=3600)
        self.stub = GRPCCommunicatorStub(channel)
        self._use_authenticator = use_authenticator
        self.kwargs = kwargs
        self._load_proxystore()
        self._load_google_drive()
        self._s3_initalized = False

    def get_configuration(self, **kwargs) -> DictConfig:
        """
        Get the federated learning configurations from the server.
        :param kwargs: additional metadata to be sent to the server
        :return: the federated learning configurations
        """
        if "_client_id" in kwargs:
            client_id = str(kwargs["_client_id"])
            del kwargs["_client_id"]
        else:
            client_id = str(self.client_id)
        meta_data = yaml.dump(kwargs)
        request = ConfigurationRequest(
            header=ClientHeader(client_id=client_id),
            meta_data=meta_data,
        )
        response = self.stub.GetConfiguration(request, timeout=3600)
        if response.header.status == ServerStatus.ERROR:
            raise Exception("Server returned an error, stopping the client.")
        configuration = OmegaConf.create(response.configuration)
        # initializing s3 here as we need experiment id so that we can keep track of the models
        self._check_and_initialize_s3(
            experiment_id=configuration.get("experiment_id", None)
        )
        return configuration

    def get_global_model(
        self, **kwargs
    ) -> Union[Union[Dict, OrderedDict], Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Get the global model from the server.
        :param kwargs: additional metadata to be sent to the server
        :return: the global model with additional metadata (if any)
        """
        self._check_and_initialize_s3()
        if "_client_id" in kwargs:
            client_id = str(kwargs["_client_id"])
            del kwargs["_client_id"]
        else:
            client_id = str(self.client_id)
        if self.use_s3bucket:
            local_model_key = (
                f"{self.experiment_id}_{str(uuid.uuid4())}_server_state_{client_id}"
            )
            local_model_url = CloudStorage.presign_upload_object(
                local_model_key, register_for_clean=True
            )
            kwargs["model_key"] = local_model_key
            kwargs["model_url"] = local_model_url
            kwargs["_use_s3"] = True

        # Request chunked model if chunking is enabled
        if self.use_model_chunking:
            kwargs["_request_chunked"] = True

        meta_data = yaml.dump(kwargs)
        request = GetGlobalModelRequest(
            header=ClientHeader(client_id=client_id),
            meta_data=meta_data,
        )
        # Use optimized protocol when optimize_memory is enabled
        if self.optimize_memory:
            # New protocol: metadata and model data are separate
            response, model_bytes = self._receive_metadata_and_model_optimized(
                self.stub.GetGlobalModel(request, timeout=3600), GetGlobalModelRespone
            )

            if response.header.status == ServerStatus.ERROR:
                raise Exception("Server returned an error, stopping the client.")

            # Load model directly from bytes
            model = self._deserialize_model_optimized(model_bytes)
            del model_bytes
            gc.collect()
        else:
            # Original protocol: model embedded in protobuf
            data_chunks = []
            chunk_count = 0
            for byte_chunk in self.stub.GetGlobalModel(request, timeout=3600):
                data_chunks.append(byte_chunk.data_bytes)
                chunk_count += 1

            # Efficiently concatenate and parse
            byte_received = efficient_bytearray_concatenation(
                data_chunks, optimize_memory=False
            )
            response = GetGlobalModelRespone()
            response.ParseFromString(byte_received)

            optimize_memory_cleanup(data_chunks, byte_received, force_gc=True)
            del data_chunks, byte_received

            if response.header.status == ServerStatus.ERROR:
                raise Exception("Server returned an error, stopping the client.")

            model = deserialize_model(response.global_model)

        if isinstance(model, Proxy):
            model = extract(model)
        if isinstance(model, dict) and "model_drive_path" in model.keys():
            model = self.colab_connector.load_model(model["model_drive_path"])
        if self.use_s3bucket:
            model = extract_model_from_s3(client_id, self.experiment_id, "grpc", model)
        meta_data = deserialize_yaml(
            response.meta_data,
            trusted=self.kwargs.get("trusted", False) or self._use_authenticator,
            warning_message="Loading metadata fails due to untrusted data in the metadata, you can fix this by setting `trusted=True` in `grpc_configs` or use an authenticator.",
        )

        # Handle chunked response - call helper to collect all chunks
        if self.use_model_chunking and "_chunk_idx" in meta_data:
            return self._streamed_get_global_model(model, meta_data, **kwargs)

        if len(meta_data) == 0:
            return model
        else:
            return model, meta_data

    def update_global_model(
        self, local_model: Union[Dict, OrderedDict, bytes], **kwargs
    ) -> Tuple[Union[Dict, OrderedDict], Dict]:
        """
        Send local model to FL server for global update, and return the new global model.
        :param local_model: the local model to be sent to the server for global aggregation
        :param kwargs: additional metadata to be sent to the server
        :return: the updated global model with additional metadata. Specifically, `meta_data["status"]` is either "RUNNING" or "DONE".
        """
        self._check_and_initialize_s3()

        # Streamed aggregation: Check BEFORE wrapping in ProxyStore/S3/Colab
        # This ensures we chunk the raw model state_dict, not a Proxy or S3 reference
        if self.use_model_chunking and isinstance(local_model, (Dict, OrderedDict)):
            mem_info = get_state_dict_memory_info(local_model)
            if mem_info["total_bytes"] > self.model_chunk_size:
                if self.logger:
                    self.logger.info(
                        f"Model size {mem_info['total_mb']:.2f} MB exceeds chunk size "
                        f"{self.model_chunk_size / (1024**2):.2f} MB, using streamed aggregation"
                    )
                return self._streamed_aggregation(local_model, **kwargs)

        # Apply transport wrappers (each chunk will go through these individually)
        if self.use_proxystore:
            local_model = self.proxystore.proxy(local_model)
            kwargs["_use_proxystore"] = True
        if self.use_colab_connector:
            local_model = self.colab_connector.upload(
                local_model, f"local_model_epoch{int(time.time())}.pt"
            )
            kwargs["_use_colab_connector"] = True
        if "_client_id" in kwargs:
            client_id = str(kwargs["_client_id"])
            del kwargs["_client_id"]
        else:
            client_id = str(self.client_id)
        if self.use_s3bucket:
            local_model = send_model_by_s3(
                self.experiment_id, "grpc", local_model, client_id
            )
            local_model_key = (
                f"{self.experiment_id}_{str(uuid.uuid4())}_server_state_{client_id}"
            )
            local_model_url = CloudStorage.presign_upload_object(
                local_model_key, register_for_clean=True
            )
            kwargs["model_key"] = local_model_key
            kwargs["model_url"] = local_model_url
            kwargs["_use_s3"] = True

        meta_data = yaml.dump(kwargs)

        # Serialize the local model
        if isinstance(local_model, Proxy) or (not isinstance(local_model, bytes)):
            if self.optimize_memory:
                local_model_serialized = self._serialize_model_optimized(local_model)
            else:
                local_model_serialized = serialize_model(local_model)
            del local_model
            gc.collect()
        else:
            local_model_serialized = local_model

        # Use optimized protocol when optimize_memory is enabled
        if self.optimize_memory:
            # Create request with empty local_model (model will be sent separately)
            request = UpdateGlobalModelRequest(
                header=ClientHeader(client_id=client_id),
                local_model=b"",  # Will be sent separately
                meta_data=meta_data,
            )

            # Send request with model using optimized streaming
            request_generator = self._send_request_with_model_optimized(
                request, local_model_serialized, self.max_message_size
            )

            # Clear local model from memory after sending
            del local_model_serialized
            gc.collect()

            # Receive response using optimized protocol
            response, model_bytes = self._receive_metadata_and_model_optimized(
                self.stub.UpdateGlobalModel(request_generator, timeout=3600),
                UpdateGlobalModelResponse,
            )

            if response.header.status == ServerStatus.ERROR:
                raise Exception("Server returned an error, stopping the client.")

            # Load model directly from bytes (if not empty - empty means chunk acknowledgment)
            if len(model_bytes) > 0:
                model = self._deserialize_model_optimized(model_bytes)
                del model_bytes
                gc.collect()
            else:
                # Empty response (e.g., chunk acknowledgment)
                model = {}
                del model_bytes
        else:
            # Original protocol: Embed model in protobuf
            request = UpdateGlobalModelRequest(
                header=ClientHeader(client_id=client_id),
                local_model=local_model_serialized,
                meta_data=meta_data,
            )

            byte_received = b""
            for byte_chunk in self.stub.UpdateGlobalModel(
                proto_to_databuffer(request, max_message_size=self.max_message_size),
                timeout=3600,
            ):
                byte_received += byte_chunk.data_bytes
            response = UpdateGlobalModelResponse()
            response.ParseFromString(byte_received)

            if response.header.status == ServerStatus.ERROR:
                raise Exception("Server returned an error, stopping the client.")

            model = deserialize_model(response.global_model)

        if isinstance(model, Proxy):
            model = extract(model)
        if isinstance(model, dict) and "model_drive_path" in model.keys():
            model = self.colab_connector.load_model(model["model_drive_path"])
        if self.use_s3bucket:
            model = extract_model_from_s3(client_id, self.experiment_id, "grpc", model)
        meta_data = deserialize_yaml(
            response.meta_data,
            trusted=self.kwargs.get("trusted", False) or self._use_authenticator,
            warning_message="Loading metadata fails due to untrusted data in the metadata, you can fix this by setting `trusted=True` in `grpc_configs` or use an authenticator.",
        )
        meta_data["status"] = (
            "DONE" if response.header.status == ServerStatus.DONE else "RUNNING"
        )
        return model, meta_data

    def _streamed_get_global_model(
        self, first_chunk: Union[Dict, OrderedDict], first_meta: Dict, **kwargs
    ) -> Union[Union[Dict, OrderedDict], Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Streamed GetGlobalModel: receive model in chunks from server.
        Client explicitly requests each chunk by passing `_chunk_id`.
        Uses incremental merging to minimize memory usage.
        """
        total_chunks = first_meta["_total_chunks"]
        if self.logger:
            self.logger.info(
                f"GetGlobalModel: Receiving chunked model ({total_chunks} chunks total)"
            )

        # Incremental merging: start with first chunk instead of collecting all
        model = OrderedDict(first_chunk)
        del first_chunk
        gc.collect()

        if self.logger:
            self.logger.info(
                f"GetGlobalModel: Received and merged chunk [1/{total_chunks}]"
            )

        # Temporarily disable chunking to avoid recursion when requesting remaining chunks
        old_use_model_chunking = self.use_model_chunking
        self.use_model_chunking = False

        try:
            for chunk_id in range(1, total_chunks):
                chunk_kwargs = kwargs.copy()
                chunk_kwargs["_chunk_id"] = chunk_id
                result = self.get_global_model(**chunk_kwargs)
                if isinstance(result, tuple):
                    chunk_model, chunk_meta = result
                else:
                    chunk_model = result
                    chunk_meta = {}
                if "_chunk_idx" not in chunk_meta:
                    raise Exception(
                        f"Expected chunk metadata in response {chunk_id + 1}/{total_chunks}"
                    )

                # Incremental merge: update model immediately and free chunk memory
                model.update(chunk_model)
                del chunk_model
                gc.collect()

                if self.logger:
                    self.logger.info(
                        f"GetGlobalModel: Received and merged chunk [{chunk_meta['_chunk_idx'] + 1}/{total_chunks}]"
                    )
        finally:
            self.use_model_chunking = old_use_model_chunking

        if self.logger:
            self.logger.info(
                "GetGlobalModel: Completed incremental merge of all chunks"
            )

        # Clean chunk metadata from response
        for key in ["_chunk_idx", "_total_chunks", "_chunk_keys"]:
            first_meta.pop(key, None)

        # Return in same format as get_global_model
        if len(first_meta) == 0:
            return model
        else:
            return model, first_meta

    def _streamed_aggregation(
        self, local_model: Union[Dict, OrderedDict], **kwargs
    ) -> Tuple[Union[Dict, OrderedDict], Dict]:
        """
        Streamed aggregation: send model in chunks, aggregate each chunk separately.
        Works with any transport mechanism (S3, ProxyStore, optimized, regular).
        Uses incremental merging to minimize memory usage.
        """
        # Split model into chunks by size
        chunks = list(split_state_dict_by_size(local_model, self.model_chunk_size))
        total_chunks = len(chunks)

        # Delete local_model after splitting to free memory (chunks have copies)
        del local_model
        gc.collect()

        # Temporarily disable chunking to avoid recursion
        old_use_chunking = self.use_model_chunking
        self.use_model_chunking = False

        try:
            # Incremental merging: build global_model progressively
            global_model = OrderedDict()
            final_metadata = None

            # Loop through each chunk and aggregate
            for chunk_idx, chunk_dict, chunk_keys in chunks:
                if self.logger:
                    chunk_size_mb = sum(
                        t.numel() * t.element_size() for t in chunk_dict.values()
                    ) / (1024 * 1024)
                    self.logger.info(
                        f"Chunked aggregation: Sending chunk [{chunk_idx + 1}/{total_chunks}] "
                        f"({len(chunk_keys)} params, {chunk_size_mb:.2f} MB)"
                    )

                # Add chunk metadata for server-side streamed aggregation
                chunk_kwargs = kwargs.copy()
                chunk_kwargs["_chunk_idx"] = chunk_idx
                chunk_kwargs["_total_chunks"] = total_chunks
                chunk_kwargs["_chunk_keys"] = chunk_keys

                # Call existing update_global_model (uses whatever transport is configured)
                aggregated_chunk, chunk_metadata = self.update_global_model(
                    chunk_dict, **chunk_kwargs
                )

                # Free local chunk memory after sending
                del chunk_dict
                gc.collect()

                # Incremental merge: update global_model immediately and free aggregated chunk
                global_model.update(aggregated_chunk)
                final_metadata = chunk_metadata

                del aggregated_chunk
                gc.collect()

                if self.logger:
                    self.logger.info(
                        f"Chunked aggregation: Received and merged chunk [{chunk_idx + 1}/{total_chunks}]"
                    )

            if self.logger:
                self.logger.info(
                    "Chunked aggregation: Completed incremental merge of all chunks"
                )

            return global_model, final_metadata

        finally:
            # Restore chunking setting
            self.use_model_chunking = old_use_chunking

    def invoke_custom_action(self, action: str, **kwargs) -> Dict:
        """
        Invoke a custom action on the server.
        :param action: the action to be invoked
        :param kwargs: additional metadata to be sent to the server
        :return: the response from the server
        """
        if "_client_id" in kwargs:
            client_id = str(kwargs["_client_id"])
            del kwargs["_client_id"]
        else:
            client_id = str(self.client_id)
        meta_data = yaml.dump(kwargs)
        request = CustomActionRequest(
            header=ClientHeader(client_id=client_id),
            action=action,
            meta_data=meta_data,
        )
        byte_received = b""
        for byte in self.stub.InvokeCustomAction(
            proto_to_databuffer(request, max_message_size=self.max_message_size),
            timeout=3600,
        ):
            byte_received += byte.data_bytes
        response = CustomActionResponse()
        response.ParseFromString(byte_received)
        if response.header.status == ServerStatus.ERROR:
            raise Exception("Server returned an error, stopping the client.")

        if action == "close_connection":
            # Clean-up proxystore
            if hasattr(self, "proxystore") and self.proxystore is not None:
                try:
                    self.proxystore.close(clear=True)
                except:  # noqa: E722
                    self.proxystore.close()

            if self.colab_connector is not None:
                self.colab_connector.cleanup()

            if self.use_s3bucket:
                CloudStorage.clean_up()
                if self.logger is not None:
                    self.logger.info("S3 bucket cleaned up.")

        if len(response.results) == 0:
            return {}
        else:
            try:
                return deserialize_yaml(
                    response.results,
                    trusted=self.kwargs.get("trusted", False)
                    or self._use_authenticator,
                    warning_message="Loading metadata fails due to untrusted data in the metadata, you can fix this by setting `trusted=True` in `grpc_configs` or use an authenticator.",
                )
            except:  # noqa E722
                return {}

    def _load_proxystore(self):
        """
        Create the proxystore for storing and sending model parameters from the server to the clients.
        """
        self.proxystore = None
        self.use_proxystore = False

        if (
            "proxystore_configs" in self.kwargs
            and "enable_proxystore" in self.kwargs["proxystore_configs"]
            and self.kwargs["proxystore_configs"]["enable_proxystore"]
        ):
            self.use_proxystore = True
            self.proxystore = Store(
                name="server-proxystore",
                connector=get_proxystore_connector(
                    self.kwargs["proxystore_configs"]["connector_type"],
                    self.kwargs["proxystore_configs"]["connector_configs"],
                ),
            )

    def _load_google_drive(self) -> None:
        self.use_colab_connector = False
        self.colab_connector = None
        if (
            "colab_connector_configs" in self.kwargs
            and "enable" in self.kwargs["colab_connector_configs"]
            and self.kwargs["colab_connector_configs"]["enable"]
        ):
            from appfl.comm.utils.colab_connector import GoogleColabConnector

            self.use_colab_connector = True
            self.colab_connector = GoogleColabConnector(
                self.kwargs["colab_connector_configs"].get(
                    "model_path", "/content/drive/MyDrive/APPFL"
                )
            )

    def _check_and_initialize_s3(self, experiment_id=None):
        if self._s3_initalized:
            return
        self.experiment_id = (
            experiment_id
            if experiment_id is not None
            else datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )
        self._s3_initalized = True
        # check if s3 enable
        self.use_s3bucket = False
        s3_bucket = None
        if (
            "s3_configs" in self.kwargs
            and "enable_s3" in self.kwargs["s3_configs"]
            and self.kwargs["s3_configs"]["enable_s3"]
        ):
            self.use_s3bucket = True
            s3_bucket = self.kwargs["s3_configs"].get("s3_bucket", None)
            self.use_s3bucket = self.use_s3bucket and s3_bucket is not None
        if self.use_s3bucket:
            if self.logger is not None:
                self.logger.info(f"Using S3 bucket {s3_bucket} for model transfer.")
            s3_creds_file = self.kwargs["s3_configs"].get("s3_creds_file", None)
            s3_temp_dir_default = str(
                pathlib.Path.home() / ".appfl" / "grpc" / "server" / self.experiment_id
            )
            s3_temp_dir = self.kwargs["s3_configs"].get(
                "s3_temp_dir", s3_temp_dir_default
            )
            if not os.path.exists(s3_temp_dir):
                pathlib.Path(s3_temp_dir).mkdir(parents=True, exist_ok=True)
            CloudStorage.init(s3_bucket, s3_creds_file, s3_temp_dir, self.logger)

    def _serialize_model_optimized(self, model):
        """Memory-efficient model serialization."""
        with io.BytesIO() as buffer:
            torch.save(model, buffer)
            serialized_data = buffer.getvalue()
        gc.collect()
        return serialized_data

    def _deserialize_model_optimized(self, model_bytes):
        """Memory-efficient model deserialization."""
        with io.BytesIO(model_bytes) as buffer:
            model = torch.load(
                buffer, map_location="cpu"
            )  # Load to CPU first for memory efficiency
        gc.collect()
        return model

    def _proto_to_databuffer_optimized(self, proto, max_message_size=(2 * 1024 * 1024)):
        """Memory-optimized streaming with garbage collection."""
        from .grpc_communicator_pb2 import DataBuffer

        max_message_size = int(0.9 * max_message_size)
        data_bytes = proto.SerializeToString()
        data_bytes_size = len(data_bytes)
        message_size = min(max_message_size, data_bytes_size)

        chunk_count = 0
        for i in range(0, data_bytes_size, message_size):
            chunk = data_bytes[i : i + message_size]
            msg = DataBuffer(data_bytes=chunk)
            yield msg

            chunk_count += 1
            # Periodic garbage collection for large requests
            if chunk_count % 20 == 0:
                gc.collect()

        # Final cleanup
        del data_bytes
        gc.collect()

    def _send_request_with_model_optimized(
        self, request_proto, model_bytes, max_message_size
    ):
        """
        Send request using optimized protocol: metadata first, then raw model bytes.
        Yields DataBuffer chunks for streaming to server.
        """
        from .grpc_communicator_pb2 import DataBuffer

        # Step 1: Send metadata-only request (with empty local_model field)
        # Save original model data and replace with empty
        original_model = request_proto.local_model
        request_proto.local_model = b""  # Empty signals model data follows

        # Stream metadata protobuf
        yield from self._proto_to_databuffer_optimized(request_proto, max_message_size)

        # Restore original (for cleanup if needed)
        request_proto.local_model = original_model

        chunk_size = int(0.9 * max_message_size)
        chunk_count = 0
        for i in range(0, len(model_bytes), chunk_size):
            chunk = model_bytes[i : i + chunk_size]
            yield DataBuffer(data_bytes=chunk)

            chunk_count += 1
            if chunk_count % 20 == 0:
                gc.collect()
        gc.collect()

    def _receive_metadata_and_model_optimized(self, stream_iterator, response_type):
        """
        Receive metadata and model data using optimized protocol.
        Returns: (response_proto, model_bytes)
        - response_proto: The protobuf message with metadata (global_model will be empty)
        - model_bytes: Raw model bytes (to be loaded with torch.load)
        """
        # Collect all chunks first
        all_chunks = []
        chunk_count = 0
        total_bytes = 0

        for byte_chunk in stream_iterator:
            all_chunks.append(byte_chunk.data_bytes)
            total_bytes += len(byte_chunk.data_bytes)
            chunk_count += 1
            # Periodic garbage collection for large models
            if chunk_count % 20 == 0:
                gc.collect()

        # Parse metadata from first chunk(s)
        response = response_type()
        metadata_size = 0

        # Try to parse protobuf from progressively more chunks
        for i in range(
            1, min(len(all_chunks) + 1, 10)
        ):  # Metadata should be in first few chunks
            try:
                metadata_bytes = b"".join(all_chunks[:i])
                response.ParseFromString(metadata_bytes)
                metadata_size = len(metadata_bytes)
                break
            except Exception:
                self.logger.info(
                    f"Metadata parsing failed with first {i} chunks, trying more."
                )
                continue

        if metadata_size == 0:
            raise Exception("Failed to parse metadata from response")

        # Check if model data follows (indicated by empty global_model field)
        if response.global_model == b"":
            # Extract model bytes from remaining chunks
            model_buffer = io.BytesIO()
            bytes_consumed = 0

            for chunk in all_chunks:
                if bytes_consumed >= metadata_size:
                    # This chunk is pure model data
                    model_buffer.write(chunk)
                elif bytes_consumed + len(chunk) > metadata_size:
                    # This chunk is partially metadata, partially model
                    offset = metadata_size - bytes_consumed
                    model_buffer.write(chunk[offset:])
                bytes_consumed += len(chunk)

            # Cleanup
            del all_chunks
            gc.collect()

            model_buffer.seek(0)
            model_bytes = model_buffer.read()
            del model_buffer

            return response, model_bytes
        else:
            del all_chunks
            return response, response.global_model
