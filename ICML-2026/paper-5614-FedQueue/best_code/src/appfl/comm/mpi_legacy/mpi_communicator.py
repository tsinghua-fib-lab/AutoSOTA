import pickle
import numpy as np
from mpi4py import MPI
from collections import OrderedDict
from appfl.compressor import Compressor
from typing import Any, Optional, Union
from appfl.misc.deprecation import deprecated


@deprecated(
    "MpiCommunicator is deprecated and will be removed in the future, please use appfl.comm.mpi.MPIServerCommunicator and appfl.comm.mpi.MPIClientCommunicator instead."
)
class MpiCommunicator:
    """
    A general MPI communicator for synchronous or asynchronous distributed/federated/decentralized
    learning experiments on multiple MPI processes, where each MPI process represents EXACTLY ONE learning client.
    """

    def __init__(self, comm: MPI.Intracomm, compressor: Optional[Compressor] = None):
        self.comm = comm
        self.comm_rank = comm.Get_rank()
        self.comm_size = comm.Get_size()
        self.dests = []
        self.recv_queue = []
        self.recv_queue_idx = []  # corresponding client index for each request in recv_queue
        self.compressor = compressor

    def _obj_to_bytes(self, obj: Any) -> bytes:
        """Convert an object to bytes."""
        return pickle.dumps(obj)

    def _bytes_to_obj(self, bytes_obj: bytes) -> Any:
        """Convert bytes to an object."""
        return pickle.loads(bytes_obj)

    def scatter(self, contents, source: int) -> Any:
        """
        Scattering the `contents` to all MPI processes from the `source`
        :param `contents`: a list/sequence of contents to be scattered to all MPI processes
        :param `source`: the rank of the source MPI process that scatters the contents
        :return `content`: the content received by the current MPI process
        """
        if source == self.comm_rank:
            assert len(contents) == self.comm_size, (
                "The size of the contents is not equal to the number of clients in scatter!"
            )
        content = self.comm.scatter(contents, root=source)
        return content

    def gather(self, content, dest: int):
        """
        Gathering contents from all MPI processes to the destination process
        :param `content`: the content to be gathered from all MPI processes
        :param `dest`: the rank of the destination MPI process that gathers the contents
        :return `contents`: a list of contents received by the destination MPI process
        """
        contents = self.comm.gather(content, root=dest)
        return contents

    def broadcast_global_model(
        self,
        model: Optional[Union[dict, OrderedDict]] = None,
        args: Optional[dict] = None,
    ):
        """
        Broadcast the global model state dictionary and additional arguments from the server MPI process
        to all other client processes. The method is ONLY called by the server MPI process.
        :param `model`: the global model state dictionary to be broadcasted
        :param `args`: additional arguments to be broadcasted
        """
        self.dests = (
            [i for i in range(self.comm_size) if i != self.comm_rank]
            if len(self.dests) == 0
            else self.dests
        )
        if model is None:
            assert args is not None, "Nothing to send to the client!"
            for i in self.dests:
                self.comm.send((0, args), dest=i, tag=i)
        else:
            model_bytes = self._obj_to_bytes(model)
            for i in self.dests:
                payload = (
                    (len(model_bytes), args) if args is not None else len(model_bytes)
                )
                self.comm.send(payload, dest=i, tag=i)
            for i in self.dests:
                self.comm.Send(
                    np.frombuffer(model_bytes, dtype=np.byte),
                    dest=i,
                    tag=i + self.comm_size,
                )
            self.recv_queue = [self.comm.irecv(source=i, tag=i) for i in self.dests]
            self.recv_queue_idx = [i for i in range(self.comm_size - 1)]

    def send_global_model_to_client(
        self,
        model: Optional[Union[dict, OrderedDict]] = None,
        args: Optional[dict] = None,
        client_idx: int = -1,
    ):
        """
        Send the global model state dict and additional arguments to a certain client
        :param `model`: the global model state dictionary to be sent
        :param `args`: additional arguments to be sent
        :param `client_idx`: the index of the destination client
        """
        assert client_idx >= 0 and client_idx < self.comm_size, (
            "Please provide a valid destination client index!"
        )

        self.dests = (
            [i for i in range(self.comm_size) if i != self.comm_rank]
            if len(self.dests) == 0
            else self.dests
        )
        if model is None:
            assert args is not None, "Nothing to send to the client!"
            self.comm.send(
                (0, args), dest=self.dests[client_idx], tag=self.dests[client_idx]
            )
        else:
            model_bytes = self._obj_to_bytes(model)
            payload = (len(model_bytes), args) if args is not None else len(model_bytes)
            self.comm.send(
                payload, dest=self.dests[client_idx], tag=self.dests[client_idx]
            )
            self.comm.Send(
                np.frombuffer(model_bytes, dtype=np.byte),
                dest=self.dests[client_idx],
                tag=self.dests[client_idx] + self.comm_size,
            )
            # print(f"Server sent the global model to client {client_idx}", flush=True)
            self.recv_queue.append(
                self.comm.irecv(
                    source=self.dests[client_idx], tag=self.dests[client_idx]
                ),
            )
            self.recv_queue_idx.append(client_idx)

    def send_local_model_to_server(self, model: Union[dict, OrderedDict], dest: int):
        """
        Client sends the local model state dict to the server (dest).
        :param `model`: the local model state dictionary to be sent
        :param `dest`: the rank of the destination MPI process (server)
        """
        if self.compressor is not None:
            model_bytes = self.compressor.compress_model(model)
        else:
            model_bytes = self._obj_to_bytes(model)
        self.comm.isend(len(model_bytes), dest=dest, tag=self.comm_rank)
        self.comm.Send(
            np.frombuffer(model_bytes, dtype=np.byte),
            dest=dest,
            tag=self.comm_rank + self.comm_size,
        )

    def recv_local_model_from_client(self, model_copy=None):
        """
        Server receives the local model state dict from one finishing client.
        :param `model_copy` [Optional]: a copy of the global model state dict ONLY used for decompression
        """
        while True:
            # print(f"Server waiting for clients...")
            queue_idx, model_size = MPI.Request.waitany(self.recv_queue)
            if queue_idx != MPI.UNDEFINED:
                model_bytes = np.zeros(int(model_size), dtype=np.byte)
                self.recv_queue.pop(queue_idx)
                client_idx = self.recv_queue_idx.pop(queue_idx)
                self.comm.Recv(
                    model_bytes,
                    source=self.dests[client_idx],
                    tag=self.dests[client_idx] + self.comm_size,
                )
                if self.compressor is not None:
                    model = self.compressor.decompress_model(model_bytes, model_copy)
                else:
                    model = self._bytes_to_obj(model_bytes.tobytes())
                # print(f"Server received model from client {client_idx}", flush=True)
                # print(f"Server has client queue: {self.recv_queue_idx}", flush=True)
                return client_idx, model

    def recv_global_model_from_server(self, source):
        """
        Client receives the global model state dict from the server (source).
        :param `source`: the rank of the source MPI process (server)
        :return (`model`, `args`): the global model state dictionary and optional arguments received by the client
        """
        meta_data = self.comm.recv(source=source, tag=self.comm_rank)
        if isinstance(meta_data, tuple):
            model_size, args = meta_data[0], meta_data[1]
        else:
            model_size, args = meta_data, None
        if model_size == 0:
            return None, args
        model_bytes = np.empty(model_size, dtype=np.byte)
        self.comm.Recv(model_bytes, source=source, tag=self.comm_rank + self.comm_size)
        model = self._bytes_to_obj(model_bytes.tobytes())
        return model if args is None else (model, args)

    def cleanup(self):
        """
        Clean up the MPI communicator by waiting for all pending requests.
        """
        while len(self.recv_queue) > 0:
            queue_idx, model_size = MPI.Request.waitany(self.recv_queue)
            if queue_idx != MPI.UNDEFINED:
                model_bytes = np.zeros(int(model_size), dtype=np.byte)
                self.recv_queue.pop(queue_idx)
                client_idx = self.recv_queue_idx.pop(queue_idx)
                self.comm.Recv(
                    model_bytes,
                    source=self.dests[client_idx],
                    tag=self.dests[client_idx] + self.comm_size,
                )
