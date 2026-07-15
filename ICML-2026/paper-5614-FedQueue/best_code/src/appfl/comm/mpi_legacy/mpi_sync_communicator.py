import io
import math
import torch
from typing import Optional
from collections import OrderedDict
from appfl.compressor import Compressor
from appfl.misc.deprecation import deprecated


@deprecated(
    "MpiCommunicator is deprecated and will be removed in the future, please use appfl.comm.mpi.MPIServerCommunicator and appfl.comm.mpi.MPIClientCommunicator instead."
)
class MpiSyncCommunicator:
    """An MPI communicator specifically designed for synchronous federated learning experiments
    on multiple MPI processes, where each process can represent MORE THAN ONE federated learning
    clients by having those clients running serially on each MPI process."""

    def __init__(self, comm, compressor: Optional[Compressor] = None):
        self.comm = comm
        self.compressor = compressor
        self.comm_rank = comm.Get_rank()
        self.comm_size = comm.Get_size()
        self.counts = []
        self.max_slice_count = -1
        self.recv_limit = math.floor(
            (pow(2, 31) - (8 * self.comm_size) - 9) / (self.comm_size - 1)
        )

    def scatter(self, contents, source):
        """Scattering the contents to all clients from the source."""
        if source == self.comm_rank:
            assert len(contents) == self.comm_size, (
                "The size of the contents is not equal to the number of clients in scatter!"
            )
        return self.comm.scatter(contents, root=source)

    def gather(self, content, dest):
        """Gathering contents from all clients to the destination."""
        return self.comm.gather(content, root=dest)

    def broadcast_global_model(self, model=None, args=None):
        """Broadcast the global model state dict and additional arguments from FL server to FL clients."""
        assert model is not None or args is not None, "Nothing to send to clients!"
        if model is None:
            self.comm.bcast((args, False), root=self.comm_rank)
        else:
            self.comm.bcast((args, True), root=self.comm_rank)
            self.comm.bcast(model, root=self.comm_rank)

    def recv_global_model_from_server(self, source):
        args, has_model = self.comm.bcast(None, root=source)
        if has_model:
            model = self.comm.bcast(None, root=source)
        else:
            model = None
        return model if args is None else (model, args)

    def recv_all_local_models_from_clients(self, num_clients, model_copy=None) -> list:
        if self.max_slice_count < 0 or self.compressor is not None:
            self.counts = self.comm.gather(0, root=self.comm_rank)
            self.max_slice_count = max(self.counts)
            self.comm.bcast(self.max_slice_count)
        recvs = {}
        for rank in range(0, self.comm_size):
            recvs[rank] = b""
        for n in range(self.max_slice_count):
            recv = self.comm.gather(None, root=self.comm_rank)
            for r in range(0, self.comm_size):
                if r != self.comm_rank and n < self.counts[r]:
                    recvs[r] = b"".join([recvs[r], recv[r]])
        local_models = [None for _ in range(num_clients)]
        for r in range(0, self.comm_size):
            if r != self.comm_rank:
                if self.compressor is not None:
                    local_model_dict = self.compressor.decompress_model(
                        recvs[r], model_copy, batched=True
                    )
                else:
                    buffer = io.BytesIO(recvs[r])
                    local_model_dict = torch.load(buffer)
                for cid, model in local_model_dict.items():
                    local_models[cid] = model
        return local_models

    def send_local_models_to_server(self, models: OrderedDict, dest):
        if self.compressor is not None:
            serialized_local_models_val, _ = self.compressor.compress_model(
                models, batched=True
            )
        else:
            serialized_local_models = io.BytesIO()
            torch.save(models, serialized_local_models)
            serialized_local_models_val = serialized_local_models.getvalue()

        length = len(serialized_local_models_val)
        count = math.ceil(length / self.recv_limit)
        if self.max_slice_count < 0 or self.compressor is not None:
            self.comm.gather(count, root=dest)
            self.max_slice_count = self.comm.bcast(None, root=dest)

        for n in range(self.max_slice_count):
            if n < count:
                start_idx = n * self.recv_limit
                if (n + 1) * self.recv_limit < length:
                    end_idx = (n + 1) * self.recv_limit
                else:
                    end_idx = length
                self.comm.gather(
                    serialized_local_models_val[start_idx:end_idx], root=dest
                )
            else:
                self.comm.gather(None, root=dest)
