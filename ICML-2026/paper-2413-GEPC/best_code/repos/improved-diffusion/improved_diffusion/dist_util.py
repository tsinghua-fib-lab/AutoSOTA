"""
Helpers for distributed training.
Patched for single-node use without MPI.
"""

import io
import os
import socket

import blobfile as bf
try:
    from mpi4py import MPI
    _HAS_MPI = True
except ImportError:
    _HAS_MPI = False
import torch as th
import torch.distributed as dist

GPUS_PER_NODE = 8
SETUP_RETRY_COUNT = 3


def setup_dist():
    if dist.is_initialized():
        return

    if _HAS_MPI:
        comm = MPI.COMM_WORLD
        backend = "gloo" if not th.cuda.is_available() else "nccl"
        if backend == "gloo":
            hostname = "localhost"
        else:
            hostname = socket.gethostbyname(socket.getfqdn())
        os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
        os.environ["RANK"] = str(comm.rank)
        os.environ["WORLD_SIZE"] = str(comm.size)
        port = comm.bcast(_find_free_port(), root=0)
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        pass


def dev():
    if th.cuda.is_available():
        if _HAS_MPI:
            return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
        return th.device("cuda:0")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    if _HAS_MPI and MPI.COMM_WORLD.size > 1:
        if MPI.COMM_WORLD.Get_rank() == 0:
            with bf.BlobFile(path, "rb") as f:
                data = f.read()
        else:
            data = None
        data = MPI.COMM_WORLD.bcast(data)
        return th.load(io.BytesIO(data), **kwargs)
    else:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    if not dist.is_initialized():
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
