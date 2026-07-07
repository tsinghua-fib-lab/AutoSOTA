import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP


def example(rank, world_size):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    # model = nn.Linear(10, 10).to(rank)
    model = nn.Linear(10, 10).to(0)
    # construct DDP model
    # ddp_model = DDP(model, device_ids=[rank])
    ddp_model = DDP(model, device_ids=[0])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    # outputs = ddp_model(torch.randn(20, 10).to(rank))
    outputs = ddp_model(torch.randn(20, 10).to(0))
    # labels = torch.randn(20, 10).to(rank)
    labels = torch.randn(20, 10).to(0)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()

def test_ddp():
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    world_size = 5
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)
