import time
import torch
from omegaconf import OmegaConf
from appfl.config import Config
from models.cnn import CNN
import appfl.run_grpc_server as grpc_server
import appfl.run_grpc_client as grpc_client
from mpi4py import MPI
from dataloader.femnist_dataloader import get_femnist

DataSet_name = "FEMNIST"
num_clients = 203
num_channel = 1  # 1 if gray, 3 if color
num_classes = 62  # number of the image classes
num_pixel = 28  # image size = (num_pixel, num_pixel)


def get_model(comm: MPI.Comm):
    ## User-defined model
    model = CNN(num_channel, num_classes, num_pixel)
    return model


def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    start_time = time.time()
    train_datasets, test_dataset = get_femnist(num_pixel, num_channel, 0)
    model = get_model(comm)
    print(
        "----------Loaded Datasets and Model----------Elapsed Time=",
        time.time() - start_time,
    )

    # read default configuration
    cfg = OmegaConf.structured(Config)
    loss_fn = torch.nn.CrossEntropyLoss()
    if comm_size > 1:
        # Try to launch both a server and clients.
        if comm_rank == 0:
            grpc_server.run_server(cfg, model, loss_fn, num_clients, test_dataset)
        else:
            # Give server some time to launch.
            time.sleep(5)
            grpc_client.run_client(
                cfg,
                comm_rank - 1,
                model,
                loss_fn,
                train_datasets[comm_rank - 1],
                comm_rank,
            )
        print("------DONE------", comm_rank)
    else:
        # Just launch a server.
        grpc_server.run_server(cfg, model, loss_fn, num_clients, test_dataset)


if __name__ == "__main__":
    main()
