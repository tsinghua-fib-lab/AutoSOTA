import time
import torch
import argparse
import appfl.run_mpi as rm
import appfl.run_serial as rs
from mpi4py import MPI
from omegaconf import OmegaConf
from appfl.config import Config
from appfl.misc.data import data_sanity_check
from models.utils import get_model
from dataloader.celeba_dataloader import get_celeba

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cpu")

## dataset
parser.add_argument("--dataset", type=str, default="CELEBA")
parser.add_argument("--num_channel", type=int, default=3)
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--num_pixel", type=int, default=218)
parser.add_argument("--model", type=str, default="resnet18-legacy")

## clients
parser.add_argument("--num_clients", type=int, default=1)
parser.add_argument("--client_optimizer", type=str, default="Adam")
parser.add_argument("--client_lr", type=float, default=1e-3)
parser.add_argument("--num_local_epochs", type=int, default=1)

## server
parser.add_argument("--server", type=str, default="ServerFedAvg")
parser.add_argument("--num_epochs", type=int, default=2)

parser.add_argument("--server_lr", type=float, required=False)
parser.add_argument("--mparam_1", type=float, required=False)
parser.add_argument("--mparam_2", type=float, required=False)
parser.add_argument("--adapt_param", type=float, required=False)

parser.add_argument("--pretrained", type=int, default=0)

## privacy preserving
parser.add_argument(
    "--use_dp",
    action="store_true",
    default=False,
    help="Whether to enable differential privacy technique to preserve privacy",
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=1,
    help="Privacy budget - stronger privacy as epsilon decreases",
)
parser.add_argument(
    "--clip_grad",
    action="store_true",
    default=False,
    help="Whether to clip the gradients",
)
parser.add_argument(
    "--clip_value", type=float, default=1.0, help="Max norm of the gradients"
)
parser.add_argument(
    "--clip_norm",
    type=float,
    default=1,
    help="Type of the used p-norm for gradient clipping",
)

args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"


def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # read default configuration
    cfg = OmegaConf.structured(Config)

    ## Reproducibility
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    train_datasets, test_dataset = get_celeba(args.num_pixel)

    if cfg.data_sanity:
        data_sanity_check(
            train_datasets, test_dataset, args.num_channel, args.num_pixel
        )

    args.num_clients = len(train_datasets)

    model = get_model(args)
    loss_fn = torch.nn.CrossEntropyLoss()
    print(
        "----------Loaded Datasets and Model----------Elapsed Time=",
        time.time() - start_time,
    )

    cfg.device = args.device

    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_epochs = args.num_local_epochs

    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs

    ## privacy preserving
    cfg.fed.args.use_dp = args.use_dp
    cfg.fed.args.epsilon = args.epsilon
    cfg.fed.args.clip_grad = args.clip_grad
    cfg.fed.args.clip_value = args.clip_value
    cfg.fed.args.clip_norm = args.clip_norm

    ## outputs
    cfg.use_tensorboard = True

    if comm_size > 1:
        if comm_rank == 0:
            rm.run_server(
                cfg, comm, model, loss_fn, args.num_clients, test_dataset, args.dataset
            )
        else:
            rm.run_client(cfg, comm, model, loss_fn, args.num_clients, train_datasets)
        print("------DONE------", comm_rank)
    else:
        rs.run_serial(cfg, model, loss_fn, train_datasets, test_dataset, args.dataset)


if __name__ == "__main__":
    main()

# To run CUDA-aware MPI:
# mpiexec -np 2 --mca opal_cuda_support 1 python ./celeba.py
# To run MPI:
# mpiexec -np 2 python ./celeba.py
# To run:
# python ./celeba.py
# To run with resnet pretrained weight:
# python ./celeba.py --model resnet18 --pretrained 1
