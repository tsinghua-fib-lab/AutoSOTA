"""
mpiexec -np 7 python flamby_mpi_async.py --num_epochs 30 --dataset TcgaBrca --num_local_steps 100 --server ServerFedAsynchronous --val_range 1
mpiexec -np 7 python flamby_mpi_async.py --num_epochs 30 --dataset TcgaBrca --num_local_steps 100 --server ServerFedCompass --val_range 1
mpiexec -np 5 python flamby_mpi_async.py --num_epochs 30 --dataset HeartDisease --num_local_steps 100 --server ServerFedAsynchronous --val_range 1
mpiexec -np 5 python flamby_mpi_async.py --num_epochs 30 --dataset HeartDisease --num_local_steps 100 --server ServerFedCompass --val_range 1
mpiexec -np 4 python flamby_mpi_async.py --num_epochs 10 --dataset IXI --num_local_steps 100 --server ServerFedAsynchronous --val_range 2
mpiexec -np 7 python flamby_mpi_async.py --num_epochs 10 --dataset ISIC2019 --num_local_steps 100 --server ServerFedAsynchronous --val_range 3
"""

import time
import torch
import argparse
import appfl.run_mpi_async as rma
import appfl.run_mpi_compass as rmc
from mpi4py import MPI
from omegaconf import OmegaConf
from appfl.config import Config
from appfl.config.fed import FedAsync
from appfl.misc.utils import set_seed
from models.flamby import flamby_train
from dataloader.flamby_dataloader import get_flamby


## Read arguments
parser = argparse.ArgumentParser()

## device
parser.add_argument("--device", type=str, default="cpu")

## dataset
parser.add_argument(
    "--dataset",
    type=str,
    default="TcgaBrca",
    choices=["TcgaBrca", "HeartDisease", "IXI", "ISIC2019", "Kits19"],
)

## clients
parser.add_argument(
    "--local_train_pattern",
    type=str,
    default="steps",
    choices=["steps", "epochs"],
    help="For local optimizer, what counter to use, number of steps or number of epochs",
)
parser.add_argument("--num_local_steps", type=int, default=100)
parser.add_argument("--num_local_epochs", type=int, default=1)

## server
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--server_lr", type=float, default=0.01)
parser.add_argument("--mparam_1", type=float, default=0.9)
parser.add_argument("--mparam_2", type=float, default=0.99)
parser.add_argument("--adapt_param", type=float, default=0.001)
parser.add_argument(
    "--server",
    type=str,
    default="ServerFedAsynchronous",
    choices=[
        "ServerFedAsynchronous",
        "ServerFedBuffer",
        "ServerFedCompass",
        "ServerFedCompassMom",
    ],
)

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

## Fed Async
parser.add_argument(
    "--gradient_based",
    type=str,
    choices=["True", "true", "False", "false"],
    default="True",
    help="Whether the algorithm requires gradient from the model",
)
parser.add_argument(
    "--alpha", type=float, default=0.9, help="Mixing parameter for FedAsync Algorithm"
)
parser.add_argument(
    "--staleness_func",
    type=str,
    choices=["constant", "polynomial", "hinge"],
    default="polynomial",
)
parser.add_argument(
    "--a", type=float, default=0.5, help="First parameter for the staleness function"
)
parser.add_argument(
    "--b", type=int, default=4, help="Second parameter for Hinge staleness function"
)
parser.add_argument(
    "--K", type=int, default=3, help="Buffer size for FedBuffer algorithm"
)
parser.add_argument(
    "--val_range",
    type=int,
    default=10,
    help="Perform server validation every several epochs",
)

## Compass scheduler setups
parser.add_argument("--use_scheduler", action="store_true")
parser.add_argument("--q_ratio", type=float, default=0.2)
parser.add_argument("--lambda_val", type=float, default=1.5)

args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"


## Run
def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    assert comm_size > 1, (
        "This script requires the toal number of processes to be greater than one!"
    )
    args.num_clients = comm_size - 1

    ## Configuration
    cfg = OmegaConf.structured(Config(fed=FedAsync()))
    cfg.device = args.device
    cfg.reproduce = True
    if cfg.reproduce:
        set_seed(1)

    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.clientname = (
        "ClientOptim" if args.local_train_pattern == "epochs" else "ClientStepOptim"
    )
    cfg.fed.args.num_local_steps = args.num_local_steps
    cfg.fed.args.num_local_epochs = args.num_local_epochs

    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs

    ## Specific configuration for datasets in FLamby
    train_datasets, test_dataset = get_flamby(args.dataset, args.num_clients)
    (
        model,
        loss_fn,
        cfg.fed.args.optim,
        cfg.fed.args.optim_args.lr,
        cfg.train_data_batch_size,
        metric,
    ) = flamby_train(args.dataset)
    cfg.test_data_batch_size = cfg.train_data_batch_size
    cfg.train_data_shuffle = True

    ## outputs
    cfg.use_tensorboard = False
    cfg.save_model_state_dict = False
    cfg.output_dirname = "./outputs_Flamby_{}_{}clients_{}_{}epochs_mpi".format(
        args.dataset,
        args.num_clients,
        args.server,
        args.num_epochs,
    )
    cfg.output_filename = "result"

    ## adaptive server
    cfg.fed.args.server_learning_rate = args.server_lr
    cfg.fed.args.server_adapt_param = args.adapt_param
    cfg.fed.args.server_momentum_param_1 = args.mparam_1
    cfg.fed.args.server_momentum_param_2 = args.mparam_2

    ## privacy preserving
    cfg.fed.args.use_dp = args.use_dp
    cfg.fed.args.epsilon = args.epsilon
    cfg.fed.args.clip_grad = args.clip_grad
    cfg.fed.args.clip_value = args.clip_value
    cfg.fed.args.clip_norm = args.clip_norm

    ## fed async/fed buffer
    cfg.fed.args.K = args.K
    cfg.fed.args.alpha = args.alpha
    cfg.fed.args.gradient_based = args.gradient_based.lower() == "true"
    cfg.fed.args.staleness_func.name = args.staleness_func
    cfg.fed.args.staleness_func.args.a = args.a
    cfg.fed.args.staleness_func.args.b = args.b
    cfg.fed.args.val_range = args.val_range
    cfg.fed.args.q_ratio = args.q_ratio
    cfg.fed.args.lambda_val = args.lambda_val
    cfg.fed.args.val_range = args.val_range

    start_time = time.time()

    print("-------Loading_Time=", time.time() - start_time)

    ## Running
    use_scheduler = args.use_scheduler or args.server.startswith("ServerFedCompass")
    if comm_rank == 0:
        if use_scheduler:
            rmc.run_server(
                cfg,
                comm,
                model,
                loss_fn,
                args.num_clients,
                test_dataset,
                args.dataset,
                metric,
            )
        else:
            rma.run_server(
                cfg,
                comm,
                model,
                loss_fn,
                args.num_clients,
                test_dataset,
                args.dataset,
                metric,
            )
    else:
        assert comm_size == args.num_clients + 1
        if use_scheduler:
            rmc.run_client(
                cfg, comm, model, loss_fn, train_datasets, test_dataset, metric
            )
        else:
            rma.run_client(
                cfg, comm, model, loss_fn, train_datasets, test_dataset, metric
            )
    print("------DONE------", comm_rank)


if __name__ == "__main__":
    main()
