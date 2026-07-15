"""
To run MPI with 5 clients (FedAsync and FedCompass):
mpiexec -np 6 python ./mnist_mpi_async.py --partition class_noiid --loss_fn losses/celoss.py --loss_fn_name CELoss --num_epochs 20 --validation
mpiexec -np 6 python ./mnist_mpi_async.py --partition class_noiid --loss_fn losses/celoss.py --loss_fn_name CELoss --num_epochs 20 --server ServerFedCompass --validation

To run MPI with 5 clients with compression (FedAsync and FedCompass):
mpiexec -np 6 python ./mnist_mpi_async.py --partition class_noiid --loss_fn losses/celoss.py --loss_fn_name CELoss --num_epochs 20 --enable_compression --validation
mpiexec -np 6 python ./mnist_mpi_async.py --partition class_noiid --loss_fn losses/celoss.py --loss_fn_name CELoss --num_epochs 20 --server ServerFedCompass --enable_compression --validation
"""

import time
import torch
import argparse
from mpi4py import MPI
from omegaconf import OmegaConf
from appfl.config import Config
from appfl.config.fed import FedAsync
from appfl.misc.utils import set_seed
from appfl.misc.data import data_sanity_check
from losses.utils import get_loss
from models.utils import get_model
from metric.utils import get_metric
import appfl.run_mpi_async as rma
import appfl.run_mpi_compass as rmc
from dataloader.mnist_dataloader import get_mnist


## read arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")

## dataset
parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--num_channel", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--num_pixel", type=int, default=28)
parser.add_argument("--model", type=str, default="CNN")
parser.add_argument(
    "--partition",
    type=str,
    default="iid",
    choices=["iid", "class_noiid", "dirichlet_noiid"],
)
parser.add_argument("--seed", type=int, default=42)

## clients
parser.add_argument("--client_optimizer", type=str, default="Adam")
parser.add_argument("--client_lr", type=float, default=1e-3)
parser.add_argument(
    "--local_train_pattern",
    type=str,
    default="steps",
    choices=["steps", "epochs"],
    help="For local optimizer, what counter to use, number of steps or number of epochs",
)
parser.add_argument("--num_local_steps", type=int, default=100)
parser.add_argument("--num_local_epochs", type=int, default=1)

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

## server
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
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--server_lr", type=float, default=0.01)
parser.add_argument("--mparam_1", type=float, default=0.9)
parser.add_argument("--mparam_2", type=float, default=0.99)
parser.add_argument("--adapt_param", type=float, default=0.001)
parser.add_argument("--validation", action="store_true")

## loss function
parser.add_argument(
    "--loss_fn",
    type=str,
    required=False,
    help="path to the custom loss function definition file, use cross-entropy loss by default if no path is specified",
)
parser.add_argument(
    "--loss_fn_name",
    type=str,
    required=False,
    help="class name for the custom loss in the loss function definition file, choose the first class by default if no name is specified",
)

## evaluation metric
parser.add_argument(
    "--metric",
    type=str,
    default="metric/acc.py",
    help="path to the custom evaluation metric function definition file, use accuracy by default if no path is specified",
)
parser.add_argument(
    "--metric_name",
    type=str,
    required=False,
    help="function name for the custom eval metric function in the metric function definition file, choose the first function by default if no name is specified",
)

## asynchronous algorithm setups
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
    default=1,
    help="Perform server validation every several epochs",
)

## Compass scheduler setups
parser.add_argument("--use_scheduler", action="store_true")
parser.add_argument("--q_ratio", type=float, default=0.2)
parser.add_argument("--lambda_val", type=float, default=1.5)

## compression
parser.add_argument("--enable_compression", action="store_true")
parser.add_argument(
    "--lossy_compressor",
    type=str,
    default="SZ3",
    choices=["SZ3", "SZ2", "SZx", "ZFP"],
)
parser.add_argument(
    "--error_bounding_mode", type=str, default="REL", choices=["ABS", "REL"]
)
parser.add_argument(
    "--error_bound",
    type=float,
    default=1e-3,
    help="Error bound value for lossy compression",
)
parser.add_argument(
    "--lossless_compressor",
    type=str,
    default="blosc",
)

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
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_steps = args.num_local_steps
    cfg.fed.args.num_local_epochs = args.num_local_epochs
    cfg.train_data_shuffle = True

    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs
    cfg.validation = args.validation

    ## outputs
    cfg.use_tensorboard = False
    cfg.save_model_state_dict = False
    cfg.output_dirname = f"./outputs_{args.dataset}_{args.partition}_{args.num_clients}clients_{args.server}_{args.num_epochs}epochs_mpi"

    ## adaptive server
    cfg.fed.args.server_learning_rate = args.server_lr  # FedAdam
    cfg.fed.args.server_adapt_param = args.adapt_param  # FedAdam
    cfg.fed.args.server_momentum_param_1 = args.mparam_1  # FedAdam, FedAvgm
    cfg.fed.args.server_momentum_param_2 = args.mparam_2  # FedAdam

    ## compression
    cfg.enable_compression = args.enable_compression
    cfg.lossy_compressor = args.lossy_compressor
    cfg.error_bounding_mode = args.error_bounding_mode
    cfg.error_bound = args.error_bound
    cfg.lossless_compressor = args.lossless_compressor

    ## privacy preserving
    cfg.fed.args.use_dp = args.use_dp
    cfg.fed.args.epsilon = args.epsilon
    cfg.fed.args.clip_grad = args.clip_grad
    cfg.fed.args.clip_value = args.clip_value
    cfg.fed.args.clip_norm = args.clip_norm

    ## fed async/fed buffer/fed compass
    cfg.fed.args.K = args.K
    cfg.fed.args.alpha = args.alpha
    cfg.fed.args.gradient_based = args.gradient_based.lower() == "true"
    cfg.fed.args.staleness_func.name = args.staleness_func
    cfg.fed.args.staleness_func.args.a = args.a
    cfg.fed.args.staleness_func.args.b = args.b
    cfg.fed.args.q_ratio = args.q_ratio
    cfg.fed.args.lambda_val = args.lambda_val
    cfg.fed.args.val_range = args.val_range

    start_time = time.time()

    ## User-defined model
    model = get_model(args)
    loss_fn = get_loss(args.loss_fn, args.loss_fn_name)
    metric = get_metric(args.metric, args.metric_name)

    ## User-defined data
    train_datasets, test_dataset = get_mnist(
        comm,
        num_clients=cfg.num_clients,
        partition=args.partition,
        visualization=True,
        output_dirname=cfg.output_dirname,
        seed=args.seed,
        alpha1=args.num_clients,
    )

    ## Sanity check for the user-defined data
    if cfg.data_sanity:
        data_sanity_check(
            train_datasets, test_dataset, args.num_channel, args.num_pixel
        )

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
