"""
mpiexec -np 5 python ./coronahack_mpi.py --num_epochs 10
"""

import time
import torch
import argparse
import appfl.run_mpi as rm
import appfl.run_mpi_sync as rms
from mpi4py import MPI
from omegaconf import OmegaConf
from appfl.config import Config
from appfl.misc.utils import set_seed
from losses.utils import get_loss
from models.utils import get_model
from metric.utils import get_metric
from dataloader.coronahack_dataloader import get_corona


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")

## dataset
parser.add_argument("--dataset", type=str, default="Coronahack")
parser.add_argument(
    "--non_processed",
    action="store_true",
    default=False,
    help="Specify if NOT preprocessed the downloaded datasets",
)
parser.add_argument("--num_channel", type=int, default=3)
parser.add_argument("--num_classes", type=int, default=7)
parser.add_argument("--num_pixel", type=int, default=32)
parser.add_argument("--model", type=str, default="CNN")

## clients
parser.add_argument("--num_clients", type=int, default=4)
parser.add_argument("--client_optimizer", type=str, default="Adam")
parser.add_argument("--client_lr", type=float, default=3e-3)
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
parser.add_argument("--server", type=str, default="ServerFedAvg")
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--server_lr", type=float, default=0.1)
parser.add_argument("--mparam_1", type=float, default=0.9)
parser.add_argument("--mparam_2", type=float, default=0.99)
parser.add_argument("--adapt_param", type=float, default=0.001)

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

args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"


def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    ## Reproducibility
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    ## configuration
    cfg = OmegaConf.structured(Config)
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

    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs

    ## outputs
    cfg.use_tensorboard = False
    cfg.save_model_state_dict = False
    cfg.output_dirname = f"./outputs_{args.dataset}_{args.num_clients}clients_{args.server}_{args.num_epochs}epochs_mpi"

    ## adaptive server
    cfg.fed.args.server_learning_rate = args.server_lr  # FedAdam
    cfg.fed.args.server_adapt_param = args.adapt_param  # FedAdam
    cfg.fed.args.server_momentum_param_1 = args.mparam_1  # FedAdam, FedAvgm
    cfg.fed.args.server_momentum_param_2 = args.mparam_2  # FedAdam

    ## privacy preserving
    cfg.fed.args.use_dp = args.use_dp
    cfg.fed.args.epsilon = args.epsilon
    cfg.fed.args.clip_grad = args.clip_grad
    cfg.fed.args.clip_value = args.clip_value
    cfg.fed.args.clip_norm = args.clip_norm

    start_time = time.time()
    train_datasets, test_dataset = get_corona(args)

    ## User-defined model
    model = get_model(args)
    loss_fn = get_loss(args.loss_fn, args.loss_fn_name)
    metric = get_metric(args.metric, args.metric_name)

    print(
        "----------Loaded Datasets and Model----------Elapsed Time=",
        time.time() - start_time,
    )

    ## Running
    if args.num_clients == comm_size - 1:
        if comm_rank == 0:
            rms.run_server(
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
            rms.run_client(
                cfg, comm, model, loss_fn, train_datasets, test_dataset, metric
            )
    else:
        if comm_rank == 0:
            rm.run_server(
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
            rm.run_client(
                cfg,
                comm,
                model,
                loss_fn,
                args.num_clients,
                train_datasets,
                test_dataset,
                metric,
            )


if __name__ == "__main__":
    main()
