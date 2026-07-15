import io
import os
import copy
import torch
import argparse
import warnings
import cProfile
import pstats
import appfl.run_mpi as rm
import appfl.run_serial as rs
from mpi4py import MPI
from omegaconf import OmegaConf
from appfl.config import Config
from appfl.misc.data import Dataset
from appfl.misc.utils import load_model_state, set_seed, load_model
from dataloader.nrel_dataloader import get_nrel
from losses.utils import get_loss
from metric.utils import get_metric
from models.utils import get_model, validate_parameter_names

warnings.filterwarnings("ignore", category=UserWarning)

## define functions for custom data type in argparses


def list_of_strings(arg):
    return arg.split(",")


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x!r} not a floating-point literal")

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x!r} not in range [0.0, 1.0]")
    return x


def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


## read arguments
parser = argparse.ArgumentParser()

## device
parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
)

## dataset
parser.add_argument("--dataset", type=str, default="NRELCA")
parser.add_argument("--n_features", type=int, default=8)
parser.add_argument("--n_lookback", type=int, default=12)
parser.add_argument("--n_lstm_layers", type=int, default=2)
parser.add_argument("--n_hidden_size", type=int, default=20)
parser.add_argument("--model", type=str, default="LSTM")
parser.add_argument("--train_test_boundary", type=restricted_float, default=0.8)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--enable_test", type=int, default=1)

## clients
parser.add_argument("--num_clients", type=int, default=1)
parser.add_argument("--client_optimizer", type=str, default="Adam")
parser.add_argument("--client_lr", type=float, default=1e-3)
parser.add_argument("--num_local_epochs", type=int, default=4)
parser.add_argument("--num_local_steps", type=int, default=4)

## server
parser.add_argument("--server", type=str, default="ServerFedAdam")
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--server_lr", type=float, required=False)
parser.add_argument("--mparam_1", type=float, required=False)
parser.add_argument("--mparam_2", type=float, required=False)
parser.add_argument("--adapt_param", type=float, required=False)

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


## model load and save
parser.add_argument("--save_model", type=int, default=1)
parser.add_argument("--save_every", type=int, default=250)
parser.add_argument("--load_model", type=int, default=0)
parser.add_argument("--load_model_suffix", type=str, default="")
parser.add_argument(
    "--dataset_dir", type=str, default=os.getcwd() + "/datasets/PreprocessedData"
)
parser.add_argument("--model_dir", type=str, default=os.getcwd())

## loss function and evaluation metric
parser.add_argument("--loss_fn", type=str, default="losses/mseloss.py")
parser.add_argument("--metric", type=str, default="metric/mae.py")

## personalization
parser.add_argument("--personalization_layers", type=list_of_strings, default=[])
parser.add_argument("--personalization_config_name", type=str, default="")
parser.add_argument("--opt_type", type=str, choices=["step", "epoch"], default="step")

## profile performance?
parser.add_argument("--profile_code", type=int, default=0)

# parse args
args = parser.parse_args()


def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # read default configuration
    cfg = OmegaConf.structured(Config)

    ## Reproducibility
    set_seed(1)

    ## Batch sizes
    cfg.train_data_batch_size = args.batch_size
    cfg.test_data_batch_size = args.batch_size

    train_datasets, test_dataset = get_nrel(args)

    # disable test according to argument
    if args.enable_test == 0:  # serial does support NOT having a test dset
        test_dataset = Dataset()

    ## Model
    model = get_model(args)
    loss_fn = get_loss(args.loss_fn)
    metric = get_metric(args.metric)

    ## If personalization is used, validate personalization
    args.personalization_layers = unique(args.personalization_layers)
    is_valid, is_empty = validate_parameter_names(model, args.personalization_layers)
    if not is_valid:
        raise TypeError(
            "The arguments containing names of personalization layers are invalid for the current model."
        )
    else:
        if not is_empty:
            cfg.personalization = True
            cfg.p_layers = args.personalization_layers
            cfg.config_name = args.personalization_config_name
        else:
            cfg.personalization = False
            cfg.p_layers = []

    ## clients
    if cfg.personalization:
        if args.opt_type == "step":
            cfg.fed.clientname = "PersonalizedClientStepOptim"
        else:
            cfg.fed.clientname = "PersonalizedClientOptim"
    cfg.num_clients = args.num_clients
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_epochs = args.num_local_epochs
    cfg.fed.args.num_local_steps = args.num_local_steps

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
    cfg.use_tensorboard = False

    ## save/load
    cfg.output_dirname = args.model_dir + "/outputs_{}_{}_{}_{}".format(
        args.dataset,
        args.server,
        args.client_optimizer,
        args.personalization_config_name,
    )
    if args.save_model:
        cfg.save_model_state_dict = True
        cfg.save_model = True
        cfg.save_model_dirname = (
            args.model_dir + "/save_models_NREL_%s" % args.personalization_config_name
        )
        cfg.save_model_filename = "model_{}_{}_{}".format(
            args.dataset,
            args.client_optimizer,
            args.server,
        )
        cfg.checkpoints_interval = args.save_every
    if args.load_model:
        cfg.load_model = True
        if cfg.personalization and comm_size > 1:  # personalization + parallel
            model_clients = [copy.deepcopy(model) for _ in range(args.num_clients)]
            cfg.load_model_dirname = (
                args.model_dir
                + "/save_models_NREL_%s" % args.personalization_config_name
            )
            cfg.load_model_filename = "model_{}_{}_{}_{}".format(
                args.dataset,
                args.client_optimizer,
                args.server,
                args.load_model_suffix,
            )
            load_model_state(cfg, model)
            for c_idx in range(args.num_clients):
                load_model_state(cfg, model_clients[c_idx], client_id=c_idx)
        elif cfg.personalization and comm_size == 1:  # personalization + serial
            model_clients = [copy.deepcopy(model) for _ in range(args.num_clients + 1)]
            cfg.load_model_dirname = (
                args.model_dir
                + "/save_models_NREL_%s" % args.personalization_config_name
            )
            cfg.load_model_filename = "model_{}_{}_{}_{}".format(
                args.dataset,
                args.client_optimizer,
                args.server,
                args.load_model_suffix,
            )
            load_model_state(cfg, model[0])
            for c_idx in range(args.num_clients):
                load_model_state(cfg, model_clients[c_idx + 1], client_id=c_idx)
        else:  # no personalization
            cfg.save_model_dirname = (
                args.model_dir
                + "/save_models_NREL_%s" % args.personalization_config_name
            )
            cfg.save_model_filename = "model_{}_{}_{}".format(
                args.dataset,
                args.client_optimizer,
                args.server,
            )
            model = load_model(cfg)
    else:
        if cfg.personalization:
            model_clients = [copy.deepcopy(model) for _ in range(args.num_clients)]
        else:
            pass  # model with empty weights is already loaded

    if comm_size > 1:
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
            if cfg.personalization:
                model_to_send = model_clients
            else:
                model_to_send = model
            rm.run_client(
                cfg,
                comm,
                model_to_send,
                loss_fn,
                args.num_clients,
                train_datasets,
                test_dataset,
                metric,
            )
        print("------DONE------", comm_rank)
    else:
        if cfg.personalization:
            model_to_send = [model] + model_clients
        else:
            model_to_send = model
        rs.run_serial(
            cfg,
            model_to_send,
            loss_fn,
            train_datasets,
            test_dataset,
            args.dataset,
            metric,
        )


if __name__ == "__main__":
    if not args.profile_code:
        main()
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        if not os.path.isdir(os.getcwd() + "/code_profile"):
            try:
                os.mkdir(os.getcwd() + "/code_profile")
            except:  # noqa E722
                pass
        with cProfile.Profile() as pr:
            main()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        ps.print_stats()
        with open(os.getcwd() + "/code_profile/rank_%d.txt" % rank, "w") as f:
            f.write(s.getvalue())
