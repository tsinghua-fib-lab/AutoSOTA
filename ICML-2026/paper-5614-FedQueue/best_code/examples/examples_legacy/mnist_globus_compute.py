"""
python mnist_globus_compute.py --client_config path_to_client_config.yaml --server_config path_to_server_config.yaml --send-final-model
"""

import torch
import argparse
from omegaconf import OmegaConf
from globus_compute_sdk import Client
from appfl.misc.utils import set_seed
from appfl.config import GlobusComputeConfig
from appfl.config.utils import (
    load_globus_compute_client_config,
    load_globus_compute_server_config,
)
from appfl.run_globus_compute_server import run_server
from appfl.comm.globus_compute.utils.logging import GlobusComputeServerLogger
from appfl.comm.globus_compute.utils.utils import get_executable_func, get_loss_func

parser = argparse.ArgumentParser()
## Client config choices
parser.add_argument(
    "--client_config", type=str, default="globus_compute/configs_client/mnist_iid.yaml"
)
# parser.add_argument("--client_config", type=str, default="globus_compute/configs_client/mnist_class_noiid.yaml")
# parser.add_argument("--client_config", type=str, default="globus_compute/configs_client/mnist_dual_dirichlet_noiid.yaml")

## Server config choices
parser.add_argument(
    "--server_config",
    type=str,
    default="globus_compute/configs_server/mnist_fedavg.yaml",
)
# parser.add_argument("--server_config", type=str, default= "globus_compute/configs_server/mnist_fedavgm.yaml")
# parser.add_argument("--server_config", type=str, default= "globus_compute/configs_server/mnist_fedadam.yaml")
# parser.add_argument("--server_config", type=str, default= "globus_compute/configs_server/mnist_fedadagrad.yaml")
# parser.add_argument("--server_config", type=str, default= "globus_compute/configs_server/mnist_fedyogi.yaml")
# parser.add_argument("--server_config", type=str, default= "globus_compute/configs_server/mnist_fedasync.yaml")
# parser.add_argument("--server_config", type=str, default= "globus_compute/configs_server/mnist_fedbuffer.yaml")

# parser.add_argument("--server_config", type=str, default= "globus_compute/configs_server/mnist_fedavg_step_optim.yaml")
# parser.add_argument("--server_config", type=str, default= "globus_compute/configs_server/mnist_fedavgm_step_optim.yaml")
# parser.add_argument("--server_config", type=str, default= "globus_compute/configs_server/mnist_fedadam_step_optim.yaml")
# parser.add_argument("--server_config", type=str, default= "globus_compute/configs_server/mnist_fedadagrad_step_optim.yaml")
# parser.add_argument("--server_config", type=str, default= "globus_compute/configs_server/mnist_fedyogi_step_optim.yaml")
# parser.add_argument("--server_config", type=str, default= "globus_compute/configs_server/mnist_fedasync_step_optim.yaml")
# parser.add_argument("--server_config", type=str, default= "globus_compute/configs_server/mnist_fedbuffer_step_optim.yaml")
# parser.add_argument("--server_config", type=str, default= "globus_compute/configs_server/mnist_fedcompass_step_optim.yaml")

## other arguments
parser.add_argument("--reproduce", action="store_true", default=False)
parser.add_argument("--load-model", action="store_true", default=False)
parser.add_argument("--load-model-dirname", type=str, default="")
parser.add_argument("--load-model-filename", type=str, default="")
parser.add_argument("--use-tensorboard", action="store_true", default=False)
parser.add_argument("--save-model", action="store_true", default=False)
parser.add_argument("--save-model-state-dict", action="store_true", default=False)
parser.add_argument("--send-final-model", action="store_true", default=False)
parser.add_argument("--checkpoints-interval", type=float, default=2)

args = parser.parse_args()


def main():
    # Configuration
    cfg = OmegaConf.structured(GlobusComputeConfig)
    cfg.send_final_model = args.send_final_model
    cfg.reproduce = args.reproduce
    cfg.load_model = args.load_model
    cfg.load_model_dirname = args.load_model_dirname
    cfg.load_model_filename = args.load_model_filename
    cfg.save_model_state_dict = args.save_model_state_dict
    cfg.save_model = args.save_model
    cfg.checkpoints_interval = args.checkpoints_interval
    cfg.use_tensorboard = args.use_tensorboard
    cfg.validation = True
    if cfg.reproduce:
        set_seed(1)

    # loading globus compute configs from file
    load_globus_compute_client_config(cfg, args.client_config)
    load_globus_compute_server_config(cfg, args.server_config)

    # config logger
    GlobusComputeServerLogger.config_logger(cfg)

    """ Server-defined model """
    ModelClass = get_executable_func(cfg.get_model)()
    model = ModelClass(**cfg.model_kwargs)
    loss_fn = get_loss_func(cfg)
    val_metric = get_executable_func(cfg.val_metric)

    if cfg.load_model:
        path = cfg.load_model_dirname + "/{}{}".format(cfg.load_model_filename, ".pt")
        model.load_state_dict(torch.load(path))
        model.eval()

    server_test_dataset = None
    server_val_dataset = None
    gcc = Client()

    run_server(
        cfg, model, loss_fn, val_metric, gcc, server_test_dataset, server_val_dataset
    )


if __name__ == "__main__":
    main()
